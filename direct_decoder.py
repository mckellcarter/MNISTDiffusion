import torch
import torch.nn as nn
from unet import ResidualBottleneck, TimeMLP


class DirectEncoder(nn.Module):
    """
    Frozen encoder from pretrained DDPM UNet.
    Processes input WITH timestep conditioning (uses pretrained time conditioning).
    """
    def __init__(self, pretrained_unet):
        super().__init__()
        self.init_conv = pretrained_unet.init_conv
        self.time_embedding = pretrained_unet.time_embedding
        self.encoder_blocks = pretrained_unet.encoder_blocks
        self.mid_block = pretrained_unet.mid_block

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, t):
        """
        Args:
            x: Input noisy images [B, 1, 28, 28]
            t: Timestep tensor [B]
        Returns:
            bottleneck_features: Features at bottleneck [B, 256, 7, 7]
            encoder_shortcuts: Skip connections from each encoder block
        """
        x = self.init_conv(x)
        t_emb = self.time_embedding(t)  # Create timestep embeddings using pretrained embedding

        # Collect skip connections (WITH timestep conditioning)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t=t_emb)  # Pass timestep embeddings
            encoder_shortcuts.append(x_shortcut)

        # Process through mid block
        bottleneck_features = self.mid_block(x)

        # Reverse shortcuts to match decoder order
        encoder_shortcuts.reverse()

        return bottleneck_features, encoder_shortcuts


class DirectDecoder(nn.Module):
    """
    Direct decoder WITH timestep conditioning.
    Maps from encoder features to denoised images, conditioned on current timestep.
    """
    def __init__(self, base_dim=32, dim_mults=[2, 4], in_channels=1, time_embedding_dim=256, max_timesteps=1000):
        super().__init__()

        self.time_embedding_dim = time_embedding_dim
        self.max_timesteps = max_timesteps

        # Timestep embedding layer (sinusoidal embeddings)
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        self.channels = self._cal_channels(base_dim, dim_mults)

        # Build decoder blocks WITH timestep conditioning
        # Note: decoder blocks go from high channels to low, so we swap in/out when reversing
        self.decoder_blocks = nn.ModuleList([
            DirectDecoderBlock(out_channels, in_channels, time_embedding_dim)
            for in_channels, out_channels in self.channels[::-1]
        ])

        # Final convolution (from base_dim//2 channels to output channels)
        self.final_conv = nn.Conv2d(base_dim // 2, in_channels, kernel_size=1)

    def forward(self, bottleneck_features, encoder_shortcuts, t):
        """
        Args:
            bottleneck_features: Features from encoder [B, 256, 7, 7]
            encoder_shortcuts: Skip connections from encoder (reversed order)
            t: Timestep tensor [B]
        Returns:
            x: Denoised image [B, 1, 28, 28]
        """
        # Create timestep embeddings
        t_emb = self._get_timestep_embedding(t)  # [B, time_embedding_dim]
        t_emb = self.time_embedding(t_emb)       # [B, time_embedding_dim]

        x = bottleneck_features

        # Process through decoder blocks with timestep conditioning
        for idx, decoder_block in enumerate(self.decoder_blocks):
            x_shortcut = encoder_shortcuts[idx]
            x = decoder_block(x, x_shortcut, t_emb)

        # Final convolution
        x = self.final_conv(x)

        return x

    def _get_timestep_embedding(self, timesteps):
        """
        Create sinusoidal timestep embeddings.
        Args:
            timesteps: Tensor of shape [B] containing timestep values
        Returns:
            embedding: Tensor of shape [B, time_embedding_dim]
        """
        device = timesteps.device
        half_dim = self.time_embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))
        return channels


class DirectDecoderBlock(nn.Module):
    """
    Decoder block WITH timestep conditioning (includes TimeMLP).
    Matches DecoderBlock from unet.py exactly.
    """
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Conv blocks - matching original DecoderBlock
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
            ResidualBottleneck(in_channels, in_channels // 2)
        )

        # Timestep conditioning
        self.time_mlp = TimeMLP(
            embedding_dim=time_embedding_dim,
            hidden_dim=in_channels,
            out_dim=in_channels // 2
        )

        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

    def forward(self, x, x_shortcut, t_emb):
        """
        Args:
            x: Input features [B, C_in, H, W]
            x_shortcut: Skip connection from encoder [B, C_in, H, W]
            t_emb: Timestep embedding [B, time_embedding_dim]
        Returns:
            x: Upsampled and processed features [B, C_out, 2H, 2W]
        """
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)

        # Apply timestep conditioning
        x = self.time_mlp(x, t_emb)

        x = self.conv1(x)
        return x


class DirectDiffusion(nn.Module):
    """
    Complete direct decoder model: frozen encoder + trainable timestep-conditioned decoder.
    Both encoder and decoder receive timestep information.
    """
    def __init__(self, pretrained_unet, base_dim=32, dim_mults=[2, 4], time_embedding_dim=256, max_timesteps=1000):
        super().__init__()

        # Frozen encoder (but still uses timestep conditioning)
        self.encoder = DirectEncoder(pretrained_unet)

        # Trainable decoder with timestep conditioning
        self.decoder = DirectDecoder(
            base_dim=base_dim,
            dim_mults=dim_mults,
            in_channels=1,
            time_embedding_dim=time_embedding_dim,
            max_timesteps=max_timesteps
        )

        # Hook for feature manipulation (attribution analysis)
        self.hook_fn = None

    def forward(self, x, t):
        """
        Args:
            x: Noisy input images [B, 1, 28, 28]
            t: Current timestep [B]
        Returns:
            x_denoised: Partially denoised images [B, 1, 28, 28]
        """
        # Encode with timestep conditioning (frozen)
        bottleneck_features, encoder_shortcuts = self.encoder(x, t)

        # Apply hook if registered (for attribution analysis)
        if self.hook_fn is not None:
            bottleneck_features = self.hook_fn(bottleneck_features)

        # Decode with timestep conditioning (trainable)
        x_denoised = self.decoder(bottleneck_features, encoder_shortcuts, t)

        return x_denoised

    def register_hook(self, hook_fn):
        """Register a hook function to manipulate bottleneck features."""
        self.hook_fn = hook_fn

    def clear_hook(self):
        """Clear the registered hook."""
        self.hook_fn = None

    def load_decoder_from_pretrained(self, pretrained_unet):
        """
        Initialize decoder weights from pretrained UNet decoder (warm start).
        """
        print("Initializing decoder from pretrained UNet decoder weights...")

        # Copy decoder block weights
        for i, (direct_block, pretrained_block) in enumerate(
            zip(self.decoder.decoder_blocks, pretrained_unet.decoder_blocks)
        ):
            # Copy conv0
            direct_block.conv0.load_state_dict(pretrained_block.conv0.state_dict())

            # Copy time_mlp
            direct_block.time_mlp.load_state_dict(pretrained_block.time_mlp.state_dict())

            # Copy conv1
            direct_block.conv1.load_state_dict(pretrained_block.conv1.state_dict())

        # Copy final conv
        self.decoder.final_conv.load_state_dict(pretrained_unet.final_conv.state_dict())

        print("Decoder initialization complete!")
