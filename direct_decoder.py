import torch
import torch.nn as nn
from unet import ResidualBottleneck, DecoderBlock


class DirectEncoder(nn.Module):
    """
    Wrapper around pretrained Unet encoder to extract bottleneck features and skip connections.
    No timestep conditioning (t=None).
    """
    def __init__(self, pretrained_unet):
        super().__init__()
        # Extract encoder components from pretrained Unet
        self.init_conv = pretrained_unet.init_conv
        self.encoder_blocks = pretrained_unet.encoder_blocks
        self.mid_block = pretrained_unet.mid_block

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input noise tensor [B, 1, 28, 28]

        Returns:
            bottleneck_features: [B, 128, 7, 7]
            encoder_shortcuts: List of skip connection tensors
        """
        x = self.init_conv(x)

        # Collect skip connections (no timestep conditioning)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t=None)  # t=None for timestep-agnostic
            encoder_shortcuts.append(x_shortcut)

        # Process through mid block
        bottleneck_features = self.mid_block(x)

        # Reverse shortcuts to match decoder order
        encoder_shortcuts.reverse()

        return bottleneck_features, encoder_shortcuts


class DirectDecoder(nn.Module):
    """
    Direct decoder that maps from encoder features to clean images.
    Similar to Unet decoder but WITHOUT timestep conditioning (no TimeMLP).
    """
    def __init__(self, base_dim=32, dim_mults=[2, 4], in_channels=1):
        super().__init__()

        self.channels = self._cal_channels(base_dim, dim_mults)

        # Build decoder blocks (similar to Unet but no time_embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            SimpleDecoderBlock(c[1], c[0]) for c in self.channels[::-1]
        ])

        # Final convolution to image
        self.final_conv = nn.Conv2d(
            in_channels=self.channels[0][0] // 2,
            out_channels=in_channels,
            kernel_size=1
        )

    def forward(self, bottleneck_features, encoder_shortcuts, hook_fn=None):
        """
        Args:
            bottleneck_features: [B, 128, 7, 7] from encoder mid_block
            encoder_shortcuts: List of skip connection tensors
            hook_fn: Optional function to manipulate bottleneck_features
                     Signature: hook_fn(features) -> modified_features

        Returns:
            x: Generated image [B, 1, 28, 28] in range [0, 1]
        """
        x = bottleneck_features

        # Apply hook if provided (for feature space manipulation)
        if hook_fn is not None:
            x = hook_fn(x)

        # Decode with skip connections
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut)

        # Final convolution
        x = self.final_conv(x)

        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))
        return channels


class SimpleDecoderBlock(nn.Module):
    """
    Decoder block WITHOUT timestep conditioning (no TimeMLP).
    Mirrors DecoderBlock from unet.py but simplified.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Conv blocks (same as DecoderBlock but no TimeMLP)
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
            ResidualBottleneck(in_channels, in_channels // 2)
        )

        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

    def forward(self, x, x_shortcut):
        """
        Args:
            x: Input features
            x_shortcut: Skip connection from encoder

        Returns:
            x: Upsampled and processed features
        """
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        # No TimeMLP here (difference from original DecoderBlock)
        x = self.conv1(x)
        return x


class DirectDiffusion(nn.Module):
    """
    Combined model: Frozen encoder + trainable direct decoder.
    This is the main model for training and inference.
    """
    def __init__(self, pretrained_unet, base_dim=32, dim_mults=[2, 4]):
        super().__init__()

        # Frozen encoder
        self.encoder = DirectEncoder(pretrained_unet)

        # Trainable decoder
        self.decoder = DirectDecoder(base_dim=base_dim, dim_mults=dim_mults, in_channels=1)

    def forward(self, noise, hook_fn=None):
        """
        Args:
            noise: Random noise [B, 1, 28, 28]
            hook_fn: Optional hook for feature manipulation

        Returns:
            Generated image [B, 1, 28, 28] in range [0, 1]
        """
        # Encode (frozen)
        with torch.no_grad():
            bottleneck_features, encoder_shortcuts = self.encoder(noise)

        # Decode (trainable, with optional hook)
        output = self.decoder(bottleneck_features, encoder_shortcuts, hook_fn=hook_fn)

        return output

    def load_decoder_from_pretrained(self, pretrained_unet):
        """
        Initialize decoder weights from pretrained Unet decoder.
        This provides a warm start for training.
        """
        # Copy decoder block weights
        for i, (new_block, old_block) in enumerate(zip(
            self.decoder.decoder_blocks,
            pretrained_unet.decoder_blocks
        )):
            # Copy conv0 and conv1 weights
            new_block.conv0.load_state_dict(old_block.conv0.state_dict())
            new_block.conv1.load_state_dict(old_block.conv1.state_dict())
            # Note: old_block.time_mlp is not copied (we don't have it)

        # Copy final conv
        self.decoder.final_conv.load_state_dict(pretrained_unet.final_conv.state_dict())

        print("Initialized DirectDecoder from pretrained Unet decoder weights")


if __name__ == "__main__":
    # Test the architecture
    from model import MNISTDiffusion

    print("Creating pretrained DDPM...")
    ddpm = MNISTDiffusion(timesteps=1000, image_size=28, in_channels=1, base_dim=64, dim_mults=[2, 4])

    print("Creating DirectDiffusion model...")
    direct_model = DirectDiffusion(ddpm.model, base_dim=64, dim_mults=[2, 4])

    print("Loading pretrained decoder weights...")
    direct_model.load_decoder_from_pretrained(ddpm.model)

    print("\nTesting forward pass...")
    noise = torch.randn(4, 1, 28, 28)
    output = direct_model(noise)

    print(f"Input shape: {noise.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\nTesting with hook...")
    def example_hook(features):
        # Example: scale features
        return features * 1.1

    output_hooked = direct_model(noise, hook_fn=example_hook)
    print(f"Hooked output shape: {output_hooked.shape}")

    print("\nArchitecture test passed!")
