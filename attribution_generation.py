import torch
import torch.nn.functional as F
from direct_decoder import DirectDiffusion
from model import MNISTDiffusion
from torchvision.utils import save_image
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with attribution control")
    parser.add_argument('--ddpm_ckpt', type=str, required=True, help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--decoder_ckpt', type=str, required=True, help='Path to trained direct decoder checkpoint')
    parser.add_argument('--ref_features_path', type=str, required=True, help='Path to reference features')
    parser.add_argument('--n_samples', type=int, default=36, help='Number of samples to generate')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of UNet')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps')
    parser.add_argument('--output_dir', type=str, default='attribution_outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    return parser.parse_args()


def create_include_hook(ref_features, sample_indices, strength=0.5):
    """
    Create hook to push generated features toward specific training samples.

    Args:
        ref_features: Reference features [N, 128, 7, 7]
        sample_indices: List of sample indices to include
        strength: How strongly to push (0.0 = no effect, 1.0 = full replacement)

    Returns:
        Hook function
    """
    target_features = ref_features[sample_indices].mean(dim=0, keepdim=True)  # [1, 128, 7, 7]

    def hook(features):
        # Move target to same device
        target = target_features.to(features.device)
        # Push toward target
        direction = target - features
        return features + strength * direction

    return hook


def create_exclude_hook(ref_features, sample_indices, strength=0.5):
    """
    Create hook to push generated features away from specific training samples.

    Args:
        ref_features: Reference features [N, 128, 7, 7]
        sample_indices: List of sample indices to exclude
        strength: How strongly to push away

    Returns:
        Hook function
    """
    exclude_features = ref_features[sample_indices].mean(dim=0, keepdim=True)  # [1, 128, 7, 7]

    def hook(features):
        # Move exclude to same device
        exclude = exclude_features.to(features.device)
        # Push away from exclude
        direction = features - exclude
        # Normalize direction and apply strength
        direction_norm = F.normalize(direction.view(features.size(0), -1), dim=1)
        direction_norm = direction_norm.view_as(features)
        return features + strength * direction_norm

    return hook


def create_class_conditional_hook(ref_features, ref_labels, target_class, strength=0.5):
    """
    Create hook to generate images of a specific class.

    Args:
        ref_features: Reference features [N, 128, 7, 7]
        ref_labels: Reference labels [N]
        target_class: Target digit class (0-9)
        strength: How strongly to condition

    Returns:
        Hook function
    """
    # Get mean features for target class
    class_mask = (ref_labels == target_class)
    class_features = ref_features[class_mask].mean(dim=0, keepdim=True)  # [1, 128, 7, 7]

    def hook(features):
        target = class_features.to(features.device)
        direction = target - features
        return features + strength * direction

    return hook


def measure_attribution(generated_features, ref_features, top_k=10):
    """
    Measure which training samples are most similar to generated samples.

    Args:
        generated_features: Features from generated images [B, 128, 7, 7]
        ref_features: Reference features [N, 128, 7, 7]
        top_k: Number of top similar samples to return

    Returns:
        similarities: Similarity scores [B, N]
        top_indices: Top-k sample indices [B, top_k]
    """
    # Flatten features
    gen_flat = generated_features.view(generated_features.size(0), -1)  # [B, 128*7*7]
    ref_flat = ref_features.view(ref_features.size(0), -1)              # [N, 128*7*7]

    # Compute cosine similarity
    gen_norm = F.normalize(gen_flat, dim=1)
    ref_norm = F.normalize(ref_flat, dim=1)

    similarities = torch.mm(gen_norm, ref_norm.t())  # [B, N]

    # Get top-k most similar
    top_similarities, top_indices = torch.topk(similarities, k=top_k, dim=1)

    return similarities, top_indices, top_similarities


def main(args):
    torch.manual_seed(args.seed)

    # Device setup
    if args.cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load pretrained DDPM
    print(f"Loading pretrained DDPM from {args.ddpm_ckpt}...")
    ddpm = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if 'model_ema' in ckpt:
        # Remove 'module.' prefix from EMA keys
        ema_state = ckpt['model_ema']
        state_dict = {}
        for k, v in ema_state.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            elif k != 'n_averaged':
                state_dict[k] = v
        ddpm.load_state_dict(state_dict)
    else:
        ddpm.load_state_dict(ckpt['model'])

    # Create DirectDiffusion model
    print("Creating DirectDiffusion model...")
    model = DirectDiffusion(
        pretrained_unet=ddpm.model,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    # Load trained decoder weights
    print(f"Loading trained decoder from {args.decoder_ckpt}...")
    decoder_ckpt = torch.load(args.decoder_ckpt, map_location=device)

    if 'model_ema' in decoder_ckpt:
        # EMA state is OrderedDict, need to remove 'module.' prefix
        ema_state = decoder_ckpt['model_ema']
        state_dict = {}
        for k, v in ema_state.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            elif k != 'n_averaged':
                state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(decoder_ckpt['model'])

    model.eval()

    # Load reference features
    print(f"Loading reference features from {args.ref_features_path}...")
    ref_data = torch.load(args.ref_features_path, map_location='cpu')
    ref_features = ref_data['features']  # [N, 128, 7, 7]
    ref_labels = ref_data['labels']      # [N]

    print(f"Loaded {ref_features.size(0)} reference features")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate random noise
    noise = torch.randn(args.n_samples, 1, 28, 28).to(device)

    # 1. Baseline generation (no hook)
    print("\n1. Baseline generation (no attribution control)...")
    with torch.no_grad():
        samples_baseline = model(noise, hook_fn=None)

    save_image(
        samples_baseline,
        os.path.join(args.output_dir, 'samples_baseline.png'),
        nrow=6,
        normalize=True
    )

    # 2. Include specific samples (e.g., first 10 samples)
    print("2. Include samples [0-9] (push toward these samples)...")
    include_indices = list(range(10))
    include_hook = create_include_hook(ref_features, include_indices, strength=0.5)

    with torch.no_grad():
        samples_include = model(noise, hook_fn=include_hook)

    save_image(
        samples_include,
        os.path.join(args.output_dir, 'samples_include.png'),
        nrow=6,
        normalize=True
    )

    # 3. Exclude specific samples (e.g., samples 100-109)
    print("3. Exclude samples [100-109] (push away from these samples)...")
    exclude_indices = list(range(100, 110))
    exclude_hook = create_exclude_hook(ref_features, exclude_indices, strength=0.5)

    with torch.no_grad():
        samples_exclude = model(noise, hook_fn=exclude_hook)

    save_image(
        samples_exclude,
        os.path.join(args.output_dir, 'samples_exclude.png'),
        nrow=6,
        normalize=True
    )

    # 4. Class-conditional generation (e.g., generate digit 3)
    print("4. Class-conditional generation (target class: 3)...")
    class_hook = create_class_conditional_hook(ref_features, ref_labels, target_class=3, strength=0.7)

    with torch.no_grad():
        samples_class = model(noise, hook_fn=class_hook)

    save_image(
        samples_class,
        os.path.join(args.output_dir, 'samples_class_3.png'),
        nrow=6,
        normalize=True
    )

    # 5. Measure attribution for baseline samples
    print("\n5. Measuring attribution for baseline samples...")
    with torch.no_grad():
        gen_features, _ = model.encoder(noise)

    similarities, top_indices, top_scores = measure_attribution(
        gen_features.cpu(),
        ref_features,
        top_k=5
    )

    print("\nTop-5 most similar training samples for each generated sample:")
    for i in range(min(5, args.n_samples)):  # Show first 5 generated samples
        print(f"\nGenerated sample {i}:")
        for j in range(5):
            idx = top_indices[i, j].item()
            score = top_scores[i, j].item()
            label = ref_labels[idx].item()
            print(f"  Rank {j+1}: Sample {idx:5d} (label={label}, similarity={score:.4f})")

    # Save attribution results
    torch.save({
        'similarities': similarities,
        'top_indices': top_indices,
        'top_scores': top_scores,
        'generated_features': gen_features.cpu(),
        'noise_seeds': noise.cpu(),
    }, os.path.join(args.output_dir, 'attribution_results.pt'))

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
