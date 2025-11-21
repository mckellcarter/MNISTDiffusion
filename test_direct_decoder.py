import argparse
import os

import torch
from torchvision.utils import save_image

from direct_decoder import DirectDiffusion
from model import MNISTDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Test Direct Decoder with Feature Manipulation")
    parser.add_argument('--ddpm_ckpt', type=str, required=True, help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--decoder_ckpt', type=str, required=True, help='Path to trained direct decoder checkpoint')
    parser.add_argument('--n_samples', type=int, default=36, help='Number of samples to generate')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of UNet')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps')
    parser.add_argument('--output_dir', type=str, default='test_outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    return parser.parse_args()


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
                state_dict[k[7:]] = v  # Remove 'module.' prefix
            elif k != 'n_averaged':  # Skip EMA metadata
                state_dict[k] = v
        model.load_state_dict(state_dict)
        print("Loaded EMA decoder weights")
    else:
        model.load_state_dict(decoder_ckpt['model'])
        print("Loaded standard decoder weights")

    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples with different hook configurations
    print(f"\nGenerating {args.n_samples} samples...")

    # 1. Standard generation (no hook)
    print("1. Standard generation (no hook)...")
    noise = torch.randn(args.n_samples, 1, 28, 28).to(device)
    # Use t=999 (0-indexed max timestep for pure noise)
    t = torch.full((args.n_samples,), 999, device=device, dtype=torch.long)

    with torch.no_grad():
        samples_standard = model(noise, t)

    save_image(
        samples_standard,
        os.path.join(args.output_dir, 'samples_standard.png'),
        nrow=6,
        normalize=True
    )
    print(f"   Saved to {args.output_dir}/samples_standard.png")

    # 2. Feature scaling hook (amplify features)
    print("2. Feature amplification hook (1.2x)...")
    def amplify_hook(features):
        return features * 1.2

    model.register_hook(amplify_hook)
    with torch.no_grad():
        samples_amplified = model(noise, t)
    model.clear_hook()

    save_image(
        samples_amplified,
        os.path.join(args.output_dir, 'samples_amplified.png'),
        nrow=6,
        normalize=True
    )
    print(f"   Saved to {args.output_dir}/samples_amplified.png")

    # 3. Feature dampening hook
    print("3. Feature dampening hook (0.8x)...")
    def dampen_hook(features):
        return features * 0.8

    model.register_hook(dampen_hook)
    with torch.no_grad():
        samples_dampened = model(noise, t)
    model.clear_hook()

    save_image(
        samples_dampened,
        os.path.join(args.output_dir, 'samples_dampened.png'),
        nrow=6,
        normalize=True
    )
    print(f"   Saved to {args.output_dir}/samples_dampened.png")

    # 4. Feature direction manipulation (example: add bias)
    print("4. Feature bias hook (add constant)...")
    def bias_hook(features):
        return features + 0.1

    model.register_hook(bias_hook)
    with torch.no_grad():
        samples_biased = model(noise, t)
    model.clear_hook()

    save_image(
        samples_biased,
        os.path.join(args.output_dir, 'samples_biased.png'),
        nrow=6,
        normalize=True
    )
    print(f"   Saved to {args.output_dir}/samples_biased.png")

    # 5. Noise interpolation example
    print("5. Noise interpolation...")
    noise1 = torch.randn(1, 1, 28, 28).to(device)
    noise2 = torch.randn(1, 1, 28, 28).to(device)
    t_interp = torch.full((1,), 999, device=device, dtype=torch.long)

    interpolation_steps = 10
    alphas = torch.linspace(0, 1, interpolation_steps)

    interpolated_samples = []
    for alpha in alphas:
        noise_interp = (1 - alpha) * noise1 + alpha * noise2
        with torch.no_grad():
            sample = model(noise_interp, t_interp)
        interpolated_samples.append(sample)

    interpolated_grid = torch.cat(interpolated_samples, dim=0)
    save_image(
        interpolated_grid,
        os.path.join(args.output_dir, 'samples_interpolation.png'),
        nrow=interpolation_steps,
        normalize=True
    )
    print(f"   Saved to {args.output_dir}/samples_interpolation.png")

    print("\nTesting completed! Check output directory for results.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
