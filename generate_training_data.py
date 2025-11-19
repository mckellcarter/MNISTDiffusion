import torch
from model import MNISTDiffusion
import os
import glob
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate training data for direct decoder")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--n_samples', type=int, default=60000, help='Number of (noise, image) pairs to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='direct_decoder_data', help='Output directory')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of UNet')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')

    return parser.parse_args()

def main(args):
    # Set seed for reproducibility
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
    print(f"Loading checkpoint from {args.ckpt}")
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # Use EMA model for better quality
    if 'model_ema' in ckpt:
        # Remove 'module.' prefix from EMA keys
        ema_state = ckpt['model_ema']
        state_dict = {}
        for k, v in ema_state.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v  # Remove 'module.' prefix
            elif k != 'n_averaged':  # Skip EMA metadata
                state_dict[k] = v
        model.load_state_dict(state_dict)
        print("Loaded EMA model")
    else:
        model.load_state_dict(ckpt['model'])
        print("Loaded standard model")

    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate data in batches
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

    all_noise = []
    all_images = []

    print(f"Generating {args.n_samples} samples in {n_batches} batches...")

    for i in tqdm(range(n_batches), desc="Generating batches"):
        # Determine batch size (last batch may be smaller)
        current_batch_size = min(args.batch_size, args.n_samples - i * args.batch_size)

        # Generate random noise seeds
        noise_seeds = torch.randn((current_batch_size, 1, 28, 28)).to(device)

        # Generate images using DDPM (1000 steps)
        with torch.no_grad():
            generated_images, _ = model.sampling(
                current_batch_size,
                clipped_reverse_diffusion=True,
                device=device,
                noise_seed=noise_seeds
            )

        # Store (move to CPU to save memory)
        all_noise.append(noise_seeds.cpu())
        all_images.append(generated_images.cpu())

        # Periodically save to avoid memory issues
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"Saving checkpoint at batch {i+1}/{n_batches}...")
            noise_tensor = torch.cat(all_noise, dim=0)
            images_tensor = torch.cat(all_images, dim=0)

            torch.save({
                'noise': noise_tensor,
                'images': images_tensor,
                'n_samples': len(noise_tensor),
                'metadata': {
                    'ckpt': args.ckpt,
                    'timesteps': args.timesteps,
                    'seed': args.seed,
                }
            }, os.path.join(args.output_dir, f'training_data_checkpoint_{i+1}.pt'))

    # Final concatenation and save
    print("Concatenating all data...")
    final_noise = torch.cat(all_noise, dim=0)[:args.n_samples]
    final_images = torch.cat(all_images, dim=0)[:args.n_samples]

    print(f"Final dataset shape: noise={final_noise.shape}, images={final_images.shape}")

    # Save final dataset
    output_path = os.path.join(args.output_dir, 'training_data.pt')
    torch.save({
        'noise': final_noise,
        'images': final_images,
        'n_samples': args.n_samples,
        'metadata': {
            'ckpt': args.ckpt,
            'timesteps': args.timesteps,
            'seed': args.seed,
        }
    }, output_path)

    print(f"Saved {args.n_samples} pairs to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")

    # Clean up checkpoint files
    print("\nCleaning up intermediate checkpoint files...")
    checkpoint_pattern = os.path.join(args.output_dir, 'training_data_checkpoint_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)

    if checkpoint_files:
        total_size = sum(os.path.getsize(f) for f in checkpoint_files) / 1024**3
        for ckpt_file in checkpoint_files:
            os.remove(ckpt_file)
        print(f"Removed {len(checkpoint_files)} checkpoint files ({total_size:.2f} GB freed)")
    else:
        print("No checkpoint files found to clean up")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
