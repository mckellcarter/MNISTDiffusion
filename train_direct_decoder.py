import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from direct_decoder import DirectDiffusion
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
from torchvision.utils import save_image
import os
import math
import argparse
import random
import numpy as np


def reset_rand(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training Direct Decoder")
    parser.add_argument('--ddpm_ckpt', type=str, required=True, help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to pre-generated training data')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/val split ratio')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of UNet')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps (for loading model)')
    parser.add_argument('--model_ema_steps', type=int, default=10, help='EMA update interval')
    parser.add_argument('--model_ema_decay', type=float, default=0.995, help='EMA decay rate')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--n_samples', type=int, default=36, help='Number of samples to generate per epoch')
    parser.add_argument('--output_dir', type=str, default='direct_decoder_results', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')

    return parser.parse_args()


def create_dataloaders(data_path, batch_size, train_split=0.9, num_workers=4):
    """
    Load pre-generated training data and split into train/val.
    """
    print(f"Loading training data from {data_path}...")
    data = torch.load(data_path)

    noise = data['noise']
    images = data['images']
    n_samples = data['n_samples']

    print(f"Loaded {n_samples} samples")
    print(f"Noise shape: {noise.shape}, range: [{noise.min():.3f}, {noise.max():.3f}]")
    print(f"Images shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")

    # Create dataset
    dataset = TensorDataset(noise, images)

    # Split into train/val
    train_size = int(train_split * n_samples)
    val_size = n_samples - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def main(args):
    reset_rand(args.seed)

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
        print("Loaded DDPM EMA weights")
    else:
        ddpm.load_state_dict(ckpt['model'])
        print("Loaded DDPM standard weights")

    # Create DirectDiffusion model
    print("Creating DirectDiffusion model...")
    model = DirectDiffusion(
        pretrained_unet=ddpm.model,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    # Initialize decoder from pretrained weights (warm start)
    model.load_decoder_from_pretrained(ddpm.model)

    # Setup EMA
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    # Load data
    train_loader, val_loader = create_dataloaders(
        args.data_path,
        args.batch_size,
        args.train_split
    )

    # Optimizer and scheduler
    # Only optimize decoder parameters (encoder is frozen)
    optimizer = AdamW(model.decoder.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.25,
        anneal_strategy='cos'
    )

    loss_fn = nn.MSELoss(reduction='mean')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    global_steps = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0

        for i, (noise, target_images) in enumerate(train_loader):
            noise = noise.to(device)
            target_images = target_images.to(device)

            # Forward pass
            pred_images = model(noise)

            # Loss
            loss = loss_fn(pred_images, target_images)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update EMA
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)

            train_loss += loss.item()
            global_steps += 1

            # Logging
            if i % args.log_freq == 0:
                print(f"Epoch[{epoch+1}/{args.epochs}], Step[{i}/{len(train_loader)}], "
                      f"Loss:{loss.item():.5f}, LR:{scheduler.get_last_lr()[0]:.6f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noise, target_images in val_loader:
                noise = noise.to(device)
                target_images = target_images.to(device)

                pred_images = model(noise)
                loss = loss_fn(pred_images, target_images)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.5f}")
        print(f"  Val Loss:   {val_loss:.5f}\n")

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_steps": global_steps,
            "val_loss": val_loss
        }

        torch.save(ckpt, os.path.join(args.output_dir, f'ckpt_epoch_{epoch+1:03d}.pt'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  Saved best model (val_loss: {val_loss:.5f})")

        # Generate samples using EMA model
        model_ema.eval()
        sample_noise = torch.randn(args.n_samples, 1, 28, 28).to(device)

        with torch.no_grad():
            samples = model_ema.module(sample_noise)

        save_image(
            samples,
            os.path.join(args.output_dir, f'samples_epoch_{epoch+1:03d}.png'),
            nrow=int(math.sqrt(args.n_samples)),
            normalize=True
        )

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
