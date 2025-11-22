import argparse
import math
import os
import random

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

from direct_decoder import DirectDiffusion
from model import MNISTDiffusion
from utils import ExponentialMovingAverage


def reset_rand(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Use warn_only=True to allow training on CUDA where some ops can't be deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training Direct Decoder")
    parser.add_argument('--ddpm_ckpt', type=str, required=True,
                        help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to pre-generated training data')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/val split ratio')
    parser.add_argument('--model_base_dim', type=int, default=64,
                        help='Base dim of UNet')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='DDPM timesteps (for loading model)')
    parser.add_argument('--model_ema_steps', type=int, default=10,
                        help='EMA update interval')
    parser.add_argument('--model_ema_decay', type=float, default=0.995,
                        help='EMA decay rate')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Logging frequency')
    parser.add_argument('--n_samples', type=int, default=36,
                        help='Number of samples to generate per epoch')
    parser.add_argument('--output_dir', type=str, default='direct_decoder_results',
                        help='Output directory')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--seed', type=int, default=8675309,
                        help='Random seed')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--loss_huber_weight', type=float, default=0.1,
                        help='Weight for Huber loss component')
    parser.add_argument('--loss_lpips_weight', type=float, default=1.0,
                        help='Weight for LPIPS loss component')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss (threshold between L1/L2)')
    parser.add_argument('--lpips_net', type=str, default='alex', choices=['alex', 'vgg'],
                        help='Backbone network for LPIPS (alex or vgg)')

    return parser.parse_args()


def default_collate_fn(batch):
    """
    Default collate function - just stack tensors.
    """
    intermediate_states, images = zip(*batch)
    intermediate_states = torch.stack(intermediate_states)
    images = torch.stack(images)
    return intermediate_states, images


def create_dataloaders(data_path, batch_size, train_split=0.9, num_workers=4):
    """
    Load pre-generated training data and split into train/val.
    """
    print(f"Loading training data from {data_path}...")
    data = torch.load(data_path)

    noise = data['noise']
    images = data['images']
    intermediate_states = data['intermediate_states']  # [N, 5, 1, 28, 28]
    n_samples = data['n_samples']
    target_timesteps_raw = data['metadata']['target_timesteps']  # [1000, 500, 100, 50, 10]

    # Convert to 0-indexed (nn.Embedding expects [0, timesteps-1])
    # [999, 499, 99, 49, 9]
    target_timesteps = [t - 1 if t > 0 else 0 for t in target_timesteps_raw]

    print(f"Loaded {n_samples} samples")
    print(f"Noise shape: {noise.shape}, range: [{noise.min():.3f}, {noise.max():.3f}]")
    print(f"Images shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Intermediate states shape: {intermediate_states.shape} at timesteps "
          f"{target_timesteps_raw} (using 0-indexed: {target_timesteps})")

    # Create dataset with intermediate states and timesteps
    dataset = TensorDataset(intermediate_states, images)

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
        pin_memory=True,
        collate_fn=default_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate_fn
    )

    return train_loader, val_loader, target_timesteps


def main(parsed_args):
    reset_rand(parsed_args.seed)

    # Device setup
    if parsed_args.cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load pretrained DDPM
    print(f"Loading pretrained DDPM from {parsed_args.ddpm_ckpt}...")
    ddpm = MNISTDiffusion(
        timesteps=parsed_args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=parsed_args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    ckpt = torch.load(parsed_args.ddpm_ckpt, map_location=device)
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
        base_dim=parsed_args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    # Initialize decoder from pretrained weights (warm start)
    model.load_decoder_from_pretrained(ddpm.model)

    # Setup EMA
    adjust = 1 * parsed_args.batch_size * parsed_args.model_ema_steps / parsed_args.epochs
    alpha = 1.0 - parsed_args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    # Load data
    train_loader, val_loader, target_timesteps = create_dataloaders(
        parsed_args.data_path,
        parsed_args.batch_size,
        parsed_args.train_split
    )

    # Convert to tensor on device
    target_timesteps = torch.tensor(target_timesteps, dtype=torch.long, device=device)

    # Optimizer and scheduler
    # Only optimize decoder parameters (encoder is frozen)
    optimizer = AdamW(model.decoder.parameters(), lr=parsed_args.lr)
    scheduler = OneCycleLR(
        optimizer,
        parsed_args.lr,
        total_steps=parsed_args.epochs * len(train_loader),
        pct_start=0.25,
        anneal_strategy='cos'
    )

    # Loss functions
    loss_fn_huber = nn.HuberLoss(reduction='mean', delta=parsed_args.huber_delta)
    loss_fn_lpips = lpips.LPIPS(net=parsed_args.lpips_net).to(device)
    loss_fn_lpips.requires_grad_(False)  # Freeze LPIPS weights

    print(f"Using hybrid loss: {parsed_args.loss_huber_weight}*Huber(δ={parsed_args.huber_delta})"
          f" + {parsed_args.loss_lpips_weight}*LPIPS({parsed_args.lpips_net})")

    # Create output directory
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_steps = 0
    best_val_loss = float('inf')

    if parsed_args.resume_ckpt:
        print(f"Resuming from checkpoint: {parsed_args.resume_ckpt}")
        resume_ckpt = torch.load(parsed_args.resume_ckpt, map_location=device)

        model.load_state_dict(resume_ckpt['model'])
        model_ema.load_state_dict(resume_ckpt['model_ema'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])

        start_epoch = resume_ckpt['epoch'] + 1
        global_steps = resume_ckpt['global_steps']
        best_val_loss = resume_ckpt.get('val_loss', float('inf'))

        print(f"Resumed from epoch {resume_ckpt['epoch'] + 1}, global_steps "
              f"{global_steps}, best_val_loss {best_val_loss:.5f}")

        # Restore scheduler state
        scheduler.last_epoch = global_steps - 1
        # pylint: disable=protected-access
        scheduler._step_count = global_steps
        scheduler.step()
    else:
        start_epoch = 0
        global_steps = 0
        best_val_loss = float('inf')

    for epoch in range(start_epoch, parsed_args.epochs):
        # Training
        model.train()
        train_loss_total = 0.0
        train_loss_huber = 0.0
        train_loss_lpips = 0.0

        for i, (intermediate_states_batch, target_images) in enumerate(train_loader):
            # intermediate_states_batch: [B, 5, 1, 28, 28]
            intermediate_states_batch = intermediate_states_batch.to(device)
            target_images = target_images.to(device)

            # Randomly select one timestep per sample in batch
            batch_size = intermediate_states_batch.shape[0]
            timestep_indices = torch.randint(0, 5, (batch_size,), device=device)

            # Extract the corresponding noisy image and timestep for each sample
            # [B, 1, 28, 28]
            noisy_images = intermediate_states_batch[
                torch.arange(batch_size, device=device), timestep_indices]
            t = target_timesteps[timestep_indices]  # [B]

            # Forward pass
            pred_images = model(noisy_images, t)

            # Hybrid loss computation
            huber_loss = loss_fn_huber(pred_images, target_images)

            # LPIPS requires larger images (min ~32x32) - upsample to 64x64 and convert to RGB
            pred_upsampled = F.interpolate(
                pred_images, size=(64, 64), mode='bilinear', align_corners=False)
            target_upsampled = F.interpolate(
                target_images, size=(64, 64), mode='bilinear', align_corners=False)
            pred_rgb = pred_upsampled.repeat(1, 3, 1, 1)
            target_rgb = target_upsampled.repeat(1, 3, 1, 1)
            lpips_loss = loss_fn_lpips(pred_rgb, target_rgb).mean()

            loss = (parsed_args.loss_huber_weight * huber_loss +
                    parsed_args.loss_lpips_weight * lpips_loss)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update EMA
            if global_steps % parsed_args.model_ema_steps == 0:
                model_ema.update_parameters(model)

            train_loss_total += loss.item()
            train_loss_huber += huber_loss.item()
            train_loss_lpips += lpips_loss.item()
            global_steps += 1

            # Logging
            if i % parsed_args.log_freq == 0:
                print(f"Epoch[{epoch+1}/{parsed_args.epochs}], Step[{i}/{len(train_loader)}], "
                      f"Loss:{loss.item():.5f} (Huber:{huber_loss.item():.5f}, "
                      f"LPIPS:{lpips_loss.item():.5f}), LR:{scheduler.get_last_lr()[0]:.6f}")

        train_loss_total /= len(train_loader)
        train_loss_huber /= len(train_loader)
        train_loss_lpips /= len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_loss_huber = 0.0
        val_loss_lpips = 0.0

        with torch.no_grad():
            for intermediate_states_batch, target_images in val_loader:
                intermediate_states_batch = intermediate_states_batch.to(device)
                target_images = target_images.to(device)

                # Randomly select one timestep per sample in batch
                batch_size = intermediate_states_batch.shape[0]
                timestep_indices = torch.randint(0, 5, (batch_size,), device=device)

                # Extract the corresponding noisy image and timestep for each sample
                noisy_images = intermediate_states_batch[
                    torch.arange(batch_size, device=device), timestep_indices]
                t = target_timesteps[timestep_indices]

                pred_images = model(noisy_images, t)

                # Hybrid loss computation
                huber_loss = loss_fn_huber(pred_images, target_images)

                # LPIPS requires larger images (min ~32x32) - upsample to 64x64 and convert to RGB
                pred_upsampled = F.interpolate(
                    pred_images, size=(64, 64), mode='bilinear', align_corners=False)
                target_upsampled = F.interpolate(
                    target_images, size=(64, 64), mode='bilinear', align_corners=False)
                pred_rgb = pred_upsampled.repeat(1, 3, 1, 1)
                target_rgb = target_upsampled.repeat(1, 3, 1, 1)
                lpips_loss = loss_fn_lpips(pred_rgb, target_rgb).mean()

                loss = (parsed_args.loss_huber_weight * huber_loss +
                        parsed_args.loss_lpips_weight * lpips_loss)

                val_loss_total += loss.item()
                val_loss_huber += huber_loss.item()
                val_loss_lpips += lpips_loss.item()

        val_loss_total /= len(val_loader)
        val_loss_huber /= len(val_loader)
        val_loss_lpips /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{parsed_args.epochs} Summary:")
        print(f"  Train Loss: {train_loss_total:.5f} "
              f"(Huber: {train_loss_huber:.5f}, LPIPS: {train_loss_lpips:.5f})")
        print(f"  Val Loss:   {val_loss_total:.5f} "
              f"(Huber: {val_loss_huber:.5f}, LPIPS: {val_loss_lpips:.5f})\n")

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_steps": global_steps,
            "val_loss": val_loss_total,
            "val_loss_huber": val_loss_huber,
            "val_loss_lpips": val_loss_lpips
        }

        torch.save(ckpt, os.path.join(parsed_args.output_dir, f'ckpt_epoch_{epoch+1:03d}.pt'))

        # Save best model
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(ckpt, os.path.join(parsed_args.output_dir,
                                          'best_model.pt'))
            print(f"  Saved best model (val_loss: {val_loss_total:.5f})")

        # Generate samples using EMA model
        model_ema.eval()
        sample_noise = torch.randn(parsed_args.n_samples, 1, 28, 28).to(device)

        # Use t=999 (0-indexed max timestep for pure noise)
        sample_t = torch.full((parsed_args.n_samples,), 999, device=device, dtype=torch.long)

        with torch.no_grad():
            samples = model_ema.module(sample_noise, sample_t)

        save_image(
            samples,
            os.path.join(parsed_args.output_dir, f'samples_epoch_{epoch+1:03d}.png'),
            nrow=int(math.sqrt(parsed_args.n_samples)),
            normalize=True
        )

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
