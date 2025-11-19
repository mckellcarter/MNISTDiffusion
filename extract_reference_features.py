import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from direct_decoder import DirectDiffusion
from model import MNISTDiffusion
import argparse
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract reference features from MNIST training set")
    parser.add_argument('--ddpm_ckpt', type=str, required=True, help='Path to pretrained DDPM checkpoint')
    parser.add_argument('--decoder_ckpt', type=str, required=True, help='Path to trained direct decoder checkpoint')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for feature extraction')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of UNet')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps')
    parser.add_argument('--output_path', type=str, default='reference_features.pt', help='Output file path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    return parser.parse_args()


def create_mnist_dataloader(batch_size, image_size=28, num_workers=4):
    """
    Create MNIST dataloader with same preprocessing as training.
    """
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] to [-1,1]
    ])

    train_dataset = MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for indexing
        num_workers=num_workers
    )

    return train_loader, train_dataset


def main(args):
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

    # Load MNIST dataset
    print("Loading MNIST training dataset...")
    train_loader, train_dataset = create_mnist_dataloader(args.batch_size)

    print(f"Total training samples: {len(train_dataset)}")

    # Extract features
    print("Extracting features from all training samples...")

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting features"):
            images = images.to(device)

            # Extract encoder features (bottleneck only)
            bottleneck_features, _ = model.encoder(images)

            # Move to CPU and store
            all_features.append(bottleneck_features.cpu())
            all_labels.append(labels)

    # Concatenate all features
    print("Concatenating features...")
    reference_features = torch.cat(all_features, dim=0)  # [N, 128, 7, 7]
    reference_labels = torch.cat(all_labels, dim=0)      # [N]

    print(f"Reference features shape: {reference_features.shape}")
    print(f"Reference labels shape: {reference_labels.shape}")

    # Save to disk
    print(f"Saving to {args.output_path}...")
    torch.save({
        'features': reference_features,
        'labels': reference_labels,
        'n_samples': len(reference_features),
        'feature_shape': list(reference_features.shape[1:]),
        'metadata': {
            'ddpm_ckpt': args.ddpm_ckpt,
            'decoder_ckpt': args.decoder_ckpt,
            'dataset': 'MNIST_train',
        }
    }, args.output_path)

    print(f"Saved {len(reference_features)} reference features")
    print(f"File size: {os.path.getsize(args.output_path) / 1024**2:.2f} MB")

    # Print some statistics
    print("\nFeature Statistics:")
    print(f"  Mean: {reference_features.mean():.4f}")
    print(f"  Std:  {reference_features.std():.4f}")
    print(f"  Min:  {reference_features.min():.4f}")
    print(f"  Max:  {reference_features.max():.4f}")

    print("\nLabel Distribution:")
    for digit in range(10):
        count = (reference_labels == digit).sum().item()
        print(f"  Digit {digit}: {count} samples")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
