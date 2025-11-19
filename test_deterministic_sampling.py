import torch
from model import MNISTDiffusion
from torchvision.utils import save_image
import os

def test_deterministic_sampling():
    """Test that deterministic sampling produces identical results."""

    # Setup
    device = torch.device("cpu")
    checkpoint_path = "jenny_6x0/steps_00046900.pt"
    n_samples = 4

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_ema' in ckpt:
        ema_state = ckpt['model_ema']
        state_dict = {}
        for k, v in ema_state.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            elif k != 'n_averaged':
                state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt['model'])

    model.eval()

    # Test 1: Generate samples with deterministic=True twice
    print("\nTest 1: Deterministic sampling (should produce identical results)")
    print("Generating first batch...")
    samples1, _ = model.sampling(
        n_samples=n_samples,
        device=device,
        deterministic=True,
        deterministic_seed=42
    )

    print("Generating second batch with same seed...")
    samples2, _ = model.sampling(
        n_samples=n_samples,
        device=device,
        deterministic=True,
        deterministic_seed=42
    )

    # Check if samples are identical
    diff = torch.abs(samples1 - samples2).max().item()
    print(f"Maximum difference between batches: {diff:.6e}")

    if diff < 1e-6:
        print("✓ Deterministic sampling works correctly!")
    else:
        print("✗ Deterministic sampling failed - results differ!")

    # Save samples for visual inspection
    os.makedirs("test_outputs", exist_ok=True)
    save_image(samples1, "test_outputs/deterministic_batch1.png", nrow=2, normalize=False)
    save_image(samples2, "test_outputs/deterministic_batch2.png", nrow=2, normalize=False)

    # Test 2: Compare with non-deterministic sampling
    print("\nTest 2: Non-deterministic sampling (should produce different results)")
    print("Generating two non-deterministic batches...")

    samples3, _ = model.sampling(
        n_samples=n_samples,
        device=device,
        deterministic=False
    )

    samples4, _ = model.sampling(
        n_samples=n_samples,
        device=device,
        deterministic=False
    )

    diff_nondet = torch.abs(samples3 - samples4).max().item()
    print(f"Maximum difference between non-deterministic batches: {diff_nondet:.6e}")

    if diff_nondet > 1e-6:
        print("✓ Non-deterministic sampling produces different results as expected!")
    else:
        print("✗ Non-deterministic sampling unexpectedly produced identical results!")

    save_image(samples3, "test_outputs/non_deterministic_batch1.png", nrow=2, normalize=False)
    save_image(samples4, "test_outputs/non_deterministic_batch2.png", nrow=2, normalize=False)

    # Test 3: Test with specific timestep capture
    print("\nTest 3: Testing timestep capture during sampling...")
    capture_timesteps = [999, 500, 100, 10, 0]

    samples5, _, captured_states = model.sampling(
        n_samples=n_samples,
        device=device,
        deterministic=True,
        deterministic_seed=123,
        capture_timesteps=capture_timesteps
    )

    print(f"Captured states at timesteps: {list(captured_states.keys())}")
    for t, state in captured_states.items():
        if state is not None:
            print(f"  t={t}: shape={state.shape}, mean={state.mean():.4f}, std={state.std():.4f}")

    print("\nAll tests completed!")
    print("Check test_outputs/ for generated images")

    return diff < 1e-6  # Return True if deterministic sampling works


if __name__ == "__main__":
    success = test_deterministic_sampling()
    exit(0 if success else 1)