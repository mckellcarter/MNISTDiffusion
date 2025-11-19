# Direct Decoder Implementation

This implementation adds a **direct decoding** capability to the MNIST Diffusion model, enabling single-step image generation and feature space manipulation for attribution analysis.

## Overview

The direct decoder learns to map from the pretrained DDPM encoder's feature space directly to clean images, bypassing the iterative 1000-step denoising process. This enables:

1. **Fast generation**: Single forward pass instead of 1000 steps
2. **Feature space manipulation**: Direct control over generated images via hooks
3. **Attribution analysis**: Track which training samples influence generation
4. **Controllable synthesis**: Include/exclude specific samples from outputs

## Architecture

```
Input: Random Noise [B, 1, 28, 28]
    ↓
[Frozen Encoder] ← From pretrained DDPM
    ↓
Bottleneck Features [B, 128, 7, 7]
    ↓
[Hook Point] ← Feature manipulation
    ↓
[Trainable Direct Decoder] ← New component (no timestep)
    ↓ (with skip connections)
Output: Clean Image [B, 1, 28, 28]
```

**Key Design Choices:**
- **Frozen encoder**: Uses pretrained DDPM encoder (up to mid_block)
- **No timestep conditioning**: Decoder operates on pure spatial features (t=None)
- **Skip connections**: Preserved from encoder for better quality
- **Warm start**: Decoder initialized from pretrained DDPM decoder weights
- **Hook mechanism**: Single manipulation point at bottleneck features

## Files

- `direct_decoder.py`: Core architecture (DirectEncoder, DirectDecoder, DirectDiffusion)
- `generate_training_data.py`: Pre-generates 60k (noise, DDPM_output) training pairs
- `train_direct_decoder.py`: Training script for direct decoder
- `test_direct_decoder.py`: Inference and hook manipulation examples

## Usage

### Step 1: Pre-generate Training Data

Generate 60k training pairs from pretrained DDPM:

```bash
python generate_training_data.py \
    --ckpt jenny_6x0/steps_00046900.pt \
    --n_samples 60000 \
    --batch_size 100 \
    --output_dir direct_decoder_data
```

**Output:** `direct_decoder_data/training_data.pt` (~1-2GB)

**What it does:**
- Loads pretrained DDPM (EMA model)
- Generates 60k random noise seeds
- Runs full 1000-step sampling for each
- Saves (noise, image) pairs for training

**Time estimate:** ~2-10 hours depending on device

### Step 2: Train Direct Decoder

Train decoder to map noise → images directly:

```bash
python train_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --data_path direct_decoder_data/training_data.pt \
    --lr 0.0001 \
    --batch_size 128 \
    --epochs 50 \
    --output_dir direct_decoder_results
```

**Key arguments:**
- `--ddpm_ckpt`: Pretrained DDPM checkpoint (for encoder initialization)
- `--data_path`: Pre-generated training data
- `--train_split`: Train/val split ratio (default: 0.9)
- `--epochs`: Number of training epochs (default: 50)

**Output:**
- `direct_decoder_results/ckpt_epoch_XXX.pt`: Checkpoints
- `direct_decoder_results/best_model.pt`: Best validation loss model
- `direct_decoder_results/samples_epoch_XXX.png`: Generated samples per epoch

**What it does:**
- Freezes encoder from pretrained DDPM
- Initializes decoder from pretrained weights (warm start)
- Trains decoder with MSE loss: `loss = ||decoder(encoder(noise)) - ddpm_output||²`
- Uses EMA for stable sampling
- 90/10 train/val split

### Step 3: Test and Visualize

Run inference with various hook configurations:

```bash
python test_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --decoder_ckpt direct_decoder_results/best_model.pt \
    --n_samples 36 \
    --output_dir test_outputs
```

**Generates:**
- `samples_standard.png`: No hook (baseline)
- `samples_amplified.png`: Features scaled 1.2x
- `samples_dampened.png`: Features scaled 0.8x
- `samples_biased.png`: Constant bias added to features
- `samples_interpolation.png`: Smooth noise interpolation

## Feature Space Manipulation

### Hook Function API

Hooks allow real-time feature manipulation at the bottleneck:

```python
def my_hook(features):
    """
    Args:
        features: Tensor [B, 128, 7, 7] - bottleneck features

    Returns:
        modified_features: Tensor [B, 128, 7, 7]
    """
    # Your manipulation logic here
    return modified_features

# Use during inference
output = model(noise, hook_fn=my_hook)
```

### Example Hooks

**1. Feature Amplification**
```python
def amplify_hook(features):
    return features * 1.2  # Increase feature magnitude
```

**2. Feature Direction Push/Pull**
```python
# Assume target_features is from a specific training sample
def attribution_hook(features, target_features, strength=0.5):
    direction = target_features - features
    return features + strength * direction  # Move toward target
```

**3. Multi-Sample Exclusion**
```python
def exclusion_hook(features, exclude_features_list, strength=0.5):
    for exclude_features in exclude_features_list:
        direction = features - exclude_features
        features = features + strength * direction  # Move away
    return features
```

**4. Conditional Manipulation**
```python
def selective_channel_hook(features):
    # Only modify certain feature channels
    features_clone = features.clone()
    features_clone[:, :64, :, :] *= 1.5  # Amplify first 64 channels
    return features_clone
```

## Attribution Workflow (Future)

Once decoder is trained, attribution analysis flow:

1. **Extract reference features** for all training samples:
   ```python
   with torch.no_grad():
       ref_features, _ = model.encoder(training_images)
   # Store ref_features for each training sample
   ```

2. **Generate with attribution control**:
   ```python
   # Include specific samples
   def include_hook(features):
       target_feat = ref_features[sample_indices].mean(dim=0)
       return features + 0.3 * (target_feat - features)

   output = model(noise, hook_fn=include_hook)
   ```

3. **Measure attribution** (cosine similarity in feature space):
   ```python
   generated_features, _ = model.encoder(noise)
   similarity = F.cosine_similarity(generated_features, ref_features)
   # High similarity → strong attribution to that sample
   ```

## Training Details

**Loss Function:**
- MSE between direct decoder output and DDPM-generated images
- Future: Can add perceptual loss (LPIPS) or GAN loss for better quality

**Decoder Architecture:**
- Mirrors DDPM decoder structure (SimpleDecoderBlock)
- No TimeMLP (timestep-agnostic)
- Uses ResidualBottleneck blocks from ShuffleNetV2
- Skip connections from encoder for multi-scale information

**Why train on DDPM outputs (not real MNIST)?**
- Learns the generative distribution of DDPM (richer than training set)
- Enables generation of novel samples (not just memorized digits)
- Required for attribution: must match DDPM's feature→image mapping

## Design Rationale

### Q: Why t=None for encoder?
**A:** Consistency. At inference, we manipulate features directly without timestep context. Using t=None during training ensures feature space structure is timestep-agnostic.

### Q: Why keep skip connections?
**A:** Quality. Skip connections preserve fine-grained spatial information. While pure bottleneck-only would be cleaner for single-point hooks, quality degrades significantly.

### Q: Why warm start from pretrained decoder?
**A:** Efficiency. The pretrained decoder already knows how to decode these feature distributions (with timesteps). Initializing from it significantly speeds convergence.

### Q: Why 60k samples?
**A:** Distribution coverage. Matches MNIST size, ensures decoder sees full diversity of DDPM's generative distribution.

## Next Steps

**Immediate improvements:**
1. Add LPIPS perceptual loss for better visual quality
2. Implement multi-scale hooks (across decoder layers)
3. Add feature extraction utilities for attribution mapping

**Advanced features:**
1. GAN discriminator for distribution matching (DMD2 approach)
2. Consistency distillation across multiple noise seeds
3. Conditional generation (class labels, style control)
4. Feature space PCA/clustering for interpretability

## Troubleshooting

**Issue: Poor sample quality**
- Ensure pretrained DDPM is high quality
- Try longer training (more epochs)
- Experiment with learning rate (default: 0.0001)
- Add perceptual loss or GAN loss

**Issue: Hooks have no effect**
- Check hook function is actually modifying features
- Increase manipulation strength
- Verify hook is applied at correct point (bottleneck)

**Issue: Training data generation too slow**
- Use smaller batch_size (trades memory for time)
- Consider DDIM sampling (50 steps instead of 1000)
- Use GPU/MPS acceleration

**Issue: Mode collapse in generated samples**
- Train longer
- Increase dataset size
- Add noise/regularization during training
- Check validation loss is still decreasing

## References

- DMD2 Paper: https://arxiv.org/abs/2405.14867
- Original DDPM: https://arxiv.org/abs/2006.11239
- Related: Consistency Models, Progressive Distillation
