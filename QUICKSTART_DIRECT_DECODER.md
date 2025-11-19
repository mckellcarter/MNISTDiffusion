# Quick Start: Direct Decoder

Fast guide to get the direct decoder running for attribution-controlled generation.

## Prerequisites

- Pretrained DDPM checkpoint: `jenny_6x0/steps_00046900.pt`
- PyTorch environment installed

## Step-by-Step Execution

### 1. Small Test Run (Recommended First)

Test with 100 samples (~5 minutes):

```bash
# Generate test data
python generate_training_data.py \
    --ckpt jenny_6x0/steps_00046900.pt \
    --n_samples 100 \
    --batch_size 10 \
    --output_dir test_data

# Train for 2 epochs
python train_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --data_path test_data/training_data.pt \
    --epochs 2 \
    --batch_size 10 \
    --output_dir test_results

# Test inference
python test_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --decoder_ckpt test_results/ckpt_epoch_002.pt \
    --n_samples 9 \
    --output_dir test_outputs
```

**Expected outputs:**
- `test_data/training_data.pt` (~2MB)
- `test_results/ckpt_epoch_002.pt` (checkpoint)
- `test_outputs/*.png` (various hook tests)

### 2. Full Production Run

After successful test, run full pipeline:

#### 2a. Generate 60k Training Pairs (~2-10 hours)

```bash
python generate_training_data.py \
    --ckpt jenny_6x0/steps_00046900.pt \
    --n_samples 60000 \
    --batch_size 100 \
    --output_dir direct_decoder_data
```

**Progress monitoring:**
- Watch for checkpoint saves every 10 batches
- Final file: `direct_decoder_data/training_data.pt` (~1-2GB)
- Can interrupt and resume by loading checkpoints

**Time estimates:**
- MPS (Apple Silicon): ~3-5 hours
- CUDA (GPU): ~2-3 hours
- CPU: ~8-10 hours

#### 2b. Train Direct Decoder (~1-3 hours)

```bash
python train_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --data_path direct_decoder_data/training_data.pt \
    --lr 0.0001 \
    --batch_size 128 \
    --epochs 50 \
    --output_dir direct_decoder_results
```

**Monitoring:**
- Watch train/val loss in terminal
- Samples saved per epoch: `direct_decoder_results/samples_epoch_XXX.png`
- Best model: `direct_decoder_results/best_model.pt`

**What to expect:**
- Loss should decrease steadily
- Generated samples should improve quality over epochs
- Val loss should stabilize (if increasing, reduce LR)

#### 2c. Extract Reference Features (~2-5 minutes)

```bash
python extract_reference_features.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --decoder_ckpt direct_decoder_results/best_model.pt \
    --batch_size 256 \
    --output_path reference_features.pt
```

**Output:**
- `reference_features.pt` (~350MB)
- Contains features for all 60k MNIST training samples

#### 2d. Attribution-Controlled Generation (~1 minute)

```bash
python attribution_generation.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --decoder_ckpt direct_decoder_results/best_model.pt \
    --ref_features_path reference_features.pt \
    --n_samples 36 \
    --output_dir attribution_outputs
```

**Outputs:**
- `samples_baseline.png`: Standard generation
- `samples_include.png`: Pushed toward samples [0-9]
- `samples_exclude.png`: Pushed away from samples [100-109]
- `samples_class_3.png`: Conditioned to generate digit 3
- `attribution_results.pt`: Similarity scores

## Troubleshooting

### Out of Memory

**During data generation:**
```bash
python generate_training_data.py \
    --ckpt jenny_6x0/steps_00046900.pt \
    --n_samples 60000 \
    --batch_size 50 \  # Reduce from 100
    --output_dir direct_decoder_data
```

**During training:**
```bash
python train_direct_decoder.py \
    --ddpm_ckpt jenny_6x0/steps_00046900.pt \
    --data_path direct_decoder_data/training_data.pt \
    --batch_size 64 \  # Reduce from 128
    --output_dir direct_decoder_results
```

### Slow Performance

Force CPU or different device:
```bash
# Force CPU
python generate_training_data.py --cpu ...

# Or set environment variable for specific GPU
CUDA_VISIBLE_DEVICES=0 python generate_training_data.py ...
```

### Poor Quality Samples

1. **Train longer:**
   ```bash
   python train_direct_decoder.py --epochs 100 ...
   ```

2. **Adjust learning rate:**
   ```bash
   python train_direct_decoder.py --lr 0.00005 ...  # Lower LR
   ```

3. **Check pretrained DDPM quality:**
   - Generate samples from original DDPM first
   - Ensure DDPM produces good digits

### Hooks Have No Effect

Increase manipulation strength in `attribution_generation.py`:
```python
# Edit line with strength parameter
include_hook = create_include_hook(ref_features, include_indices, strength=0.8)
```

## Expected Results

**After successful training:**
- Direct decoder generates MNIST-like digits in single forward pass
- Quality should be comparable to DDPM (but much faster)
- Hooks should visibly affect generated images
- Attribution scores should identify similar training samples

**Quality metrics (informal):**
- Generated digits should be recognizable
- Variety across samples (not mode collapse)
- Hook effects should be clear when comparing baseline vs manipulated

## Custom Hook Examples

Once pipeline is working, create custom hooks:

```python
# In attribution_generation.py or custom script

# 1. Interpolate between two samples
def interpolate_hook(features, idx1, idx2, alpha=0.5):
    feat1 = ref_features[idx1]
    feat2 = ref_features[idx2]
    target = (1 - alpha) * feat1 + alpha * feat2
    return features + 0.5 * (target - features)

# 2. Multi-sample averaging
def multi_include_hook(features, indices_list, strength=0.5):
    target = ref_features[indices_list].mean(dim=0)
    return features + strength * (target - features)

# 3. Selective channel manipulation
def channel_hook(features, channel_range, scale=1.2):
    features_clone = features.clone()
    features_clone[:, channel_range[0]:channel_range[1]] *= scale
    return features_clone

# Use in generation
with torch.no_grad():
    samples = model(noise, hook_fn=interpolate_hook)
```

## Next Steps

After basic pipeline works:

1. **Analyze attribution patterns:**
   - Which training samples contribute most?
   - Are specific digits easier to exclude?

2. **Experiment with hook strengths:**
   - Find optimal strength values (0.0 to 1.0)
   - Test combinations (include + exclude)

3. **Add advanced losses:**
   - LPIPS perceptual loss
   - GAN discriminator

4. **Scale to multi-scale hooks:**
   - Hook at multiple decoder layers
   - Finer-grained control

## File Checklist

After full run, you should have:

```
MNISTDiffusion/
├── direct_decoder_data/
│   └── training_data.pt              (~1-2GB)
├── direct_decoder_results/
│   ├── best_model.pt                  (checkpoint)
│   ├── ckpt_epoch_*.pt               (per epoch)
│   └── samples_epoch_*.png           (visualizations)
├── reference_features.pt              (~350MB)
└── attribution_outputs/
    ├── samples_baseline.png
    ├── samples_include.png
    ├── samples_exclude.png
    ├── samples_class_3.png
    └── attribution_results.pt
```

## Time Budget

**Total time (full pipeline):**
- Data generation: 2-10 hours
- Training: 1-3 hours
- Feature extraction: 2-5 minutes
- Attribution generation: 1 minute
- **Total: ~3-13 hours** (mostly unattended)

**Recommended schedule:**
- Day 1 morning: Start data generation (leave running)
- Day 1 evening: Start training (leave overnight)
- Day 2 morning: Extract features + test attribution (10 minutes)

## Questions?

See detailed documentation:
- `DIRECT_DECODER_README.md` - Comprehensive guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `CLAUDE.md` - Project overview
