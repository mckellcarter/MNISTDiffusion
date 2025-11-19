# Vast.ai Deployment Guide

This guide covers deploying the MNIST Diffusion model to vast.ai for GPU training.

## Prerequisites

- Docker Hub account (free): https://hub.docker.com
- Vast.ai account (pay-as-you-go): https://vast.ai
- Docker installed locally (for custom container option)

## Training Workflows

This project has two distinct training workflows with different data requirements:

### 1. DDPM Training (train_mnist.py)
- **Purpose:** Train the main diffusion model from scratch
- **Input data:** `results/jenny_s0_delYgt0.08_663del_include.pt` (~350MB) - subset indices
- **Output location:** `jenny_s0_delYgt0.08t663/` directory
- **Output files:**
  - `steps_*.pt` - Model checkpoints (both model and EMA states)
  - `steps_*.png` - Generated sample images

### 2. Direct Decoder Training (train_direct_decoder.py)
- **Purpose:** Train single-step decoder using pre-generated DDPM outputs
- **Input data:** `direct_decoder_data/training_data.pt` (~359MB) - noise/image pairs
- **Requires:** Pre-trained DDPM checkpoint (e.g., `jenny_6x0/steps_00046900.pt`)
- **Output location:** `direct_decoder_results/` (default, configurable via `--output_dir`)
- **Output files:**
  - `ckpt_epoch_*.pt` - Checkpoints for each epoch
  - `best_model.pt` - Best model based on validation loss
  - `samples_epoch_*.png` - Generated samples

---

## Deployment Options

### Option 1: Pre-built Custom Container (Recommended)

**Advantages:**
- Faster instance startup
- Fully reproducible environment
- No dependency installation on remote host

**Build and Push:**
```bash
# Build the container
docker build -t yourusername/mnist-diffusion:latest .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push yourusername/mnist-diffusion:latest
```

### Option 2: Git Clone Method (Faster Testing)

**Advantages:**
- No Docker build required
- Quick iteration during development
- Good for testing code changes

**Setup on remote instance:**
```bash
# Clone repository
git clone https://github.com/yourusername/MNISTDiffusion.git
cd MNISTDiffusion

# Install dependencies
pip install -r requirements.txt
```

---

## Workflow A: DDPM Training

### 1. Rent Instance on Vast.ai

**Instance Requirements:**
- GPU: RTX 4070/4070 Ti/5080 (12GB+ VRAM)
- CUDA: 12.4+
- Disk: 20GB minimum
- Docker image: `yourusername/mnist-diffusion:latest` OR `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

**Recommended filters:**
- Interruptible: Yes (saves 30-50%)
- CUDA 12.4+
- Disk >= 20GB

### 2. Upload Training Data

```bash
# Get SSH details from vast.ai dashboard (hostname, port)
# Upload subset indices file (required by train_mnist.py:46)
scp -P PORT results/jenny_s0_delYgt0.08_663del_include.pt root@INSTANCE_IP:/workspace/results/
```

**Note:** MNIST dataset downloads automatically on first run (~50MB)

### 3. Start Training

```bash
# SSH into instance
ssh -p PORT root@INSTANCE_IP

# For custom container:
cd /workspace

# For git clone method:
cd MNISTDiffusion

# Run training with default parameters (100 epochs, batch 128)
./run_training.sh

# Or with custom parameters:
./run_training.sh --epochs 200 --batch_size 256 --lr 0.0005
```

**Using tmux for persistent sessions:**
```bash
# Start tmux to keep training running if disconnected
tmux new -s ddpm

# Run training
./run_training.sh

# Detach: Ctrl+B, then D
# Reattach after reconnecting: tmux attach -t ddpm
```

### 4. Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check output directory
ls -lh jenny_s0_delYgt0.08t663/
```

### 5. Download Results

```bash
# From local machine, download all checkpoints and samples
scp -P PORT -r root@INSTANCE_IP:/workspace/jenny_s0_delYgt0.08t663/ ./

# Or download only latest checkpoint
scp -P PORT root@INSTANCE_IP:/workspace/jenny_s0_delYgt0.08t663/steps_*.pt ./latest/
```

### 6. Destroy Instance

Go to vast.ai dashboard → Instances → Destroy instance

**Estimated cost for 100 epochs:** ~2-3 hours @ $0.25/hr = **$0.60-0.75**

---

## Workflow B: Direct Decoder Training

### Prerequisites
- Pre-trained DDPM checkpoint (e.g., `jenny_6x0/steps_00046900.pt`)
- Pre-generated training data (`direct_decoder_data/training_data.pt`)

### 1. Rent Instance

Same requirements as Workflow A

### 2. Upload Data and Checkpoint

```bash
# Upload pre-generated training data (359MB)
scp -P PORT direct_decoder_data/training_data.pt root@INSTANCE_IP:/workspace/direct_decoder_data/

# Upload pretrained DDPM checkpoint (required for decoder initialization)
scp -P PORT jenny_6x0/steps_00046900.pt root@INSTANCE_IP:/workspace/checkpoints/ddpm_checkpoint.pt
```

**Note:** Create directories if needed:
```bash
ssh -p PORT root@INSTANCE_IP "mkdir -p /workspace/direct_decoder_data /workspace/checkpoints"
```

### 3. Start Training

```bash
ssh -p PORT root@INSTANCE_IP

# Start tmux session
tmux new -s decoder

# Run training
python train_direct_decoder.py \
    --ddpm_ckpt checkpoints/ddpm_checkpoint.pt \
    --data_path direct_decoder_data/training_data.pt \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.0001 \
    --output_dir direct_decoder_results

# Detach: Ctrl+B, then D
```

### 4. Monitor Training

```bash
# Reattach to tmux
tmux attach -t decoder

# Watch GPU usage
nvidia-smi

# Check outputs
ls -lh direct_decoder_results/
```

### 5. Download Results

```bash
# Download all results
scp -P PORT -r root@INSTANCE_IP:/workspace/direct_decoder_results/ ./

# Or just best model and final samples
scp -P PORT root@INSTANCE_IP:/workspace/direct_decoder_results/best_model.pt ./
scp -P PORT root@INSTANCE_IP:/workspace/direct_decoder_results/samples_epoch_050.png ./
```

**Estimated cost for 50 epochs:** ~1-3 hours @ $0.25/hr = **$0.25-0.75**

---

## Instance Selection Guide

### GPU Options

| GPU | VRAM | Typical Cost | Best For |
|-----|------|--------------|----------|
| RTX 4070 Ti | 12GB | $0.20-0.30/hr | DDPM training (sufficient) |
| RTX 5080 | 16GB | $0.35-0.50/hr | Future larger models |
| RTX 4090 | 24GB | $0.50-0.80/hr | Overkill for this project |

**Recommendation:** RTX 4070 Ti interruptible (~$0.20/hr)

### Interruptible vs On-Demand

**Interruptible:**
- 30-50% cheaper
- May get stopped if someone offers higher price
- Good for: Short runs (<5 hours), experiments
- **Use checkpointing** to resume if interrupted

**On-Demand:**
- More expensive but guaranteed
- Good for: Long runs (>10 hours), critical deadlines

---

## Cost Optimization Tips

### 1. Use Interruptible Instances
- Enable checkpointing (already implemented)
- Resume with `--ckpt` flag if interrupted

### 2. Test Locally First
```bash
# Catch bugs before paying for GPU time
python train_mnist.py --cpu --epochs 1
```

### 3. Monitor and Destroy Immediately
- Don't leave instances running after training completes
- Download results and destroy within minutes

### 4. Use Efficient Batch Sizes
```bash
# Find optimal batch size that fills GPU memory
# RTX 4070 Ti can handle batch 256+ for 28x28 images
./run_training.sh --batch_size 256
```

### 5. Checkpoint Management
```bash
# Resume from checkpoint instead of restarting
./run_training.sh --ckpt jenny_s0_delYgt0.08t663/steps_00012345.pt --epochs 200
```

---

## Data Transfer Strategies

### For Short Runs (<5 hours)
- Upload via SCP (simple, fast enough for <1GB)
- Download results immediately after completion

### For Long Runs (>10 hours)
- Consider vast.ai persistent storage ($0.10/GB/month)
- Or mount cloud storage (S3, GCS, etc.)
- Enables resuming across multiple instances

### Bandwidth Tips
```bash
# Compress before upload (if not already compressed)
tar -czf training_data.tar.gz direct_decoder_data/

# Upload compressed
scp -P PORT training_data.tar.gz root@IP:/workspace/

# Extract on remote
ssh -p PORT root@IP "cd /workspace && tar -xzf training_data.tar.gz"
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
./run_training.sh --batch_size 64

# Or reduce num_workers (train_mnist.py line 26)
```

### Missing Data File
```bash
# Verify upload succeeded
ssh -p PORT root@IP "ls -lh results/jenny_s0_delYgt0.08_663del_include.pt"

# Check script is looking in right place
# train_mnist.py:46 expects: results/jenny_s0_delYgt0.08_663del_include.pt
# train_direct_decoder.py expects: --data_path argument
```

### Connection Dropped During Training
- Training continues in background
- Reconnect and find process: `ps aux | grep python`
- **Always use tmux/screen** to prevent this issue

### Slow Training
```bash
# Verify GPU is being used
nvidia-smi

# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU utilization (should be >80%)
watch -n 1 nvidia-smi
```

### Docker Container Not Found
- Verify push succeeded: Check Docker Hub repository
- Use full image name: `docker.io/yourusername/mnist-diffusion:latest`
- Or use base image and git clone method instead

---

## Quick Reference Commands

### Build & Deploy Container
```bash
docker build -t yourusername/mnist-diffusion:latest .
docker push yourusername/mnist-diffusion:latest
```

### DDPM Training Workflow
```bash
# Upload data
scp -P PORT results/jenny_s0_delYgt0.08_663del_include.pt root@IP:/workspace/results/

# SSH and train
ssh -p PORT root@IP
tmux new -s ddpm
./run_training.sh

# Download results
scp -P PORT -r root@IP:/workspace/jenny_s0_delYgt0.08t663/ ./
```

### Direct Decoder Training Workflow
```bash
# Upload data and checkpoint
scp -P PORT direct_decoder_data/training_data.pt root@IP:/workspace/direct_decoder_data/
scp -P PORT jenny_6x0/steps_00046900.pt root@IP:/workspace/checkpoints/ddpm.pt

# SSH and train
ssh -p PORT root@IP
tmux new -s decoder
python train_direct_decoder.py \
    --ddpm_ckpt checkpoints/ddpm.pt \
    --data_path direct_decoder_data/training_data.pt

# Download results
scp -P PORT -r root@IP:/workspace/direct_decoder_results/ ./
```

### Resume From Checkpoint
```bash
# DDPM
./run_training.sh --ckpt jenny_s0_delYgt0.08t663/steps_00012345.pt

# Direct Decoder (modify command to load from checkpoint)
# Note: train_direct_decoder.py doesn't have --ckpt arg - you'd need to add it
```

---

## Testing the Container Locally

Before deploying to vast.ai, test the container locally:

```bash
# Build container
docker build -t mnist-diffusion:test .

# Run with GPU support (Linux/WSL)
docker run --gpus all -it mnist-diffusion:test

# Run CPU-only (macOS/no GPU)
docker run -it mnist-diffusion:test

# Inside container, test training (1 epoch)
python train_mnist.py --cpu --epochs 1
```

---

## Advanced: Parallel Experiments

Run multiple hyperparameter configs simultaneously:

```bash
# Rent 3 cheap instances (~$0.15/hr each)
# Instance 1: Default params
./run_training.sh

# Instance 2: Higher learning rate
./run_training.sh --lr 0.002

# Instance 3: Larger model
./run_training.sh --model_base_dim 128

# Compare results after all complete
```

Total cost: 3 instances × 3 hours × $0.15/hr = **$1.35** (vs $1.35 sequential)
Time saved: 9 hours → 3 hours

---

## Next Steps

1. **Test container locally** (optional but recommended)
2. **Rent a cheap instance** ($0.10-0.15/hr) for smoke test
3. **Run 1-2 epochs** to verify everything works
4. **Scale up** to full training once validated
5. **Destroy instance immediately** after downloading results

## Additional Resources

- Vast.ai documentation: https://vast.ai/docs/
- Docker Hub: https://hub.docker.com
- tmux cheat sheet: https://tmuxcheatsheet.com/
- PyTorch CUDA troubleshooting: https://pytorch.org/get-started/locally/
