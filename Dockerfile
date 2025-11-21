# Base image with PyTorch 2.5.1, CUDA 12.4, cuDNN 9
# Compatible with RTX 4070/5080
#build for vast in repo dir with "docker build -t mckellcarter/mnist-diffusion:latest ." 
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace/

# Activate virtual environment from base image
#RUN . /venv/main/bin/activate

#ports 
EXPOSE 22 

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data, outputs, and checkpoints
RUN mkdir -p mnist_data results checkpoints direct_decoder_data

# Copy source code
COPY *.py ./
COPY direct_decoder_data/training_data.pt direct_decoder_data/training_data.pt
COPY jenny_6x0/steps_00046900.pt checkpoints/ddpm_checkpoint.pt

# Set environment variables for better GPU memory management
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

# Default command opens bash for interactive work
CMD ["bash"]