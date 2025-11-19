# Base image with PyTorch 2.5.1, CUDA 12.4, cuDNN 9
# Compatible with RTX 4070/5080
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/workspace-internal/

# Activate virtual environment from base image
RUN . /venv/main/bin/activate

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py ./

# Create directories for data and outputs
RUN mkdir -p mnist_data results

# Set environment variables for better GPU memory management
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

# Default command opens bash for interactive work
CMD ["bash"]