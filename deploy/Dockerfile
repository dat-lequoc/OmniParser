FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Clone repositories
RUN git clone https://github.com/dat-lequoc/OmniParser.git && \
    cd OmniParser && \
    mkdir -p weights && \
    git clone https://huggingface.co/microsoft/OmniParser && \
    mv OmniParser/* weights/ && \
    mv weights ../ && \
    mv utils.py ../ && \
    mv requirements.txt ../ && \
    cd .. && \
    rm -rf OmniParser

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod>=1.2.0

# Copy handler code
COPY handler.py .

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "handler.py"]