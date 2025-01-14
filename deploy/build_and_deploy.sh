#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t datlequoc/omniparser:runtime .

# Test the image locally (optional)
echo "Testing image locally..."
docker run --gpus all -p 8000:8000 datlequoc/omniparser:runtime

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push datlequoc/omniparser:runtime

echo "Deployment files ready. Please configure RunPod with:"
echo "- Image: datlequoc/omniparser:runtime"
echo "- GPU: NVIDIA with CUDA 11.8 support"
echo "- Min Memory: 16GB"
echo "- Storage: 20GB"
