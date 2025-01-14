#!/bin/bash

# Activate virtual environment if using one
# source .venv/bin/activate  # Uncomment if needed

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="weights/icon_detect_v1_5/model_v1_5.pt"

# Run the FastAPI server with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload 


echo "Server is running!"
echo "To test the API, run:"
echo "python test_api.py"