FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt
RUN uv pip install "fastapi[all]" uvicorn vllm

# Copy application code and models
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]