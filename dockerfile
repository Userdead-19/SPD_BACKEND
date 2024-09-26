# Use the smaller Python slim image
FROM python:3.9-slim-buster

# Install only necessary tools
RUN apt-get update &&
    apt-get install -y --no-install-recommends ffmpeg &&
    apt-get clean &&
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install only necessary requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Optimize TensorFlow for CPU usage (optional)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Start FastAPI with one worker to save memory
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
