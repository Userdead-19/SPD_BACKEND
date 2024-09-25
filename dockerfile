# Stage 1: Build Stage
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 &&
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies only in the builder stage
RUN pip install --no-cache-dir --upgrade pip &&
    pip install --no-cache-dir --upgrade -r requirements.txt

# Install OpenAI Whisper
RUN pip install --no-cache-dir --upgrade openai-whisper

# Stage 2: Final Stage
FROM python:3.9-slim

# Install runtime dependencies (ffmpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 &&
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy the rest of the application code
WORKDIR /app
COPY . .

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
