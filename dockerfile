# Use the official Python image as a base
FROM python:3.9-slim

# Install ffmpeg and other required packages
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrage pymongo

# Copy the rest of the application code
COPY . .

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
