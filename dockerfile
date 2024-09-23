# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
