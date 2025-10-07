# Use an official Python runtime (3.10) as a parent image
FROM python:3.10-slim

# Set environment variables to reduce Python output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create and switch to the app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements before code (leverages Docker cache)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . ./

# Expose Flask default port
EXPOSE 5000

# Launch the app via run.py
CMD ["python", "run.py"]