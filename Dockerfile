# Use Python 3.12 base image
FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libglib2.0-0 \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install distutils separately, then the rest of the dependencies
RUN pip install --no-cache-dir distutils --no-build-isolation \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the Flask app port
EXPOSE 8080

# Run Flask using Python
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
