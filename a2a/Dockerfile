FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . .

# Run the application
CMD ["python", "-m", "agent"]
