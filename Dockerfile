FROM python:3.13-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run the server
CMD ["python", "server.py"]
