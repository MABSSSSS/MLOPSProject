# Use an official Python base image
FROM python:3.10-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy/pandas/scikit-learn)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip & install dependencies with extended timeout and reliable mirror
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --disable-pip-version-check -r requirements.txt
# Copy the entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
