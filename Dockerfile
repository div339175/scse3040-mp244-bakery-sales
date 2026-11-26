# Base image specified by STUDENT_HANDBOOK.md section 9
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# Set working directory
WORKDIR /app

# Copy dependency definition first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY src/ ./src/
COPY app/ ./app/
COPY params.yaml .

# Expose container port specified in PROBLEM.md section 6
EXPOSE 8000

# Start FastAPI service with Uvicorn on 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
