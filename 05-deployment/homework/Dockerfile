# Use the base image from the course (includes Python + dependencies)
FROM agrigorev/zoomcamp-model:2025

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and all other project files
COPY main.py .
COPY pipeline_v1.bin .

# Expose the port FastAPI will listen on
EXPOSE 8000

# Run your FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
