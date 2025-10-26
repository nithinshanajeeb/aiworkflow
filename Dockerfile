# Dockerfile
# Container image for the Streamlit intake application

FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable output flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# Install system dependencies needed for psycopg and Streamlit
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for deterministic dependency install
RUN python -m venv /opt/venv

WORKDIR /app

# Install Python dependencies separately to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Streamlit listens on 8501 by default
EXPOSE 8501

# Default command launches the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
