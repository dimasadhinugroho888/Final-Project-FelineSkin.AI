FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies untuk OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke dalam container
COPY . .

EXPOSE 8080

# Jalankan streamlit (pastikan file utama kamu app.py)
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
