# Gunakan base image Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Copy all files into the container (general copy)
COPY . .

# Explicitly copy the Dataset directory to ensure it’s included
COPY Dataset /app/Dataset

# Install dependensi dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Streamlit (default: 8501)
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "UI_Master.py", "--server.port=8502", "--server.enableCORS=false", "--server.address=0.0.0.0"]
