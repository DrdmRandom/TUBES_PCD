# Gunakan base image Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install dependensi dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Streamlit (default: 8501)
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]
