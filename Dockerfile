FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponha a porta padrão do Cloud Run
EXPOSE 8080

# Use o shell para passar a variável de ambiente PORT ao Streamlit
CMD streamlit run app/main.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
