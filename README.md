# Previsão de Obesidade – Tech Challenge FIAP (Fase 4)

Aplicação Streamlit estruturada em pacote Python (`src/`) com modelos KNN, Random Forest e SVM.

## Executar localmente
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Docker (produção)
```bash
docker build -f docker/Dockerfile -t obesity-predictor .
docker run -p 8080:8080 obesity-predictor
```

## Deploy no Cloud Run via Cloud Build
1. Ative as APIs `run.googleapis.com` e `cloudbuild.googleapis.com`.
2. Configure o trigger do Cloud Build apontando para `cloudbuild.yaml`.
3. Cada `git push` em `main` gera nova imagem e publica.
