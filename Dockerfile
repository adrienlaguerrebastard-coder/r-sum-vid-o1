# Image de déploiement : Python + ffmpeg + tesseract (FR/EN) pour le TikTok Foot Generator.
FROM python:3.11-slim

# Dépendances système : ffmpeg (montage), tesseract + langues (OCR du score)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python (couche cachée tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Modèle Whisper léger par défaut pour tenir dans une instance ~2 Go.
# (base < small < medium : + petit = - de RAM/CPU mais - précis)
ENV WHISPER_MODEL=base \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 1 SEUL worker (l'état des jobs est en mémoire process + threads de fond) ;
# plusieurs threads pour servir les requêtes ; timeout 0 = pas de kill des requêtes longues.
CMD ["sh", "-c", "gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:${PORT} app:app"]
