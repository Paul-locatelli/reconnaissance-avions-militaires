# Application de démonstration : détection et classification d'avions militaires.
# Inférence sur CPU : l'image tourne sur n'importe quelle machine, sans GPU ni pilote NVIDIA.
#
#   docker build -t avions-militaires .
#   docker run --rm -p 7860:7860 avions-militaires                     # puis http://localhost:7860
#   docker run --rm -p 7860:7860 -v "$PWD/student-dataset 2:/app/student-dataset 2:ro" avions-militaires
#                                         (avec le dataset : exemples et comparaison à la vérité terrain)
FROM python:3.12-slim

# Bibliothèques système chargées par OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    YOLO_OFFLINE=1
WORKDIR /app

# 1. PyTorch en version CPU (bien plus léger que la version CUDA), mêmes versions que le développement
# 2. Le reste des dépendances, aux versions exactes du fichier lock (torch déjà installé est conservé)
COPY requirements.txt requirements-lock.txt ./
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E "^(torch|torchvision)==" requirements-lock.txt > /tmp/constraints.txt \
    && pip install -r requirements.txt -c /tmp/constraints.txt

# Code, liste des images de test, seuils choisis sur la validation et poids des 3 modèles
COPY src/ src/
COPY data/splits/ data/splits/
COPY reports/comparaison_resume.csv reports/
COPY runs/yolo11s_20cls/weights/best.pt runs/yolo11s_20cls/weights/
COPY runs/yolo11s_avion/weights/best.pt runs/yolo11s_avion/weights/
COPY runs/cls_convnext_tiny/best.pt runs/cls_convnext_tiny/

EXPOSE 7860
# Sans TTA : sur CPU, les 8 orientations par avion font passer une image de 2 s à plus de 10 s
CMD ["python", "-m", "src.app", "--host", "0.0.0.0", "--no-tta"]
