"""Classifieur de vignettes : un CNN pré-entraîné sur ImageNet, ré-entraîné sur nos 20 classes.

Ce module est partagé entre l'entraînement (src/train_classifier.py) et le pipeline
d'inférence (src/pipeline.py) : mêmes prétraitements aux deux endroits.
"""
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

IMG_SIZE = 224
# Moyenne et écart-type d'ImageNet : le réseau pré-entraîné attend des images normalisées ainsi
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """Charge le réseau pré-entraîné et remplace sa dernière couche par une couche à num_classes sorties."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def train_transform() -> T.Compose:
    return T.Compose([
        # Recadrage aléatoire léger : simule une boîte du détecteur un peu décalée ou trop large
        T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        # Vue de dessus : toutes les orientations et symétries sont réalistes
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def save_checkpoint(path: Path, model: torch.nn.Module, model_name: str, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "classes": classes, "state_dict": model.state_dict()}, path)


def load_classifier(path: Path, device: str) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["classes"]


@torch.no_grad()
def classify_crops(model: torch.nn.Module, crops_bgr: list[np.ndarray], device: str) -> torch.Tensor:
    """Probabilités (N, nb_classes) pour des vignettes OpenCV (BGR)."""
    if not crops_bgr:
        return torch.empty(0)
    transform = eval_transform()
    batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr])
    with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        logits = model(batch.to(device))
    # softmax : transforme les scores bruts en probabilités qui somment à 1
    return logits.float().softmax(dim=1).cpu()
