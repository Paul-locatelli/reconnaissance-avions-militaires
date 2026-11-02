"""Les deux approches de reconnaissance, derrière la même interface predict(image).

- YoloRecognizer     : baseline, un seul modèle YOLO 20 classes.
- TwoStageRecognizer : détecteur YOLO 1 classe "avion" -> vignette -> classifieur CNN.
  Score final = confiance de détection x probabilité de la classe choisie.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src.classifier import classify_crops, load_classifier
from src.config import CLASSES
from src.crops import square_crop

BASELINE = "YOLO 20 classes"   # nom dans les rapports et l'interface


@dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    classe: str
    score: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class YoloRecognizer:
    label = BASELINE

    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 800, device: str | None = None):
        self.model, self.conf, self.imgsz = YOLO(weights), conf, imgsz
        self.device = device or default_device()

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                    max_det=300, verbose=False)[0]
        return [Detection(*box, CLASSES[int(c)], float(s)) for box, c, s in
                zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]


class TwoStageRecognizer:
    def __init__(self, detector_weights: Path, classifier_weights: Path, conf: float = 0.25,
                 imgsz: int = 800, margin: float = 0.1, tta: bool = False, device: str | None = None):
        self.device = device or default_device()
        self.detector = YOLO(detector_weights)
        self.classifier, self.classes = load_classifier(classifier_weights, self.device)
        self.conf, self.imgsz, self.margin, self.tta = conf, imgsz, margin, tta
        # Nom de la configuration, ex. « Détecteur + resnet50 + TTA » (dossier runs/cls_<nom>)
        self.label = f"Détecteur + {Path(classifier_weights).parent.name.removeprefix('cls_')}" + (" + TTA" if tta else "")

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        # Étape 1 : où sont les avions ? (seuil bas, le filtrage se fait sur le score final)
        result = self.detector.predict(image_bgr, imgsz=self.imgsz, conf=min(self.conf, 0.01), device=self.device,
                                       max_det=300, verbose=False)[0]
        boxes, det_scores = result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
        # Étape 2 : quel modèle d'avion ? (même découpe de vignette qu'à l'entraînement)
        probs = classify_crops(self.classifier, [square_crop(image_bgr, b, self.margin) for b in boxes], self.device,
                               tta=self.tta)
        detections = []
        for box, det_score, p in zip(boxes, det_scores, probs):
            cls_prob, cls_index = p.max(dim=0)
            score = det_score * float(cls_prob)
            if score >= self.conf:
                detections.append(Detection(*box, self.classes[int(cls_index)], score))
        return detections
