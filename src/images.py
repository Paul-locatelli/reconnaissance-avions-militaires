"""Lecture / écriture d'images robustes aux chemins Windows (espaces, accents)."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Image BGR (convention OpenCV). cv2.imread échoue sur certains chemins Windows."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Impossible de lire l'image {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise IOError(f"Impossible d'encoder l'image {path}")
    buffer.tofile(str(path))
