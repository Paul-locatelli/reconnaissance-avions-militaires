"""Extraction de vignettes centrées sur un avion.

Cette fonction sert à la fois pour générer le dataset du classifieur et, plus tard,
dans le pipeline d'inférence (boîte prédite par le détecteur -> vignette -> classifieur).
Utiliser la même fonction aux deux endroits garantit que le classifieur voit en
production des vignettes construites exactement comme à l'entraînement.
"""
import cv2
import numpy as np


def square_crop(image: np.ndarray, box: tuple[float, float, float, float], margin: float = 0.1) -> np.ndarray:
    """Découpe un carré centré sur la boîte, de côté max(w, h) * (1 + 2 * margin).

    Un carré conserve les proportions de l'avion lors du redimensionnement en 224x224
    et laisse de la place pour les rotations. La partie qui sort de l'image est
    complétée par du noir.
    """
    h_img, w_img = image.shape[:2]
    xmin, ymin, xmax, ymax = box
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    side = int(round(max(xmax - xmin, ymax - ymin) * (1 + 2 * margin)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side

    crop = image[max(y0, 0):min(y1, h_img), max(x0, 0):min(x1, w_img)]
    top, bottom = max(-y0, 0), max(y1 - h_img, 0)
    left, right = max(-x0, 0), max(x1 - w_img, 0)
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return crop
