"""Dessin des détections sur une image (interface de démo, corpus annoté).

Pillow plutôt qu'OpenCV pour le texte : les polices d'OpenCV n'ont pas d'accents.
La couleur n'est jamais la seule information : l'étiquette dit aussi le verdict.
"""
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from src.detection_metrics import CORRECT, FALSE_POSITIVE, WRONG_CLASS

FONT_PATH = font_manager.findfont("DejaVu Sans")   # fournie avec matplotlib, accents compris, sur tout système

PREDICTION, MISSED = "prédiction", "manqué"
COLORS = {  # RGB
    PREDICTION: (42, 120, 214),     # bleu : pas de vérité terrain disponible
    CORRECT: (31, 157, 85),         # vert
    WRONG_CLASS: (232, 137, 12),    # orange : bon avion, mauvais type
    FALSE_POSITIVE: (217, 58, 58),  # rouge : rien à cet endroit
    MISSED: (217, 58, 58),          # rouge en pointillés : avion non trouvé
}
LEGEND = ("**Couleurs** : vert = correct · orange = mauvaise classe · rouge = fausse détection · "
          "pointillés rouges = avion manqué")
LEGEND_NO_TRUTH = "**Couleur** : bleu = prédiction (pas de vérité terrain pour cette image, ou comparaison désactivée)"


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box, color, width: int, dash: int = 6) -> None:
    x1, y1, x2, y2 = box
    for start, end, horizontal in (((x1, y1), x2, True), ((x1, y2), x2, True), ((x1, y1), y2, False), ((x2, y1), y2, False)):
        x, y = start
        pos = x if horizontal else y
        while pos < end:
            stop = min(pos + dash, end)
            draw.line([(pos, y), (stop, y)] if horizontal else [(x, pos), (x, stop)], fill=color, width=width)
            pos += 2 * dash


def _label(draw: ImageDraw.ImageDraw, box, text: str, color, font) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 6, bottom - top + 4
    x, y = box[0], box[1] - h if box[1] >= h else box[1]   # au-dessus de la boîte, sinon dedans
    draw.rectangle([x, y, x + w, y + h], fill=color)
    draw.text((x + 3 - left, y + 2 - top), text, fill=(255, 255, 255), font=font)


def draw_detections(image_bgr: np.ndarray, detections: list, verdicts: list | None = None,
                    missed: list = (), labels: bool = True) -> np.ndarray:
    """Image RGB annotée. verdicts : (verdict, vraie classe) par détection, ou None sans vérité terrain."""
    image = Image.fromarray(image_bgr[:, :, ::-1].copy())
    draw = ImageDraw.Draw(image)
    width = max(2, round(max(image.size) / 400))
    font = ImageFont.truetype(FONT_PATH, size=max(11, round(max(image.size) / 70)))
    for box in missed:
        _dashed_rectangle(draw, box.as_xyxy(), COLORS[MISSED], width)
        if labels:
            _label(draw, box.as_xyxy(), f"manqué {box.name}", COLORS[MISSED], font)
    for k, d in enumerate(detections):
        verdict, truth = verdicts[k] if verdicts else (PREDICTION, None)
        draw.rectangle(d.as_xyxy(), outline=COLORS[verdict], width=width)
        if labels:
            text = f"{d.classe} {d.score:.2f}"
            if verdict == WRONG_CLASS:
                text += f" (vrai {truth})"
            elif verdict == FALSE_POSITIVE:
                text += " (faux)"
            _label(draw, d.as_xyxy(), text, COLORS[verdict], font)
    return np.asarray(image)
