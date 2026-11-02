"""Annotation automatique d'un corpus d'images par le modèle (livrable : corpus de test annoté).

Pour chaque image :
- un fichier XML Pascal VOC au même format que les annotations fournies (une balise <object>
  par avion, avec sa classe et sa boîte), complété d'une balise <score> (confiance du modèle) ;
- une copie de l'image avec les boîtes et les classes dessinées.

Sans --images, le script annote les images de notre split de test (data/splits/test.txt).

Usage :
  python -m src.annotate_corpus --images <dossier du corpus de test> [--zip]
  python -m src.annotate_corpus [--approach yolo] [--conf 0.6] [--no-tta]
Sorties : <out>/annotations/*.xml, <out>/images/*.jpg, <out>/detections.csv (+ <out>.zip)
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.compare_models import validation_thresholds
from src.config import (BASELINE_WEIGHTS, CLASSES, CLASSIFIER_WEIGHTS, DETECTOR_WEIGHTS, RAW_DIR, ROOT, SPLITS_DIR,
                        USE_TTA)
from src.drawing import draw_detections
from src.images import read_image, write_image
from src.pipeline import Detection, TwoStageRecognizer, YoloRecognizer
from src.voc import read_split

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path | None) -> list[Path]:
    if images_dir is None:
        return [RAW_DIR / "images" / f"{i}.jpg" for i in read_split(SPLITS_DIR / "test.txt")]
    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths, key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem))


def to_voc(image_path: Path, width: int, height: int, detections: list[Detection]) -> ET.ElementTree:
    """Même structure que les XML fournis ; coordonnées entières, bornées à l'image."""
    root = ET.Element("annotation")
    for tag, text in (("folder", "Unknown"), ("filename", image_path.name), ("path", "Unknown")):
        ET.SubElement(root, tag).text = text
    size = ET.SubElement(root, "size")
    for tag, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, tag).text = str(value)
    ET.SubElement(root, "segmented").text = "0"
    for d in detections:
        obj = ET.SubElement(root, "object")
        for tag, text in (("name", d.classe), ("pose", "Unspecified"), ("truncated", "0"), ("difficult", "0"),
                          ("score", f"{d.score:.4f}")):
            ET.SubElement(obj, tag).text = text
        box = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = d.as_xyxy()
        for tag, value, limit in (("xmin", x1, width), ("ymin", y1, height), ("xmax", x2, width), ("ymax", y2, height)):
            ET.SubElement(box, tag).text = str(min(max(round(value), 0), limit))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, help="dossier d'images à annoter (défaut : notre split de test)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "corpus_test_annote")
    parser.add_argument("--approach", choices=("pipeline", "yolo"), default="pipeline",
                        help="pipeline = détecteur + classifieur (modèle retenu), yolo = baseline 20 classes")
    parser.add_argument("--conf", type=float, help="seuil de confiance (défaut : celui choisi sur la validation)")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=USE_TTA,
                        help="classification sur les 8 orientations (--no-tta : plus rapide sur CPU)")
    parser.add_argument("--no-images", action="store_true", help="ne pas écrire les images annotées")
    parser.add_argument("--zip", action="store_true", help="créer aussi <out>.zip, à partager")
    args = parser.parse_args()

    recognizer = (TwoStageRecognizer(DETECTOR_WEIGHTS, CLASSIFIER_WEIGHTS, tta=args.tta) if args.approach == "pipeline"
                  else YoloRecognizer(BASELINE_WEIGHTS))
    recognizer.conf = args.conf or validation_thresholds().get(recognizer.label, 0.5)
    images = list_images(args.images)
    print(f"{recognizer.label} | seuil {recognizer.conf:.2f} | {len(images)} images -> {args.out}")

    if args.out.exists():
        shutil.rmtree(args.out)   # pas de mélange avec un ancien passage
    (args.out / "annotations").mkdir(parents=True)
    rows = []
    for path in tqdm(images, unit="image"):
        image = read_image(path)
        detections = sorted(recognizer.predict(image), key=lambda d: -d.score)
        to_voc(path, image.shape[1], image.shape[0], detections).write(
            args.out / "annotations" / f"{path.stem}.xml", encoding="utf-8")
        if not args.no_images:
            rgb = draw_detections(image, detections)
            write_image(args.out / "images" / f"{path.stem}.jpg", np.ascontiguousarray(rgb[:, :, ::-1]))
        rows += [{"image": path.name, "classe": d.classe, "score": round(d.score, 4),
                  "xmin": round(d.xmin), "ymin": round(d.ymin), "xmax": round(d.xmax), "ymax": round(d.ymax)}
                 for d in detections]

    detections = pd.DataFrame(rows, columns=["image", "classe", "score", "xmin", "ymin", "xmax", "ymax"])
    detections.to_csv(args.out / "detections.csv", index=False)
    print(f"{len(detections)} avions annotés dans {len(images)} images")
    print(detections.classe.value_counts().reindex(CLASSES, fill_value=0).to_string())
    if args.zip:
        print("archive :", shutil.make_archive(str(args.out), "zip", args.out))


if __name__ == "__main__":
    main()
