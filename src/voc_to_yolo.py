"""Conversion Pascal VOC -> format YOLO (Ultralytics).

VOC   : une boîte = coins en pixels (xmin, ymin, xmax, ymax), un XML par image.
YOLO  : une boîte = "classe cx cy w h", centre et taille normalisés entre 0 et 1,
        un .txt par image, dans un dossier labels/ parallèle à images/.

Les images ne sont pas copiées mais liées (hardlink) : aucun espace disque en plus.
Le même dataset sert pour le modèle 20 classes et pour le détecteur 1 classe
(option single_cls=True d'Ultralytics, qui fusionne toutes les classes en "avion").

Usage : python -m src.voc_to_yolo
Sorties : data/yolo/{images,labels}/{train,val,test}/ et data/yolo/data.yaml
"""
import os
import shutil

import yaml

from src.config import CLASS_TO_ID, CLASSES, SPLITS, SPLITS_DIR, YOLO_DIR
from src.voc import Annotation, load_annotations, read_split


def to_yolo_lines(ann: Annotation) -> list[str]:
    lines = []
    for b in ann.boxes:
        # On borne les coordonnées à l'image par sécurité
        xmin, xmax = max(0, b.xmin), min(ann.width, b.xmax)
        ymin, ymax = max(0, b.ymin), min(ann.height, b.ymax)
        cx, cy = (xmin + xmax) / 2 / ann.width, (ymin + ymax) / 2 / ann.height
        w, h = (xmax - xmin) / ann.width, (ymax - ymin) / ann.height
        lines.append(f"{CLASS_TO_ID[b.name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def link_or_copy(src, dst) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:  # autre disque ou système de fichiers sans hardlink
        shutil.copy2(src, dst)


def main() -> None:
    annotations = load_annotations()
    for split in SPLITS:
        images_dir, labels_dir = YOLO_DIR / "images" / split, YOLO_DIR / "labels" / split
        # On repart de zéro pour ne pas garder d'images d'un ancien split
        # (supprimer un hardlink ne touche pas à l'image d'origine)
        for d in (images_dir, labels_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        ids = read_split(SPLITS_DIR / f"{split}.txt")
        for image_id in ids:
            ann = annotations[image_id]
            link_or_copy(ann.image_path, images_dir / ann.image_path.name)
            (labels_dir / f"{image_id}.txt").write_text("\n".join(to_yolo_lines(ann)) + "\n")
        print(f"{split:5} : {len(ids)} images converties")

    data_yaml = {
        "path": str(YOLO_DIR),  # chemin absolu : à régénérer sur chaque machine
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    with open(YOLO_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Configuration écrite : {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
