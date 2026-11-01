"""Génération des vignettes (crops) pour entraîner le classifieur.

Chaque avion annoté devient une image carrée centrée sur lui (voir src/crops.py),
rangée dans un dossier par classe : c'est le format attendu par
torchvision.datasets.ImageFolder. Les vignettes héritent du split de leur image
d'origine, donc aucune fuite entre train, val et test.

Les vignettes sont gardées à leur taille d'origine ; le redimensionnement (224x224)
se fera au moment de l'entraînement, avec l'augmentation de données.

Usage : python -m src.make_crops [--margin 0.1] [--skip-truncated]
Sorties : data/crops/{train,val,test}/{A1..A20}/*.jpg et data/crops/crops.csv
"""
import argparse
import shutil

import pandas as pd
from tqdm import tqdm

from src.config import CLASSES, CROPS_DIR, SPLITS, SPLITS_DIR
from src.crops import square_crop
from src.images import read_image, write_image
from src.voc import load_annotations, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, default=0.1, help="marge ajoutée de chaque côté (fraction du plus grand côté)")
    parser.add_argument("--skip-truncated", action="store_true", help="ignorer les avions coupés par le bord de l'image")
    args = parser.parse_args()

    annotations = load_annotations()
    shutil.rmtree(CROPS_DIR, ignore_errors=True)
    rows = []
    for split in SPLITS:
        for name in CLASSES:
            (CROPS_DIR / split / name).mkdir(parents=True)
        for image_id in tqdm(read_split(SPLITS_DIR / f"{split}.txt"), desc=split):
            ann = annotations[image_id]
            image = read_image(ann.image_path)
            for k, box in enumerate(ann.boxes):
                if args.skip_truncated and box.truncated:
                    continue
                path = CROPS_DIR / split / box.name / f"{image_id}_{k:02d}.jpg"
                write_image(path, square_crop(image, box.as_xyxy(), args.margin))
                rows.append({
                    "path": path.relative_to(CROPS_DIR).as_posix(), "split": split, "classe": box.name,
                    "image_id": image_id, "xmin": box.xmin, "ymin": box.ymin, "xmax": box.xmax, "ymax": box.ymax,
                    "tronque": box.truncated,
                })

    crops = pd.DataFrame(rows)
    crops.to_csv(CROPS_DIR / "crops.csv", index=False)
    counts = crops.pivot_table(index="classe", columns="split", values="path", aggfunc="count", fill_value=0)
    print(counts.reindex(index=CLASSES, columns=list(SPLITS)).to_string())
    print(f"{len(crops)} vignettes écrites dans {CROPS_DIR}")


if __name__ == "__main__":
    main()
