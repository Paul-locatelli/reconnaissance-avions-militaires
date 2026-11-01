"""Découpage train / val / test (70 / 15 / 15) au niveau des IMAGES.

Pourquoi par image et pas par avion : 79 % des images ne contiennent qu'un seul type
d'avion, garé à côté de copies quasi identiques. Si on découpait par vignette, des
avions de la même image se retrouveraient en train ET en test : le score de test
serait artificiellement élevé (fuite de données).

Le split est stratifié sur la classe dominante de chaque image, pour que les classes
rares (A6 : 29 images seulement) soient présentes dans les trois sous-ensembles.

Usage : python -m src.split_dataset
Sorties : data/splits/{train,val,test}.txt et reports/split_summary.csv
"""
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASSES, REPORTS_DIR, SEED, SPLITS, SPLITS_DIR
from src.voc import Annotation, load_annotations


def dominant_class(ann: Annotation, global_counts: Counter) -> str:
    """Classe la plus fréquente dans l'image ; en cas d'égalité, la plus rare du dataset."""
    counts = Counter(b.name for b in ann.boxes)
    top = max(counts.values())
    return min((name for name, c in counts.items() if c == top), key=lambda n: global_counts[n])


def summarize(splits: dict[str, list[str]], annotations: dict[str, Annotation]) -> pd.DataFrame:
    rows = []
    for name in CLASSES:
        row = {"classe": name}
        for split, ids in splits.items():
            row[f"{split}_avions"] = sum(b.name == name for i in ids for b in annotations[i].boxes)
            row[f"{split}_images"] = sum(any(b.name == name for b in annotations[i].boxes) for i in ids)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    annotations = load_annotations()
    global_counts = Counter(b.name for a in annotations.values() for b in a.boxes)
    ids = list(annotations)
    strata = [dominant_class(annotations[i], global_counts) for i in ids]

    train_ids, rest_ids, _, rest_strata = train_test_split(
        ids, strata, test_size=0.30, stratify=strata, random_state=SEED)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, stratify=rest_strata, random_state=SEED)
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (SPLITS_DIR / f"{split}.txt").write_text("\n".join(sorted(splits[split], key=int)) + "\n")

    summary = summarize(splits, annotations)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "split_summary.csv", index=False)

    for split in SPLITS:
        n_boxes = sum(len(annotations[i].boxes) for i in splits[split])
        print(f"{split:5} : {len(splits[split]):4} images, {n_boxes:4} avions")
    print(summary.to_string(index=False))
    missing = [(row.classe, s) for row in summary.itertuples() for s in SPLITS if getattr(row, f"{s}_avions") == 0]
    if missing:
        print("ATTENTION, classes absentes d'un sous-ensemble :", missing)


if __name__ == "__main__":
    main()
