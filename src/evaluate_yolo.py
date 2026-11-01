"""Évaluation d'un modèle YOLO entraîné, sur le sous-ensemble de TEST.

Le test n'a servi ni à l'entraînement ni au choix du meilleur modèle (fait sur val) :
c'est l'estimation honnête de la performance sur des images jamais vues.

Métriques :
- précision : parmi les avions détectés, part de détections correctes ;
- rappel    : parmi les vrais avions, part retrouvée ;
- AP50      : aire sous la courbe précision/rappel, une détection compte si elle
              recouvre la vérité à au moins 50 % (IoU >= 0.5) ;
- AP50-95   : moyenne des AP pour des IoU de 0.5 à 0.95, plus exigeante sur la
              précision des boîtes. mAP = moyenne des AP sur les classes.

Usage : python -m src.evaluate_yolo --name yolo11s_20cls [--single-cls]
Sorties : reports/<name>_test.csv, reports/figures/<name>_confusion_matrix.png
"""
import argparse
import shutil

import pandas as pd
from ultralytics import YOLO

from src.config import FIGURES_DIR, REPORTS_DIR, ROOT, YOLO_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="expérience à évaluer (dossier dans runs/)")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--imgsz", type=int, default=800)
    args = parser.parse_args()

    model = YOLO(ROOT / "runs" / args.name / "weights" / "best.pt")
    out_name = f"{args.name}_test"
    metrics = model.val(
        data=str(YOLO_DIR / "data.yaml"), split="test", imgsz=args.imgsz, batch=16,
        single_cls=args.single_cls, project=str(ROOT / "runs"), name=out_name, exist_ok=True, plots=True,
    )

    box = metrics.box
    rows = []
    for i, class_index in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        name = "avion" if args.single_cls else metrics.names[class_index]
        rows.append({"classe": name, "precision": p, "rappel": r, "f1": 2 * p * r / (p + r + 1e-9),
                     "AP50": ap50, "AP50_95": ap})
    rows.append({"classe": "moyenne", "precision": box.mp, "rappel": box.mr,
                 "f1": 2 * box.mp * box.mr / (box.mp + box.mr + 1e-9), "AP50": box.map50, "AP50_95": box.map})
    results = pd.DataFrame(rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / f"{out_name}.csv", index=False, float_format="%.4f")
    confusion = ROOT / "runs" / out_name / "confusion_matrix_normalized.png"
    if confusion.exists():
        shutil.copy2(confusion, FIGURES_DIR / f"{args.name}_confusion_matrix.png")

    print(results.round(3).to_string(index=False))
    print(f"\nTEST {args.name} : mAP50 = {box.map50:.3f} | mAP50-95 = {box.map:.3f}")


if __name__ == "__main__":
    main()
