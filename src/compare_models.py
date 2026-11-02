"""Comparaison équitable des deux approches sur le TEST, avec le même code de métriques.

1. Chaque approche prédit sur val et test avec un seuil très bas (nécessaire pour l'AP).
2. Le seuil de confiance « de production » est choisi sur la VALIDATION (meilleur F1).
3. Les métriques sont calculées sur le TEST avec ce seuil.

Usage : python -m src.compare_models [--classifier cls_resnet50] [--device cpu]
Sorties : reports/comparaison_test.csv (par classe), reports/comparaison_resume.csv (seuil, scores, vitesse)
"""
import argparse
import time

import pandas as pd

from src.config import CLASSES, REPORTS_DIR, ROOT, SPLITS_DIR
from src.detection_metrics import best_conf_threshold, evaluate
from src.images import read_image
from src.pipeline import BASELINE, TWO_STAGE, TwoStageRecognizer, YoloRecognizer
from src.voc import load_annotations, read_split

SUMMARY_PATH = REPORTS_DIR / "comparaison_resume.csv"


def validation_thresholds() -> dict[str, float]:
    """Seuil retenu pour chaque approche (meilleur F1 de validation), lu dans le résumé de la comparaison."""
    if not SUMMARY_PATH.exists():
        return {}
    return {a: float(t) for a, t in pd.read_csv(SUMMARY_PATH)[["approche", "seuil (val)"]].values}


def predict_split(recognizer, annotations: dict) -> tuple[dict, float]:
    """Prédictions pour chaque image, et temps moyen par image en millisecondes."""
    predictions, start = {}, time.perf_counter()
    for image_id, ann in annotations.items():
        predictions[image_id] = recognizer.predict(read_image(ann.image_path))
    return predictions, (time.perf_counter() - start) / len(annotations) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="yolo11s_20cls")
    parser.add_argument("--detector", default="yolo11s_avion")
    parser.add_argument("--classifier", default="cls_resnet50")
    parser.add_argument("--device", default=None, help="cuda ou cpu (par défaut : GPU si disponible)")
    args = parser.parse_args()

    runs = ROOT / "runs"
    recognizers = {BASELINE: YoloRecognizer(runs / args.baseline / "weights" / "best.pt", conf=0.01, device=args.device)}
    detector, classifier = runs / args.detector / "weights" / "best.pt", runs / args.classifier / "best.pt"
    if detector.exists() and classifier.exists():
        recognizers[TWO_STAGE] = TwoStageRecognizer(detector, classifier, conf=0.01, device=args.device)
    else:
        print(f"Pipeline 2 étapes ignoré : {detector if not detector.exists() else classifier} introuvable")

    annotations = load_annotations()
    val = {i: annotations[i] for i in read_split(SPLITS_DIR / "val.txt")}
    test = {i: annotations[i] for i in read_split(SPLITS_DIR / "test.txt")}

    per_class, summary = [], []
    for label, recognizer in recognizers.items():
        val_preds, _ = predict_split(recognizer, val)
        threshold = best_conf_threshold(val_preds, val, CLASSES)
        test_preds, ms = predict_split(recognizer, test)
        results = evaluate(test_preds, test, CLASSES, threshold)
        results.insert(0, "approche", label)
        per_class.append(results)
        g = results.iloc[-1]
        summary.append({"approche": label, "seuil (val)": threshold, "mAP50": g.AP50, "mAP50-95": g.AP50_95,
                        "precision": g.precision, "rappel": g.rappel, "f1": g.f1, "ms/image": ms})

    all_results = pd.concat(per_class, ignore_index=True)
    all_results.to_csv(REPORTS_DIR / "comparaison_test.csv", index=False, float_format="%.4f")
    for metric in ("f1", "AP50_95"):
        print(f"\n{metric} par classe (test)")
        print(all_results.pivot(index="classe", columns="approche", values=metric)
              .reindex(CLASSES + ["global"]).round(3).to_string())
    summary = pd.DataFrame(summary)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.4f")
    print("\nRésumé (test)")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
