"""Interface de démonstration (Gradio) : détection et classification d'avions militaires.

On dépose une image satellite, ou on choisit un exemple du test : l'interface affiche les
avions trouvés, leur classe et leur score, avec l'approche choisie. Si l'image vient du
dataset, chaque prédiction est comparée à la vérité terrain (couleur et verdict).

Usage : python -m src.app [--host 127.0.0.1] [--port 7860]
Puis ouvrir http://127.0.0.1:7860
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from src.compare_models import validation_thresholds
from src.config import BASELINE_WEIGHTS, CLASSIFIER_WEIGHTS, DETECTOR_WEIGHTS, RAW_DIR, SPLITS_DIR
from src.detection_metrics import CORRECT, FALSE_POSITIVE, WRONG_CLASS, compare_to_ground_truth
from src.drawing import LEGEND, LEGEND_NO_TRUTH, draw_detections
from src.images import read_image
from src.pipeline import BASELINE, TWO_STAGE, TwoStageRecognizer, YoloRecognizer
from src.voc import load_annotations, read_split

EXAMPLES = ["3607", "1749", "3821", "1680", "2477", "539"]  # images du test, jamais vues à l'entraînement
COLUMNS = ["n°", "classe", "score", "verdict", "x1", "y1", "x2", "y2"]


def count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n <= 1 else plural}"


def load_recognizers(args) -> dict:
    """Les approches dont les poids sont présents ; seuil bas, le filtrage se fait dans l'interface."""
    recognizers = {}
    detector, classifier = Path(args.detector), Path(args.classifier)
    if detector.exists() and classifier.exists():
        recognizers[TWO_STAGE] = TwoStageRecognizer(detector, classifier, conf=0.05)
    if Path(args.baseline).exists():
        recognizers[BASELINE] = YoloRecognizer(Path(args.baseline), conf=0.05)
    if not recognizers:
        raise SystemExit("Aucun poids trouvé : entraîner les modèles ou passer --baseline / --detector / --classifier")
    for recognizer in recognizers.values():   # premier passage (initialisation CUDA) avant la démo
        recognizer.predict(np.zeros((800, 800, 3), np.uint8))
    return recognizers


class Demo:
    def __init__(self, recognizers: dict):
        self.recognizers = recognizers
        self.annotations = load_annotations() if (RAW_DIR / "annotations").exists() else {}
        self.test_ids = set(read_split(SPLITS_DIR / "test.txt")) if self.annotations else set()

    def ground_truth(self, image_path: str, image: np.ndarray):
        """Annotation de l'image si elle vient du dataset (même nom et mêmes dimensions)."""
        ann = self.annotations.get(Path(image_path).stem)
        return ann if ann is not None and (ann.height, ann.width) == image.shape[:2] else None

    def analyse(self, image_path, approach, threshold, compare, labels):
        if not image_path:
            return None, "Déposez une image satellite ou choisissez un exemple.", pd.DataFrame(columns=COLUMNS)
        image = read_image(Path(image_path))
        recognizer = self.recognizers[approach]
        start = time.perf_counter()
        detections = sorted((d for d in recognizer.predict(image) if d.score >= threshold), key=lambda d: -d.score)
        ms = (time.perf_counter() - start) * 1000

        truth = self.ground_truth(image_path, image) if compare else None
        verdicts, missed = compare_to_ground_truth(detections, truth.boxes) if truth else (None, [])
        annotated = draw_detections(image, detections, verdicts, missed, labels)

        per_class = Counter(d.classe for d in detections)
        lines = [f"### {count(len(detections), 'avion détecté', 'avions détectés')}",
                 f"{approach} · seuil {threshold:.2f} · {ms:.0f} ms ({recognizer.device.upper()})",
                 "**Par classe** : " + (" · ".join(f"{c} × {n}" for c, n in sorted(per_class.items(),
                                                     key=lambda x: int(x[0][1:]))) or "aucune")]
        if truth:
            counts = Counter(v for v, _ in verdicts)
            origin = "image du test" if truth.image_id in self.test_ids else "image vue à l'entraînement ou en validation"
            lines.append(f"**Vérité terrain** ({origin}, {count(len(truth.boxes), 'avion annoté', 'avions annotés')}) : "
                         f"{count(counts[CORRECT], 'correct', 'corrects')} · "
                         f"{count(counts[WRONG_CLASS], 'mauvaise classe', 'mauvaises classes')} · "
                         f"{count(counts[FALSE_POSITIVE], 'fausse détection', 'fausses détections')} · "
                         f"{count(len(missed), 'manqué', 'manqués')}")
        lines.append(LEGEND if truth else LEGEND_NO_TRUTH)

        rows = [{"n°": k + 1, "classe": d.classe, "score": round(d.score, 3),
                 "verdict": (f"{verdicts[k][0]} (vrai {verdicts[k][1]})" if verdicts[k][0] == WRONG_CLASS
                             else verdicts[k][0]) if verdicts else "",
                 "x1": round(d.xmin), "y1": round(d.ymin), "x2": round(d.xmax), "y2": round(d.ymax)}
                for k, d in enumerate(detections)]
        rows += [{"n°": "", "classe": b.name, "score": "", "verdict": "manqué",
                  "x1": b.xmin, "y1": b.ymin, "x2": b.xmax, "y2": b.ymax} for b in missed]
        return annotated, "\n\n".join(lines), pd.DataFrame(rows, columns=COLUMNS)


def build_app(demo: Demo) -> gr.Blocks:
    thresholds = validation_thresholds()
    approaches = list(demo.recognizers)
    examples = [[str(RAW_DIR / "images" / f"{i}.jpg")] for i in EXAMPLES if (RAW_DIR / "images" / f"{i}.jpg").exists()]

    with gr.Blocks(title="Reconnaissance d'avions militaires") as app:
        gr.Markdown("# Reconnaissance d'avions militaires\n"
                    "Localisation et identification du modèle (20 classes, A1 à A20) sur images satellites.")
        with gr.Row():
            with gr.Column(scale=2):
                image_in = gr.Image(type="filepath", label="Image satellite", sources=["upload", "clipboard"], height=360)
                approach = gr.Radio(approaches, value=approaches[0], label="Approche")
                threshold = gr.Slider(0.05, 0.95, value=thresholds.get(approaches[0], 0.5), step=0.05,
                                      label="Seuil de confiance", info="Par défaut : seuil choisi sur la validation")
                with gr.Row():
                    compare = gr.Checkbox(value=True, label="Comparer à la vérité terrain")
                    labels = gr.Checkbox(value=True, label="Étiquettes")
                run = gr.Button("Analyser", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=image_in, label="Exemples (images du test)", examples_per_page=6)
            with gr.Column(scale=3):
                image_out = gr.Image(label="Résultat", type="numpy", buttons=["download", "fullscreen"])
                summary = gr.Markdown()
        table = gr.Dataframe(headers=COLUMNS, label="Détections", interactive=False, max_height=320)

        inputs, outputs = [image_in, approach, threshold, compare, labels], [image_out, summary, table]
        run.click(demo.analyse, inputs, outputs)
        image_in.change(demo.analyse, inputs, outputs)
        threshold.release(demo.analyse, inputs, outputs)
        compare.change(demo.analyse, inputs, outputs)
        labels.change(demo.analyse, inputs, outputs)
        # Changer d'approche remet son seuil de validation, puis relance l'analyse
        approach.change(lambda a: thresholds.get(a, 0.5), approach, threshold).then(demo.analyse, inputs, outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default=str(BASELINE_WEIGHTS))
    parser.add_argument("--detector", default=str(DETECTOR_WEIGHTS))
    parser.add_argument("--classifier", default=str(CLASSIFIER_WEIGHTS))
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour un accès depuis un autre poste ou Docker")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(Demo(load_recognizers(args)))
    app.launch(server_name=args.host, server_port=args.port, theme=gr.themes.Soft(), footer_links=[],
               allowed_paths=[str(RAW_DIR / "images")])   # les exemples sont lus dans le dataset


if __name__ == "__main__":
    main()
