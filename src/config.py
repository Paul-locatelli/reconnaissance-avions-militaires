"""Chemins et constantes partagés par tous les scripts du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Données brutes fournies par l'enseignant (jamais modifiées)
RAW_DIR = ROOT / "student-dataset 2"

# Données générées par les scripts
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"   # listes d'identifiants train/val/test (versionnées)
YOLO_DIR = DATA_DIR / "yolo"       # dataset au format YOLO (régénérable)
CROPS_DIR = DATA_DIR / "crops"     # vignettes pour le classifieur (régénérables)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Poids des modèles retenus (non versionnés : produits par l'entraînement)
RUNS_DIR = ROOT / "runs"
BASELINE_WEIGHTS = RUNS_DIR / "yolo11s_20cls" / "weights" / "best.pt"
DETECTOR_WEIGHTS = RUNS_DIR / "yolo11s_avion" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = RUNS_DIR / "cls_resnet50" / "best.pt"

CLASSES = [f"A{i}" for i in range(1, 21)]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SPLITS = ("train", "val", "test")

SEED = 42
