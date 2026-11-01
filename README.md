# Détection et classification d'avions militaires sur images satellites

Projet du module « Introduction à l'IA » (IPSA Toulouse) : localiser les avions dans
une image satellite et déterminer leur modèle (20 classes, A1 à A20).

## Installation

Python 3.12 et [uv](https://docs.astral.sh/uv/) :

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --torch-backend=auto
```

`--torch-backend=auto` installe la version de PyTorch adaptée au GPU NVIDIA de la machine.
`requirements-lock.txt` fige les versions exactes testées (PyTorch 2.14 + CUDA 13.0, Ultralytics 8.4).
Le dataset fourni doit se trouver dans `student-dataset 2/` (`images/` et `annotations/`).

## Préparation des données

À lancer dans l'ordre, depuis la racine du projet :

```bash
.venv/Scripts/python -m src.analyze_dataset   # contrôles, statistiques, figures
.venv/Scripts/python -m src.split_dataset     # split train/val/test par image
.venv/Scripts/python -m src.voc_to_yolo       # dataset au format YOLO
.venv/Scripts/python -m src.make_crops        # vignettes pour le classifieur
```

| Étape | Sortie |
|---|---|
| Analyse | `reports/class_stats.csv`, `reports/figures/*.png` |
| Split | `data/splits/{train,val,test}.txt` (versionné), `reports/split_summary.csv` |
| YOLO | `data/yolo/images/`, `data/yolo/labels/`, `data/yolo/data.yaml` |
| Vignettes | `data/crops/{split}/{classe}/*.jpg`, `data/crops/crops.csv` |

## Choix de préparation

- **Split 70/15/15 par image, stratifié sur la classe dominante.** 79 % des images ne
  contiennent qu'un type d'avion : découper par vignette mettrait des copies du même
  avion en train et en test (fuite de données).
- **Un seul dataset YOLO pour les deux détecteurs.** Le modèle 20 classes l'utilise tel
  quel ; le détecteur « avion » l'utilise avec `single_cls=True`.
- **Vignettes carrées avec 10 % de marge par côté.** Le carré conserve les proportions
  au redimensionnement ; la même fonction (`src/crops.py`) servira à l'inférence.

## Structure

```
src/
├── config.py           chemins et liste des classes
├── voc.py              lecture des annotations Pascal VOC
├── images.py           lecture/écriture d'images
├── crops.py            découpe d'une vignette autour d'une boîte
├── analyze_dataset.py
├── split_dataset.py
├── voc_to_yolo.py
└── make_crops.py
```
