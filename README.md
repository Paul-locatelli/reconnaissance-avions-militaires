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

## Entraînement et évaluation

Deux approches sont comparées :

1. **YOLO 20 classes** (baseline) : un seul modèle localise et reconnaît le type d'avion.
2. **Détecteur + classifieur** : un YOLO 1 classe trouve les avions, puis un CNN (ResNet50)
   reconnaît le type sur chaque vignette. Score final = confiance de détection × probabilité de la classe.

```bash
.venv/Scripts/python -m src.train_yolo --name yolo11s_20cls                  # baseline 20 classes
.venv/Scripts/python -m src.evaluate_yolo --name yolo11s_20cls               # score Ultralytics sur le test
.venv/Scripts/python -m src.train_yolo --name yolo11s_avion --single-cls     # détecteur 1 classe
.venv/Scripts/python -m src.evaluate_yolo --name yolo11s_avion --single-cls
.venv/Scripts/python -m src.train_classifier --name resnet50                 # classifieur de vignettes
.venv/Scripts/python -m src.compare_models                                   # comparaison des 2 approches
```

Après un plantage, `python -m src.train_yolo --name <nom> --resume` reprend depuis le dernier
checkpoint, et `python -m src.train_classifier --name <nom> --eval-only` refait seulement le test.

Les poids sont écrits dans `runs/` (non versionné), les scores dans `reports/`.

### Résultats sur le test (200 images, 1287 avions)

Comparaison avec le même code de métriques (`src/detection_metrics.py`, style COCO) ; le seuil
de confiance est choisi sur la validation (meilleur F1), puis appliqué au test.

| Approche | mAP50 | mAP50-95 | Précision | Rappel | F1 | Temps / image |
|---|---|---|---|---|---|---|
| YOLO 20 classes | 0,967 | 0,793 | 0,971 | 0,950 | 0,960 | 41 ms |
| Détecteur + classifieur | 0,969 | 0,793 | 0,984 | 0,944 | 0,964 | 78 ms |

- **Égalité globale** : l'écart (F1 0,964 contre 0,960) est trop faible pour départager les
  approches sur 200 images. Le pipeline est plus précis mais rate un peu plus d'avions, et il
  est deux fois plus lent (GPU RTX 4060 Laptop).
- **Forces complémentaires** : le pipeline corrige les classes difficiles de la baseline
  (F1 A5 0,87 contre 0,79 ; A11 0,88 contre 0,81 ; A14 0,97 contre 0,91 ; A15 0,96 contre 0,89),
  mais fait moins bien sur A1 (0,88 contre 0,94), A12 (0,93 contre 0,99) et A20 (0,96 contre 0,98).
  Détail par classe : `reports/comparaison_test.csv`.
- **Les erreurs du pipeline viennent de la classification** : seul, le détecteur 1 classe trouve
  99,5 % des avions (précision 0,993, AP50 0,994, AP50-95 0,810, `reports/yolo11s_avion_test.csv`).
  Sur les vignettes du test, le classifieur atteint 97,8 % d'exactitude (F1 macro 0,977) ; ses
  confusions principales sont A1 ↔ A19 et A11 → A14 (`reports/figures/cls_resnet50_confusion_matrix.png`).

Entraînements :

- **YOLO11s 20 classes** : pré-entraîné COCO, 800 px, 100 époques (1 h), meilleure époque 78.
  Principale erreur : 29 % des A5 prédits A13, presque tous dans une seule image (539, ombres
  très marquées), voir `reports/figures/yolo11s_20cls_erreur_image539.jpg`.
- **YOLO11s 1 classe** : mêmes réglages avec `single_cls`, 100 époques, meilleure époque 100.
- **ResNet50** : pré-entraîné ImageNet, vignettes 224 px, perte pondérée par classe, arrêt
  anticipé à l'époque 24, meilleure époque 16 (F1 macro de validation 0,955).

Ultralytics annonce un mAP50 plus élevé pour la baseline (0,989 contre 0,967 ici), pour deux raisons :
son évaluation garde plusieurs classes candidates par boîte, ce qui gonfle le mAP d'un modèle
multi-classes, et son AP interpole la courbe précision/rappel au lieu de la lire en marches comme COCO.
Sur le détecteur 1 classe, où la première raison ne joue pas, les deux calculs ne diffèrent que de
0,009 en AP50 et de 0,006 en AP50-95.

## Interface de démonstration

```bash
.venv/Scripts/python -m src.app        # puis ouvrir http://127.0.0.1:7860
```

- Déposer une image satellite, ou choisir un des 6 exemples (images du test, jamais vues à l'entraînement).
- Choisir l'approche ; le seuil de confiance prend la valeur choisie sur la validation, et reste réglable.
- Si l'image vient du dataset, chaque prédiction est comparée à la vérité terrain : vert = correct,
  orange = mauvaise classe, rouge = fausse détection, pointillés rouges = avion manqué.
- Le tableau liste chaque détection (classe, score, verdict, coordonnées de la boîte).

Les modèles sont chargés une fois au démarrage, avec un premier passage à vide pour que la
première analyse de la démo soit rapide. Options `--host 0.0.0.0` (accès depuis un autre poste)
et `--baseline / --detector / --classifier` (autres poids).

## Corpus de test annoté

```bash
.venv/Scripts/python -m src.annotate_corpus --images <dossier du corpus de test> --zip
.venv/Scripts/python -m src.annotate_corpus          # sans --images : les 200 images de notre split de test
```

Le modèle retenu (détecteur + classifieur, seuil de validation 0,60) annote chaque image :

- `outputs/corpus_test_annote/annotations/*.xml` : Pascal VOC, même format que les annotations
  fournies, avec en plus une balise `<score>` (confiance du modèle) par avion ;
- `outputs/corpus_test_annote/images/*.jpg` : images avec les boîtes et les classes dessinées ;
- `outputs/corpus_test_annote/detections.csv` : toutes les détections dans un seul tableau ;
- `outputs/corpus_test_annote.zip` (option `--zip`) : l'ensemble, à partager.

Contrôle : relus comme des annotations et comparés à la vérité terrain, les XML de notre split de
test redonnent exactement les scores de la comparaison (précision 0,984, rappel 0,944, F1 0,964).
Les XML et le CSV de ce split sont versionnés ; les images et le zip, trop lourds, ne le sont pas.

## Docker

L'image contient le code et les poids des 3 modèles, et fait l'inférence sur CPU : elle tourne
sur n'importe quelle machine, sans GPU. Les poids doivent être présents dans `runs/` avant la
construction (ils ne sont pas versionnés).

```bash
docker build -t avions-militaires .
docker run --rm -p 7860:7860 avions-militaires          # puis ouvrir http://localhost:7860
```

Avec le dataset monté, l'interface propose les exemples et la comparaison à la vérité terrain
(sous PowerShell, remplacer `$(pwd)` par `${PWD}`) :

```bash
docker run --rm -p 7860:7860 -v "$(pwd)/student-dataset 2:/app/student-dataset 2:ro" avions-militaires
```

Annoter un corpus avec le conteneur :

```bash
docker run --rm -v "<dossier du corpus>:/corpus:ro" -v "$(pwd)/outputs:/app/outputs" avions-militaires \
    python -m src.annotate_corpus --images /corpus --zip
```

Choix : PyTorch en version CPU (image bien plus légère que la version CUDA), versions exactes
reprises de `requirements-lock.txt`, et `.dockerignore` en liste blanche pour ne copier que le
nécessaire. Sur CPU, le pipeline prend de 0,4 à 1,4 s par image selon le nombre d'avions.

## Structure

```
src/
├── config.py           chemins et liste des classes
├── voc.py              lecture des annotations Pascal VOC
├── images.py           lecture/écriture d'images
├── crops.py            découpe d'une vignette autour d'une boîte
├── plotting.py         style commun des figures
├── analyze_dataset.py
├── split_dataset.py
├── voc_to_yolo.py
├── make_crops.py
├── train_yolo.py       entraînement YOLO (20 classes ou 1 classe)
├── evaluate_yolo.py    métriques Ultralytics par classe sur le test
├── classifier.py       CNN de vignettes : modèle, transformations, chargement
├── train_classifier.py entraînement et test du classifieur
├── pipeline.py         les deux approches derrière la même interface predict(image)
├── detection_metrics.py  AP, mAP, précision, rappel, F1 codés à la main (style COCO)
├── compare_models.py   comparaison des deux approches sur le test
├── drawing.py          dessin des boîtes et des verdicts
├── app.py              interface de démonstration (Gradio)
└── annotate_corpus.py  annotation d'un corpus d'images (XML Pascal VOC + images annotées)
```
