"""Entraînement d'un modèle YOLO (Ultralytics) sur le dataset converti.

Deux usages :
- baseline de bout en bout : 20 classes, détection et classification en une passe ;
- détecteur du pipeline en 2 étapes : --single-cls fusionne toutes les classes en "avion".

On part de poids pré-entraînés sur COCO (transfer learning) : le réseau sait déjà
extraire des formes génériques, il n'a plus qu'à s'adapter aux avions vus du ciel.

Usage :
  python -m src.train_yolo --name yolo11s_20cls
  python -m src.train_yolo --name yolo11s_avion --single-cls
  python -m src.train_yolo --name yolo11s_avion --resume     (reprise après un plantage)
Sorties : runs/<name>/ (poids weights/best.pt, courbes, matrice de confusion)
"""
import argparse

from ultralytics import YOLO

from src.config import ROOT, SEED, YOLO_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom de l'expérience (dossier dans runs/)")
    parser.add_argument("--model", default="yolo11s.pt", help="poids pré-entraînés de départ")
    parser.add_argument("--single-cls", action="store_true", help="une seule classe 'avion' (détecteur du pipeline)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reprendre runs/<name> depuis weights/last.pt après un arrêt")
    args = parser.parse_args()

    if args.resume:
        # Réglages, optimiseur et époque courante sont relus dans le checkpoint : les autres options sont ignorées
        YOLO(ROOT / "runs" / args.name / "weights" / "last.pt").train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=str(YOLO_DIR / "data.yaml"),
        single_cls=args.single_cls,
        epochs=args.epochs,
        patience=30,            # arrêt anticipé si le mAP de validation stagne 30 époques
        imgsz=args.imgsz,       # taille native : on ne réduit pas les petits avions
        batch=args.batch,
        workers=args.workers,
        cache=True,             # images gardées en RAM (environ 2 Go) : époques plus rapides
        # Augmentation adaptée à la vue de dessus : un avion retourné reste un avion
        flipud=0.5,
        fliplr=0.5,
        degrees=0.0,            # une rotation quelconque élargirait les boîtes horizontales
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        seed=SEED,
        plots=True,
    )


if __name__ == "__main__":
    main()
