"""Entraînement du classifieur de vignettes (étape 2 du pipeline).

- Transfer learning : CNN pré-entraîné sur ImageNet, toutes les couches ré-entraînées.
- Déséquilibre des classes : perte pondérée (une erreur sur A6, rare, coûte plus cher
  qu'une erreur sur A16, fréquente).
- Le meilleur modèle est choisi sur le F1 macro de VALIDATION ; le test n'est utilisé
  qu'une fois, à la fin.

Usage : python -m src.train_classifier --name resnet50 [--model resnet50] [--epochs 30]
        python -m src.train_classifier --name resnet50 --eval-only   (test seul, modèle déjà entraîné)
Sorties : runs/cls_<name>/best.pt et history.csv,
          reports/cls_<name>_test.csv, reports/figures/cls_<name>_confusion_matrix.png
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.classifier import build_model, eval_transform, load_classifier, save_checkpoint, train_transform
from src.config import CLASS_TO_ID, CLASSES, CROPS_DIR, FIGURES_DIR, REPORTS_DIR, ROOT, SEED
from src.plotting import plot_confusion_matrix


class CropFolder(ImageFolder):
    """ImageFolder trie les dossiers par ordre alphabétique (A1, A10, A11...) : on impose A1..A20."""

    def find_classes(self, directory):
        return CLASSES, CLASS_TO_ID


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    """Une passe sur le loader ; entraîne si un optimiseur est fourni. Renvoie perte, vérités, prédictions."""
    training = optimizer is not None
    model.train(training)
    total_loss, targets, preds = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()   # rétropropagation : calcul des gradients
                scaler.step(optimizer)          # descente de gradient : mise à jour des poids
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * len(labels)
            targets += labels.tolist()
            preds += logits.argmax(dim=1).tolist()
    return total_loss / len(targets), np.array(targets), np.array(preds)


def make_loader(dataset, batch: int, workers: int, shuffle: bool = False, persistent: bool = False) -> DataLoader:
    # Sous Windows, chaque processus de chargement recharge PyTorch et CUDA (environ 1 Go de mémoire virtuelle)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      persistent_workers=persistent and workers > 0)


def train(args, device: str, out_dir) -> None:
    """Entraîne et garde dans out_dir/best.pt le modèle au meilleur F1 macro de validation."""
    train_set = CropFolder(CROPS_DIR / "train", train_transform())
    val_set = CropFolder(CROPS_DIR / "val", eval_transform())
    # Processus persistants : pas de relance à chaque époque ; ils sont libérés à la sortie de la fonction
    train_loader = make_loader(train_set, args.batch, args.workers, shuffle=True, persistent=True)
    val_loader = make_loader(val_set, args.batch, args.workers, persistent=True)

    # Poids inversement proportionnels à la fréquence de chaque classe dans le train
    counts = np.bincount(train_set.targets, minlength=len(CLASSES))
    class_weights = torch.tensor(len(train_set) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    model = build_model(args.model, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Montée progressive du taux d'apprentissage (warmup) puis décroissance en cosinus
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    print(f"{args.model} | {len(train_set)} / {len(val_set)} vignettes | {device}", flush=True)
    history, best_f1, epochs_without_progress = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, y_tr, p_tr = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        val_loss, y_val, p_val = run_epoch(model, val_loader, criterion, device)
        val_f1 = f1_score(y_val, p_val, average="macro")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": accuracy_score(y_tr, p_tr),
                        "val_loss": val_loss, "val_acc": accuracy_score(y_val, p_val), "val_f1_macro": val_f1})
        improved = val_f1 > best_f1
        if improved:
            best_f1, epochs_without_progress = val_f1, 0
            save_checkpoint(out_dir / "best.pt", model, args.model, CLASSES)
        else:
            epochs_without_progress += 1
        h = history[-1]
        print(f"époque {epoch:2d} | perte {train_loss:.3f} / {val_loss:.3f} | acc {h['train_acc']:.3f} / {h['val_acc']:.3f}"
              f" | F1 macro val {val_f1:.3f}{' *' if improved else ''} | {time.time() - start:.0f} s", flush=True)
        if epochs_without_progress >= args.patience:
            print(f"Arrêt anticipé : pas de progrès depuis {args.patience} époques")
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)


def evaluate_test(name: str, out_dir, device: str, batch: int, workers: int) -> None:
    """Évaluation finale, une seule fois, sur le test, avec le meilleur modèle de validation."""
    model, _ = load_classifier(out_dir / "best.pt", device)
    test_loader = make_loader(CropFolder(CROPS_DIR / "test", eval_transform()), batch, workers)
    _, y_test, p_test = run_epoch(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    report = classification_report(y_test, p_test, labels=range(len(CLASSES)), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.rename(columns={"precision": "precision", "recall": "rappel",
                                                       "f1-score": "f1", "support": "vignettes"})
    per_class.to_csv(REPORTS_DIR / f"cls_{name}_test.csv", float_format="%.4f")
    plot_confusion_matrix(confusion_matrix(y_test, p_test, labels=range(len(CLASSES))), CLASSES,
                          f"Classifieur {name} : matrice de confusion (test)",
                          FIGURES_DIR / f"cls_{name}_confusion_matrix.png")
    print(per_class.round(3).to_string())
    print(f"\nTEST classifieur {name} : accuracy {accuracy_score(y_test, p_test):.3f} | "
          f"F1 macro {f1_score(y_test, p_test, average='macro'):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom de l'expérience (dossier runs/cls_<name>)")
    parser.add_argument("--model", default="resnet50", help="architecture timm pré-entraînée sur ImageNet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true", help="évaluer runs/cls_<name>/best.pt sans ré-entraîner")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "runs" / f"cls_{args.name}"
    if not args.eval_only:
        train(args, device, out_dir)
    evaluate_test(args.name, out_dir, device, args.batch, args.workers)


if __name__ == "__main__":
    main()
