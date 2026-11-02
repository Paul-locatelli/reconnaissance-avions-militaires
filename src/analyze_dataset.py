"""Analyse exploratoire du dataset : contrôles de cohérence, statistiques et figures.

Usage : python -m src.analyze_dataset
Sorties : reports/class_stats.csv et reports/figures/*.png
"""
import random
import statistics as st
from collections import Counter

import cv2
import pandas as pd
from PIL import Image

from src.config import CLASSES, FIGURES_DIR, REPORTS_DIR, SEED
from src.crops import square_crop
from src.images import read_image
from src.plotting import BLUE, BLUE_BGR, INK_2, SURFACE, plt, setup_style
from src.voc import Annotation, load_annotations


def check_consistency(annotations: dict[str, Annotation]) -> list[str]:
    """Vérifie que chaque XML a son image, la bonne taille et des boîtes valides."""
    problems = []
    for ann in annotations.values():
        if not ann.image_path.exists():
            problems.append(f"{ann.image_id}: image manquante")
            continue
        with Image.open(ann.image_path) as img:
            if img.size != (ann.width, ann.height):
                problems.append(f"{ann.image_id}: taille XML {ann.width}x{ann.height} != image {img.size[0]}x{img.size[1]}")
        for b in ann.boxes:
            if b.name not in CLASSES:
                problems.append(f"{ann.image_id}: classe inconnue {b.name!r}")
            if not (0 <= b.xmin < b.xmax <= ann.width and 0 <= b.ymin < b.ymax <= ann.height):
                problems.append(f"{ann.image_id}: boîte hors image {b.as_xyxy()}")
    return problems


def class_stats(annotations: dict[str, Annotation]) -> pd.DataFrame:
    rows = []
    for name in CLASSES:
        boxes = [b for a in annotations.values() for b in a.boxes if b.name == name]
        rows.append({
            "classe": name,
            "avions": len(boxes),
            "images": sum(any(b.name == name for b in a.boxes) for a in annotations.values()),
            "largeur_mediane_px": st.median(b.width for b in boxes),
            "hauteur_mediane_px": st.median(b.height for b in boxes),
            "tronques": sum(b.truncated for b in boxes),
        })
    return pd.DataFrame(rows)


def label_extremes(ax, values: list[int]) -> None:
    """Étiquette seulement le minimum et le maximum : le message est le déséquilibre."""
    for i in (values.index(min(values)), values.index(max(values))):
        ax.text(values[i], i, f" {values[i]}", va="center", color=INK_2, fontsize=9)


def plot_class_distribution(stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, column, title in ((axes[0], "avions", "Avions par classe"),
                              (axes[1], "images", "Images contenant la classe")):
        values = stats[column].tolist()
        ax.barh(stats["classe"], values, height=0.75, color=BLUE)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        label_extremes(ax, values)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sizes_and_density(annotations: dict[str, Annotation]) -> None:
    sides = [max(b.width, b.height) for a in annotations.values() for b in a.boxes]
    per_image = [len(a.boxes) for a in annotations.values()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(sides, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    median = st.median(sides)
    axes[0].axvline(median, color=INK_2, linewidth=1, linestyle="--")
    axes[0].text(median, axes[0].get_ylim()[1] * 0.92, f"  médiane {median:.0f} px", color=INK_2, fontsize=9)
    axes[0].set_title("Taille des avions (plus grand côté)")
    axes[0].set_xlabel("pixels")

    axes[1].hist(per_image, bins=range(1, max(per_image) + 2), color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[1].set_title("Nombre d'avions par image")
    axes[1].set_xlabel("avions")
    axes[1].set_ylabel("images")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sizes_and_density.png", dpi=150)
    plt.close(fig)


def plot_samples_per_class(annotations: dict[str, Annotation]) -> None:
    """Une vignette par classe, prise parmi les avions non tronqués de taille médiane."""
    rng = random.Random(SEED)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
    for ax, name in zip(axes.flat, CLASSES):
        candidates = [(a, b) for a in annotations.values() for b in a.boxes if b.name == name and not b.truncated]
        candidates.sort(key=lambda ab: max(ab[1].width, ab[1].height))
        middle = candidates[len(candidates) // 4: 3 * len(candidates) // 4] or candidates
        ann, box = rng.choice(middle)
        crop = cv2.resize(square_crop(read_image(ann.image_path), box.as_xyxy()), (160, 160))
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=11, loc="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "samples_per_class.png", dpi=150)
    plt.close(fig)


def plot_annotated_examples(annotations: dict[str, Annotation]) -> None:
    anns = list(annotations.values())
    densest = max(anns, key=lambda a: len(a.boxes))
    most_mixed = max(anns, key=lambda a: len({b.name for b in a.boxes}))
    others = random.Random(SEED).sample([a for a in anns if a not in (densest, most_mixed)], 2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, ann in zip(axes.flat, [densest, most_mixed, *others]):
        image = read_image(ann.image_path)
        for b in ann.boxes:
            cv2.rectangle(image, (b.xmin, b.ymin), (b.xmax, b.ymax), BLUE_BGR, 2)
            cv2.putText(image, b.name, (b.xmin, max(b.ymin - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_BGR, 2)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {ann.image_id} : {len(ann.boxes)} avions", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "annotated_examples.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()

    problems = check_consistency(annotations)
    print(f"Contrôle de cohérence : {len(problems)} problème(s)")
    for p in problems[:20]:
        print("  -", p)

    boxes = [b for a in annotations.values() for b in a.boxes]
    per_image = [len(a.boxes) for a in annotations.values()]
    sizes = Counter((a.width, a.height) for a in annotations.values())
    classes_per_image = Counter(len({b.name for b in a.boxes}) for a in annotations.values())
    print(f"Images : {len(annotations)} | avions : {len(boxes)} | tronqués : {sum(b.truncated for b in boxes)}")
    print(f"Avions par image : min {min(per_image)}, médiane {st.median(per_image)}, "
          f"moyenne {st.mean(per_image):.1f}, max {max(per_image)}")
    print(f"Images 800x800 : {sizes[(800, 800)]} / {len(annotations)}")
    print(f"Classes distinctes par image : {dict(sorted(classes_per_image.items()))}")

    stats = class_stats(annotations)
    stats.to_csv(REPORTS_DIR / "class_stats.csv", index=False)
    print(stats.to_string(index=False))

    setup_style()
    plot_class_distribution(stats)
    plot_sizes_and_density(annotations)
    plot_samples_per_class(annotations)
    plot_annotated_examples(annotations)
    print(f"Figures écrites dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()
