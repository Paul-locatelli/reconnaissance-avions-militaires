"""Style commun des figures du rapport (palette de référence, mode clair)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
BLUE_BGR = (214, 120, 42)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Rampe séquentielle à une seule teinte (bleu clair -> bleu foncé) pour les heatmaps
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    })


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, path) -> None:
    """Matrice normalisée par ligne : chaque ligne (vraie classe) somme à 1."""
    setup_style()
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("vraie classe")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
