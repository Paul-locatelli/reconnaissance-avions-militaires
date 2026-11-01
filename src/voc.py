"""Lecture des annotations au format Pascal VOC (un fichier XML par image)."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from src.config import RAW_DIR


@dataclass
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    truncated: bool = False

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax


@dataclass
class Annotation:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box]


def parse_voc(xml_path: Path, images_dir: Path) -> Annotation:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append(Box(obj.findtext("name").strip(), xmin, ymin, xmax, ymax,
                         truncated=obj.findtext("truncated", "0").strip() == "1"))
    return Annotation(
        image_id=xml_path.stem,
        # Le champ <filename> n'est pas fiable (145 XML indiquent "104.xml" au lieu
        # de "104.jpg") : on déduit le nom de l'image du nom du fichier XML.
        image_path=images_dir / f"{xml_path.stem}.jpg",
        width=int(size.findtext("width")),
        height=int(size.findtext("height")),
        boxes=boxes,
    )


def load_annotations(raw_dir: Path = RAW_DIR) -> dict[str, Annotation]:
    """Charge toutes les annotations, indexées par identifiant d'image (ex. "1008")."""
    images_dir = raw_dir / "images"
    xml_files = sorted((raw_dir / "annotations").glob("*.xml"), key=lambda p: int(p.stem))
    return {p.stem: parse_voc(p, images_dir) for p in xml_files}


def read_split(split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
