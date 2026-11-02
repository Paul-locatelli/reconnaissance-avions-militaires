"""Métriques de détection (AP, mAP, précision, rappel, F1) codées à la main.

Les deux approches sont évaluées avec exactement ce même code : la comparaison est
équitable, et on maîtrise chaque étape du calcul.

Principe, pour une classe et un seuil d'IoU :
1. toutes les détections de la classe sont triées par score décroissant ;
2. chacune est appariée à la vraie boîte de même classe, non encore prise, qui la
   recouvre le plus (si IoU >= seuil) : vrai positif ; sinon : faux positif ;
3. en descendant la liste, on obtient une courbe précision / rappel ;
4. AP = aire sous cette courbe (interpolation sur 101 points de rappel, comme COCO).
"""
import numpy as np
import pandas as pd

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)  # 0.50, 0.55, ..., 0.95


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre chaque boîte de a (N, 4) et chaque boîte de b (M, 4), format xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(preds: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray], iou_thr: float) -> np.ndarray:
    """preds = (image_id, score, boîte) déjà triées par score ; renvoie un booléen 'vrai positif' par prédiction."""
    taken = {image_id: np.zeros(len(boxes), bool) for image_id, boxes in gts.items()}
    is_tp = np.zeros(len(preds), bool)
    for k, (image_id, _, box) in enumerate(preds):
        gt_boxes = gts.get(image_id)
        if gt_boxes is None or len(gt_boxes) == 0:
            continue
        ious = box_iou(box[None], gt_boxes)[0]
        ious[taken[image_id]] = -1
        best = int(ious.argmax())
        if ious[best] >= iou_thr:
            taken[image_id][best] = True
            is_tp[k] = True
    return is_tp


def average_precision(is_tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or len(is_tp) == 0:
        return 0.0
    tp = np.cumsum(is_tp)
    recall = tp / n_gt
    precision = tp / np.arange(1, len(is_tp) + 1)
    # Enveloppe : la précision à un rappel r est la meilleure précision atteinte pour un rappel >= r
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def evaluate(predictions: dict, annotations: dict, classes: list[str], conf_threshold: float) -> pd.DataFrame:
    """predictions : image_id -> liste de Detection ; annotations : image_id -> Annotation.

    AP50 et AP50-95 utilisent toutes les prédictions (indépendants du seuil) ;
    précision, rappel et F1 sont calculés à IoU 0.5 en ne gardant que score >= conf_threshold.
    """
    rows, total = [], {"tp": 0, "fp": 0, "fn": 0}
    for name in classes:
        gts = {i: np.array([b.as_xyxy() for b in a.boxes if b.name == name], float).reshape(-1, 4)
               for i, a in annotations.items()}
        n_gt = sum(len(g) for g in gts.values())
        preds = sorted(((i, d.score, np.array(d.as_xyxy(), float)) for i, dets in predictions.items()
                        for d in dets if d.classe == name), key=lambda p: -p[1])
        aps = [average_precision(match(preds, gts, t), n_gt) for t in IOU_THRESHOLDS]

        kept = [p for p in preds if p[1] >= conf_threshold]
        tp = int(match(kept, gts, 0.5).sum())
        fp, fn = len(kept) - tp, n_gt - tp
        total["tp"] += tp; total["fp"] += fp; total["fn"] += fn
        p, r = tp / max(tp + fp, 1), tp / max(n_gt, 1)
        rows.append({"classe": name, "avions": n_gt, "precision": p, "rappel": r,
                     "f1": 2 * p * r / max(p + r, 1e-9), "AP50": aps[0], "AP50_95": float(np.mean(aps))})

    results = pd.DataFrame(rows)
    p = total["tp"] / max(total["tp"] + total["fp"], 1)
    r = total["tp"] / max(total["tp"] + total["fn"], 1)
    results.loc[len(results)] = {"classe": "global", "avions": int(results.avions.sum()), "precision": p, "rappel": r,
                                 "f1": 2 * p * r / max(p + r, 1e-9), "AP50": results.AP50.mean(),
                                 "AP50_95": results.AP50_95.mean()}
    return results


CORRECT, WRONG_CLASS, FALSE_POSITIVE = "correct", "mauvaise classe", "fausse détection"


def compare_to_ground_truth(detections: list, boxes: list, iou_thr: float = 0.5) -> tuple[list, list]:
    """Pour expliquer les erreurs d'une image : appariement par IoU sans regarder la classe.

    Renvoie (verdict, vraie classe) pour chaque détection, dans l'ordre reçu, et les vraies
    boîtes qu'aucune détection ne recouvre (avions manqués).
    """
    gt = np.array([b.as_xyxy() for b in boxes], float).reshape(-1, 4)
    taken = np.zeros(len(gt), bool)
    results = [(FALSE_POSITIVE, None)] * len(detections)
    for k in sorted(range(len(detections)), key=lambda k: -detections[k].score):
        if len(gt) == 0:
            break
        ious = box_iou(np.array([detections[k].as_xyxy()], float), gt)[0]
        ious[taken] = -1
        best = int(ious.argmax())
        if ious[best] >= iou_thr:
            taken[best] = True
            truth = boxes[best].name
            results[k] = (CORRECT if truth == detections[k].classe else WRONG_CLASS, truth)
    return results, [b for b, t in zip(boxes, taken) if not t]


def best_conf_threshold(predictions: dict, annotations: dict, classes: list[str]) -> float:
    """Seuil de confiance qui maximise le F1 global : à choisir sur la VALIDATION, puis appliquer au test."""
    candidates = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = [evaluate(predictions, annotations, classes, t).iloc[-1]["f1"] for t in candidates]
    return float(candidates[int(np.argmax(f1s))])
