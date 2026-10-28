"""Boyun USG anatomi sınıfları (tek tanım yeri).

Etiket haritasında (labels.png) piksel değeri sınıf kimliğidir; 0 arka plan.
Nodül en son çizilir (tiroidin içinde yer aldığı için üstte kalır).
Eğitimde çok sınıflı model bu sırayla kanal üretir: kanal i = sınıf i.
"""

from __future__ import annotations

import cv2
import numpy as np

CLASSES = [
    # id, anahtar, Türkçe ad, renk (hex)
    (1, "nodule", "Nodül", "#ffc83d"),
    (2, "thyroid", "Tiroid", "#3fb950"),
    (3, "carotid", "Karotis", "#f85149"),
    (4, "ijv", "Juguler ven", "#6e8cff"),
    (5, "trachea", "Trakea", "#4cc3f0"),
]
NUM_CLASSES = len(CLASSES) + 1          # arka plan dahil
BY_ID = {c[0]: c for c in CLASSES}
DRAW_ORDER = [2, 3, 4, 5, 1]            # nodül en üstte


def public() -> list[dict]:
    return [{"id": i, "key": k, "name": n, "color": c} for i, k, n, c in CLASSES]


def label_map(shape, masks: dict) -> np.ndarray:
    """masks: {sınıf_id: 0/1 maske} -> uint8 etiket haritası."""
    lab = np.zeros(shape[:2], np.uint8)
    for cid in DRAW_ORDER:
        m = masks.get(cid)
        if m is not None:
            lab[m > 0] = cid
    return lab


def contours_from_labels(lab: np.ndarray, min_area: int = 60, eps: float = 1.5,
                         offset=(0, 0), scale: float = 1.0, max_parts: int = 3) -> dict:
    """Etiket haritasından sınıf başına sadeleştirilmiş konturlar (tam kare px)."""
    out = {}
    for cid, *_ in CLASSES:
        m = (lab == cid).astype(np.uint8)
        if not m.any():
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = sorted([c for c in cs if cv2.contourArea(c) >= min_area],
                    key=cv2.contourArea, reverse=True)[:max_parts]
        polys = []
        for c in cs:
            c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
            c = c / scale + np.array(offset, np.float32)
            polys.append(np.round(c).astype(int).tolist())
        if polys:
            out[str(cid)] = polys
    return out
