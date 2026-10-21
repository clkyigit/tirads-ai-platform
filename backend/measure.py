"""Maske/kontur ölçümleri ve kalibrasyon yardımcıları.

Ölçümler piksel cinsinden hesaplanır; mm'ye çevirmek için ekrandaki derinlik
skalasından alınan mm/piksel oranı kullanılır. Lineer probda tarama dönüşümü
izotropiktir, bu yüzden tek bir oran yeterlidir. Cihazda derinlik veya zoom
değiştiğinde oran değişir: yeniden kalibre edilmelidir.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def mm_per_px_from_points(p1, p2, distance_mm: float) -> float:
    d = math.dist(p1, p2)
    if d < 5:
        raise ValueError("Kalibrasyon noktaları birbirine çok yakın.")
    if distance_mm <= 0:
        raise ValueError("Mesafe (mm) pozitif olmalı.")
    return distance_mm / d


def clean_mask(prob_or_mask: np.ndarray, threshold: float = 0.5,
               click: tuple[int, int] | None = None,
               min_area_px: int = 30) -> np.ndarray:
    """Eşikle, küçük gürültüyü temizle, tek bileşen seç, boşlukları doldur.

    click verilirse o noktayı içeren (yoksa en yakın) bileşen seçilir;
    verilmezse en büyük bileşen.
    """
    m = (prob_or_mask >= threshold).astype(np.uint8) if \
        prob_or_mask.dtype != np.uint8 else (prob_or_mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not cand:
        return np.zeros_like(m)
    chosen = None
    if click is not None:
        x, y = int(click[0]), int(click[1])
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] \
                and labels[y, x] in cand:
            chosen = labels[y, x]
        else:
            # Tıklanan noktaya en yakın bileşen (sınır mesafesi ile)
            best = None
            for i in cand:
                comp = (labels == i).astype(np.uint8)
                cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                d = -cv2.pointPolygonTest(cs[0], (float(x), float(y)), True)
                if best is None or d < best[0]:
                    best = (d, i)
            chosen = best[1]
    else:
        chosen = max(cand, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    out = (labels == chosen).astype(np.uint8)
    cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cs, -1, 1, thickness=cv2.FILLED)
    return filled


def mask_to_contour(mask: np.ndarray, epsilon_px: float = 1.0) -> list:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon_px, True)
    return [[int(p[0][0]), int(p[0][1])] for p in c]


def contour_metrics(contour) -> dict:
    """Kontur üzerinden piksel cinsinden ölçümler.

    - width: yatay (transvers) en geniş çap
    - height: dikey (AP, ses demeti yönü) en büyük çap
    - feret: en uzun çap (herhangi bir yönde)
    Her biri için çizim uçları da döner.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("Kontur için en az 3 nokta gerekir.")
    xs, ys = pts[:, 0], pts[:, 1]

    def extent_along_axis(axis: int):
        # En geniş yatay/dikey kesit: konturu tarama çizgileriyle kesip
        # en uzun iç parçayı bulur (AP ve transvers çap pratiği gibi).
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        x0, y0, w, h = cv2.boundingRect(poly)
        canvas = np.zeros((h + 3, w + 3), np.uint8)
        cv2.fillPoly(canvas, [poly - [x0 - 1, y0 - 1]], 1)
        best = (0, None)
        if axis == 0:  # yatay çizgiler -> width
            for r in range(canvas.shape[0]):
                cols = np.flatnonzero(canvas[r])
                if cols.size:
                    length = cols[-1] - cols[0] + 1
                    if length > best[0]:
                        best = (length, ((cols[0] + x0 - 1, r + y0 - 1),
                                         (cols[-1] + x0 - 1, r + y0 - 1)))
        else:  # dikey çizgiler -> height
            for c in range(canvas.shape[1]):
                rows = np.flatnonzero(canvas[:, c])
                if rows.size:
                    length = rows[-1] - rows[0] + 1
                    if length > best[0]:
                        best = (length, ((c + x0 - 1, rows[0] + y0 - 1),
                                         (c + x0 - 1, rows[-1] + y0 - 1)))
        (a, b) = best[1]
        return float(best[0]), [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]

    width, w_line = extent_along_axis(0)
    height, h_line = extent_along_axis(1)

    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    if len(hull) >= 2:
        d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        feret = float(d[i, j])
        f_line = [[int(hull[i][0]), int(hull[i][1])],
                  [int(hull[j][0]), int(hull[j][1])]]
    else:
        feret, f_line = 0.0, []

    area = float(cv2.contourArea(pts.astype(np.float32)))
    return {
        "width_px": width, "width_line": w_line,
        "height_px": height, "height_line": h_line,
        "feret_px": feret, "feret_line": f_line,
        "area_px": area,
        "bbox": [float(xs.min()), float(ys.min()),
                 float(xs.max() - xs.min()), float(ys.max() - ys.min())],
    }


def to_mm(metrics: dict, mm_per_px: float | None) -> dict:
    if not mm_per_px:
        return {}
    return {
        "width_mm": metrics["width_px"] * mm_per_px,
        "height_mm": metrics["height_px"] * mm_per_px,
        "feret_mm": metrics["feret_px"] * mm_per_px,
        "area_mm2": metrics["area_px"] * mm_per_px ** 2,
    }
