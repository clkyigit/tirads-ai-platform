"""Canlı biyopsi rehberi: capture karelerini arka planda işleyip iğne,
hedef ve rehber ölçümlerini yayınlar."""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import needle as N

log = logging.getLogger(__name__)


def _seg_dist(p, a, b) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    L2 = ax * ax + ay * ay
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L2))
    return math.dist(p, (a[0] + t * ax, a[1] + t * ay))


class BiopsyTracker:
    def __init__(self, capture, get_cfg):
        self.capture = capture
        self.get_cfg = get_cfg            # -> config dict (roi, calibration)
        self.lock = threading.Lock()
        self.plan = {"target": None, "entry": None, "radius_mm": 5.0,
                     "track_target": True}
        self.smoother = N.NeedleSmoother()
        self.target = N.TargetTracker()
        self.state: dict = {"active": False}
        self.version = 0
        self._stop = threading.Event()
        self._thread = None
        self._prev_small = None
        self._motion = None
        self._last_gray = None
        self._fps_n, self._fps_t, self.fps = 0, time.time(), 0.0

    # ----------------------------------------------------------- yönetim
    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.active:
            return
        self._stop.clear()
        self.smoother.reset()
        self._prev_small = self._motion = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="biopsy")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        with self.lock:
            self.state = {"active": False}
            self.version += 1

    def set_plan(self, target=None, entry=None, radius_mm=None, track_target=None,
                 clear=False):
        with self.lock:
            if clear:
                self.plan.update(target=None, entry=None)
                self.target.clear()
            if target is not None:
                self.plan["target"] = [float(target[0]), float(target[1])]
                if self._last_gray is not None:
                    self.target.set(self._last_gray, self.plan["target"])
            if entry is not None:
                self.plan["entry"] = [float(entry[0]), float(entry[1])] if entry else None
            if radius_mm is not None:
                self.plan["radius_mm"] = float(max(1.0, min(30.0, radius_mm)))
            if track_target is not None:
                self.plan["track_target"] = bool(track_target)
            self.smoother.reset()
            return dict(self.plan)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    # ------------------------------------------------------------ döngü
    def _roi_px(self, W, H):
        from .segmentation import roi_to_px
        return roi_to_px(self.get_cfg().get("roi"), W, H)

    def _run(self):
        last_fid = -1
        while not self._stop.is_set():
            fid, frame = self.capture.latest()
            if frame is None or fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid
            t0 = time.perf_counter()
            try:
                st = self._process(fid, frame)
            except Exception as e:  # algılama hatası takibi durdurmasın
                log.exception("Biyopsi takibi hatası")
                st = {"active": True, "error": str(e)}
            st["proc_ms"] = (time.perf_counter() - t0) * 1000
            self._fps_n += 1
            now = time.time()
            if now - self._fps_t >= 1:
                self.fps = self._fps_n / (now - self._fps_t)
                self._fps_n, self._fps_t = 0, now
            st["fps"] = round(self.fps, 1)
            with self.lock:
                self.state = st
                self.version += 1

    def _process(self, fid, frame):
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        cfg = self.get_cfg()
        roi = self._roi_px(W, H)
        mmpp = cfg.get("calibration", {}).get("mm_per_px")

        # Hareket haritası (küçültülmüş ROI): iğne ilerlerken öne çıkar
        x0, y0, x1, y1 = roi
        crop = gray[y0:y1, x0:x1]
        s = min(1.0, N.WORK_WIDTH / max(1, crop.shape[1]))
        small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) \
            if s < 1 else crop
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            diff = cv2.GaussianBlur(cv2.absdiff(small, self._prev_small).astype(np.float32),
                                    (0, 0), 0.8)
            # Azalan maksimum: ilerleyen ucun bıraktığı iz birkaç kare kalır
            self._motion = diff if self._motion is None or self._motion.shape != diff.shape \
                else np.maximum(0.85 * self._motion, diff)
        self._prev_small = small

        prev_needle = self.smoother.state
        with self.lock:
            self._last_gray = gray
            plan = dict(self.plan)
            if plan["target"] is not None and self.target.tpl is None:
                self.target.set(gray, plan["target"])
            tgt_frozen = False
            if plan["target"] is not None and plan["track_target"]:
                # İğne hedef bölgesine girince görüntü parçası bozulur (parlak
                # iğne şablonu yanıltır): hedef konumu o sırada sabit tutulur.
                if prev_needle is not None and _seg_dist(
                        plan["target"], prev_needle["entry"], prev_needle["tip"]) \
                        < self.target.patch / self.target.SCALE * 0.75:
                    tgt_frozen = True
                else:
                    pos = self.target.update(gray)
                    if pos is not None:
                        self.plan["target"] = pos
                        plan["target"] = pos
            tgt_lost = (self.target.lost and not tgt_frozen) if plan["target"] is not None else False
            tgt_score = self.target.score

        prev = self.smoother.state
        det = N.detect(gray, roi, plan, prev=prev, motion=self._motion)
        sm = self.smoother.update(det)
        g = N.guidance(sm, plan["target"], plan["entry"], plan["radius_mm"], mmpp,
                       skin_y=y0)
        return {
            "active": True, "frame_id": fid, "frame_size": [W, H], "t": time.time(),
            "roi_px": list(roi), "mm_per_px": mmpp,
            "plan": plan, "target_lost": tgt_lost, "target_score": tgt_score,
            "target_frozen": tgt_frozen,
            "needle": sm, "raw_found": det is not None, "guide": g,
        }


# --------------------------------------------------------- kayıt (çizim)
def _dashed(img, p, q, color, thick=2, dash=12):
    L = math.dist(p, q)
    if L < 1:
        return
    n = int(L // dash)
    for i in range(0, n, 2):
        a = (p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
        b = (p[0] + (q[0] - p[0]) * min(n, i + 1) / n, p[1] + (q[1] - p[1]) * min(n, i + 1) / n)
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, thick, cv2.LINE_AA)


STATUS_ASCII = {"searching": "Igne araniyor", "tracking": "Igne izleniyor",
                "on_track": "Rotada", "off_track": "Sapma var",
                "in_target": "Uc hedefte", "passed": "Uc hedefi gecti"}
STATUS_COLOR = {"searching": (160, 160, 160), "tracking": (80, 220, 255),
                "on_track": (90, 220, 90), "off_track": (40, 170, 255),
                "in_target": (90, 255, 90), "passed": (60, 60, 255)}


def render_overlay(frame: np.ndarray, st: dict) -> np.ndarray:
    img = frame.copy()
    plan, g, nd = st.get("plan") or {}, st.get("guide") or {}, st.get("needle")
    tgt, ent = plan.get("target"), plan.get("entry")
    R = g.get("target_radius_px", 25)
    if tgt is not None and ent is not None:
        _dashed(img, ent, tgt, (240, 200, 70), 2)
    if tgt is not None:
        c = tuple(map(int, tgt))
        cv2.circle(img, c, int(R), (240, 200, 70), 2, cv2.LINE_AA)
        cv2.drawMarker(img, c, (240, 200, 70), cv2.MARKER_CROSS, 18, 2)
    status = g.get("status", "searching")
    col = STATUS_COLOR.get(status, (200, 200, 200))
    if nd and nd.get("found"):
        e, t = nd["entry"], nd["tip"]
        cv2.line(img, tuple(map(int, e)), tuple(map(int, t)), col, 3, cv2.LINE_AA)
        u = np.subtract(t, e)
        u = u / (np.linalg.norm(u) + 1e-9)
        far = (t[0] + u[0] * 2000, t[1] + u[1] * 2000)
        _dashed(img, t, far, col, 1, 10)
        cv2.circle(img, tuple(map(int, t)), 7, col, 2, cv2.LINE_AA)
    unit = g.get("unit", "px")
    lines = [STATUS_ASCII.get(status, status)]
    if g.get("tip_to_target") is not None:
        lines.append(f"Uc-hedef: {g['tip_to_target']:.1f} {unit}")
    if g.get("miss") is not None:
        lines.append(f"Tahmini sapma: {g['miss']:.1f} {unit}")
    if g.get("needle_angle_deg") is not None:
        lines.append(f"Aci: {g['needle_angle_deg']:.0f} derece")
    lines.append(time.strftime("%d.%m.%Y %H:%M:%S"))
    # Bilgi kutusu: sol alt köşe, koyu zemin (cihaz üst şeridiyle çakışmasın)
    lh, pad = 26, 10
    bw = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for ln in lines) + 2 * pad
    bh = lh * len(lines) + pad
    H = img.shape[0]
    x0, y0 = 10, H - bh - 10
    box = img.copy()
    cv2.rectangle(box, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
    img = cv2.addWeighted(box, 0.65, img, 0.35, 0)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x0 + pad, y0 + pad + 14 + i * lh), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, col if i == 0 else (235, 235, 235), 2, cv2.LINE_AA)
    return img


def save_snapshot(root: Path, frame: np.ndarray, st: dict, meta: dict) -> Path:
    d = root / (time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2))
    d.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(d / "frame.png"), frame)
    cv2.imwrite(str(d / "overlay.jpg"), render_overlay(frame, st),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    (d / "state.json").write_text(json.dumps({**meta, "state": st}, indent=2,
                                             ensure_ascii=False), encoding="utf-8")
    return d
