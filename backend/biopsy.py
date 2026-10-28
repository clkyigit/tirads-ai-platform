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


class Session:
    """Bir biyopsi işleminin ölçümleri (eğitim ve belgeleme için).

    - süre, iğnenin görüldüğü kare oranı ("uç görüntüde %")
    - ilk algılama ve hedefe ilk ulaşma zamanı
    - en yakın uç–hedef mesafesi, rotada geçen süre oranı
    - en büyük ve ortalama tahmini sapma
    - uç yolu (seyreltilmiş)
    """

    def __init__(self, sid: str, plan: dict):
        self.id = sid
        self.t0 = time.time()
        self.plan0 = json.loads(json.dumps(plan))
        self.frames = 0
        self.found = 0
        self.on_track = 0
        self.first_found = None
        self.first_target = None
        self.min_dist = None
        self.misses: list[float] = []
        self.path: list[list[float]] = []
        self._last_path_t = 0.0
        self.unit = "mm"
        self.ended = None

    def update(self, st: dict):
        now = time.time()
        self.frames += 1
        g = st.get("guide") or {}
        self.unit = g.get("unit", self.unit)
        nd = st.get("needle")
        if nd and nd.get("found"):
            self.found += 1
            if self.first_found is None:
                self.first_found = now - self.t0
            if now - self._last_path_t > 0.25:
                self.path.append([round(now - self.t0, 2), *[round(v, 1) for v in nd["tip"]]])
                self._last_path_t = now
                self.path = self.path[-600:]
        status = g.get("status")
        if status in ("on_track", "in_target"):
            self.on_track += 1
        if status == "in_target" and self.first_target is None:
            self.first_target = now - self.t0
        d = g.get("tip_to_target")
        if d is not None:
            self.min_dist = d if self.min_dist is None else min(self.min_dist, d)
        if g.get("miss") is not None:
            self.misses.append(float(g["miss"]))
            self.misses = self.misses[-5000:]

    def summary(self) -> dict:
        end = self.ended or time.time()
        f = max(1, self.frames)
        return {
            "id": self.id, "active": self.ended is None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
            "duration_s": round(end - self.t0, 1),
            "frames": self.frames,
            "tip_in_view_pct": round(100 * self.found / f, 1),
            "on_track_pct": round(100 * self.on_track / f, 1),
            "first_detection_s": None if self.first_found is None else round(self.first_found, 1),
            "time_to_target_s": None if self.first_target is None else round(self.first_target, 1),
            "min_tip_to_target": None if self.min_dist is None else round(self.min_dist, 1),
            "max_miss": round(max(self.misses), 1) if self.misses else None,
            "mean_miss": round(float(np.mean(self.misses)), 1) if self.misses else None,
            "unit": self.unit,
        }


class BiopsyTracker:
    def __init__(self, capture, get_cfg, segmenter_getter=None):
        self.capture = capture
        self.get_cfg = get_cfg            # -> config dict (roi, calibration)
        self.get_segmenter = segmenter_getter
        self.session: Session | None = None
        self.anatomy_on = False
        self.anatomy: dict = {}
        self._anat_thread = None
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
        if self.anatomy_on:
            self.set_anatomy(True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        if self._anat_thread:
            self._anat_thread.join(timeout=3)
        self._anat_thread = None
        if self.session and self.session.ended is None:
            self.session.ended = time.time()
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

    # --------------------------------------------------------- oturum
    def start_session(self) -> dict:
        with self.lock:
            sid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            self.session = Session(sid, self.plan)
            return self.session.summary()

    def stop_session(self) -> dict | None:
        with self.lock:
            if not self.session:
                return None
            if self.session.ended is None:
                self.session.ended = time.time()
            return self.session.summary()

    # ----------------------------------------------------- canlı anatomi
    def set_anatomy(self, on: bool):
        self.anatomy_on = bool(on)
        if not on:
            with self.lock:
                self.anatomy = {}
        elif self.active and (self._anat_thread is None or not self._anat_thread.is_alive()):
            self._anat_thread = threading.Thread(target=self._anat_run, daemon=True,
                                                 name="anatomy")
            self._anat_thread.start()

    def _anat_run(self):
        """Segmentasyonu kendi hızında çalıştırır (iğne takibini bekletmez)."""
        from . import anatomy as A
        from .segmentation import roi_to_px
        last = -1
        while not self._stop.is_set() and self.anatomy_on:
            seg = self.get_segmenter() if self.get_segmenter else None
            fid, frame = self.capture.latest()
            if seg is None or not seg.ready or frame is None or fid == last:
                time.sleep(0.05)
                continue
            last = fid
            t0 = time.perf_counter()
            try:
                H, W = frame.shape[:2]
                x0, y0, x1, y1 = roi_to_px(self.get_cfg().get("roi"), W, H)
                crop = frame[y0:y1, x0:x1]
                work = 384
                lab = seg.predict_labels(crop, work_width=work)
                sc = lab.shape[1] / max(1, crop.shape[1])
                polys = A.contours_from_labels(lab, min_area=int(40 * sc * sc) or 20,
                                               offset=(x0, y0), scale=sc)
                ms = (time.perf_counter() - t0) * 1000
                with self.lock:
                    self.anatomy = {"polys": polys, "frame_id": fid, "ms": round(ms),
                                    "multiclass": seg.multiclass, "t": time.time()}
            except Exception as e:
                log.exception("Canlı anatomi hatası")
                with self.lock:
                    self.anatomy = {"error": str(e)}
                time.sleep(0.5)
            time.sleep(0.02)

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
                if self.session and self.session.ended is None:
                    self.session.update(st)
                st["session"] = self.session.summary() if self.session else None
                st["anatomy_on"] = self.anatomy_on
                st["anatomy"] = self.anatomy if self.anatomy_on else None
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


def _hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def render_overlay(frame: np.ndarray, st: dict, summary: dict | None = None) -> np.ndarray:
    img = frame.copy()
    anat = (st.get("anatomy") or {}).get("polys") or {}
    if anat:
        from .anatomy import BY_ID
        tint = img.copy()
        for cid, polys in anat.items():
            c = BY_ID.get(int(cid))
            if not c:
                continue
            col_ = _hex_bgr(c[3])
            pts = [np.asarray(p_, np.int32).reshape(-1, 1, 2) for p_ in polys]
            cv2.fillPoly(tint, pts, col_)
            cv2.polylines(img, pts, True, col_, 1, cv2.LINE_AA)
        img = cv2.addWeighted(tint, 0.22, img, 0.78, 0)
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
    if summary:
        lines.append(f"Sure: {summary['duration_s']:.0f} sn  Uc gorunur: %{summary['tip_in_view_pct']:.0f}")
        if summary.get("time_to_target_s") is not None:
            lines.append(f"Hedefe ulasma: {summary['time_to_target_s']:.1f} sn")
        if summary.get("min_tip_to_target") is not None:
            lines.append(f"En yakin: {summary['min_tip_to_target']:.1f} {summary['unit']}")
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
