"""Capture card / video kaynağından arka planda kare okuma.

Windows'ta DirectShow (yoksa Media Foundation), Linux'ta V4L2, macOS'ta
AVFoundation kullanılır. Okuma ayrı bir thread'de yapılır; son kare
kilitle korunur. Böylece tarayıcı akışı ve "dondur" komutu kameranın
okuma hızını beklemez.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
SYSTEM = platform.system()


def _api_preference() -> list[int]:
    if SYSTEM == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if SYSTEM == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if SYSTEM == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _api_name(api: int) -> str:
    try:
        return cv2.videoio_registry.getBackendName(api)
    except Exception:
        return str(api)


def open_device(index: int, width: int, height: int, fps: int,
                fourcc: str = "") -> tuple[cv2.VideoCapture, str]:
    last_err = None
    for api in _api_preference():
        cap = cv2.VideoCapture(int(index), api)
        if not cap.isOpened():
            cap.release()
            last_err = f"{_api_name(api)} ile açılamadı"
            continue
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            cap.set(cv2.CAP_PROP_FPS, int(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            return cap, _api_name(api)
        cap.release()
        last_err = f"{_api_name(api)} ile kare okunamadı"
    raise RuntimeError(f"Cihaz {index} açılamadı ({last_err}).")


def probe_devices(max_index: int = 6) -> list[dict]:
    found = []
    for i in range(max_index):
        for api in _api_preference()[:1]:
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "backend": _api_name(api),
                                  "width": frame.shape[1],
                                  "height": frame.shape[0]})
            cap.release()
    return found


class DemoSource:
    """Donanım olmadan denemek için sentetik tiroid USG benzeri görüntü.
    Ekranda üst bilgi şeridi ve 1 cm aralıklı derinlik skalası bulunur
    (kalibrasyon pratiği için: 1 cm = 40 piksel)."""

    PX_PER_CM = 40

    def __init__(self, width=1280, height=720):
        self.w, self.h = width, height
        rng = np.random.default_rng(7)
        self.speckle = rng.rayleigh(1.0, (height * 2, width * 2)).astype(np.float32)
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        w, h = self.w, self.h
        img = np.zeros((h, w), np.float32)
        x0, x1, y0, y1 = int(w * .18), int(w * .82), int(h * .12), int(h * .95)
        ox = int(20 + 10 * np.sin(t * 0.7))
        oy = int(20 + 6 * np.cos(t * 0.5))
        sp = self.speckle[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
        field = 95 * sp
        # Ciltaltı katman ve tiroid parankimi
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
        field *= np.where(yy < 50, 0.6, 1.0)
        # Hipoekoik solid nodül (hafif nefes hareketi)
        cx = (x1 - x0) * 0.48 + 6 * np.sin(t * 1.3)
        cy = (y1 - y0) * 0.45
        a, b = 92, 70  # yarı eksenler (px): çap 184×140 px ≈ 4,6×3,5 cm
        nod = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
        field[nod] *= 0.38
        # Küçük anekoik kist
        kist = ((xx - (x1 - x0) * 0.78) ** 2 + (yy - (y1 - y0) * 0.30) ** 2) <= 22 ** 2
        field[kist] *= 0.05
        # Derinlik zayıflaması
        field *= np.linspace(1.15, 0.7, y1 - y0)[:, None]
        img[y0:y1, x0:x1] = field
        img = np.clip(img, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Üst bilgi şeridi (gizlilik maskesi denemesi için sahte hasta bilgisi)
        cv2.rectangle(frame, (0, 0), (w, int(h * .08)), (40, 40, 40), -1)
        cv2.putText(frame, "HASTA: DEMO DEMO  ID: 0000000  "
                    + time.strftime("%d.%m.%Y %H:%M:%S"), (20, int(h * .055)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(frame, "L12 TIROID  DEMO", (w - 330, int(h * .055)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 255), 2)
        # Derinlik skalası: 1 cm aralık
        sx = x1 + 18
        for i in range(0, int((y1 - y0) / self.PX_PER_CM) + 1):
            y = y0 + i * self.PX_PER_CM
            cv2.line(frame, (sx, y), (sx + (14 if i % 5 == 0 else 8), y),
                     (220, 220, 220), 2)
            if i % 2 == 0:
                cv2.putText(frame, str(i), (sx + 20, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        time.sleep(1 / 30)
        return True, frame

    def release(self):
        pass


class CaptureManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._src = None
        self.cfg: dict = {}
        self.status = "kapalı"
        self.error: str | None = None
        self.backend_name = ""
        self.fps = 0.0

    # -------------------------------------------------------------- yönetim
    def start(self, cfg: dict):
        self.stop()
        self.cfg = dict(cfg)
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="capture")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._release()
        self.status = "kapalı"

    def _release(self):
        if self._src is not None:
            try:
                self._src.release()
            except Exception:
                pass
        self._src = None

    def _open(self):
        c = self.cfg
        kind = c.get("type", "demo")
        if kind == "demo":
            self.backend_name = "demo"
            return DemoSource()
        if kind == "file":
            path = c.get("path") or ""
            if not Path(path).exists():
                raise RuntimeError(f"Dosya bulunamadı: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Video açılamadı: {path}")
            self.backend_name = "dosya"
            return cap
        cap, name = open_device(int(c.get("index", 0)), c.get("width", 1920),
                                c.get("height", 1080), c.get("fps", 30),
                                c.get("fourcc", ""))
        self.backend_name = name
        return cap

    def _run(self):
        backoff = 1.0
        n, t_last = 0, time.time()
        while not self._stop.is_set():
            if self._src is None:
                try:
                    self.status = "bağlanıyor"
                    self._src = self._open()
                    self.status = "açık"
                    self.error = None
                    backoff = 1.0
                except Exception as e:
                    self.status = "hata"
                    self.error = str(e)
                    log.warning("Kaynak açılamadı: %s", e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 10)
                    continue
            ok, frame = self._src.read()
            if not ok or frame is None:
                if self.cfg.get("type") == "file":
                    self._src.set(cv2.CAP_PROP_POS_FRAMES, 0)  # döngü
                    time.sleep(0.03)
                    continue
                self.status = "hata"
                self.error = "Kare okunamadı; yeniden bağlanılıyor."
                self._release()
                self._stop.wait(1.0)
                continue
            if self.cfg.get("type") == "file":
                time.sleep(1 / 30)
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            n += 1
            now = time.time()
            if now - t_last >= 1.0:
                self.fps = n / (now - t_last)
                n, t_last = 0, now
        self._release()

    # ---------------------------------------------------------------- erişim
    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def info(self) -> dict:
        fid, f = self.latest()
        return {"status": self.status, "error": self.error,
                "backend": self.backend_name, "fps": round(self.fps, 1),
                "source": self.cfg, "frame_id": fid,
                "resolution": None if f is None else [f.shape[1], f.shape[0]]}
