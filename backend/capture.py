"""Capture card / video kaynağından arka planda kare okuma.

Windows'ta DirectShow (yoksa Media Foundation), Linux'ta V4L2, macOS'ta
AVFoundation kullanılır. Okuma ayrı bir thread'de yapılır; son kare
kilitle korunur. Böylece tarayıcı akışı ve "dondur" komutu kameranın
okuma hızını beklemez.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import devices

log = logging.getLogger(__name__)


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


class NoDevice(RuntimeError):
    """Capture card takılı değil (hata değil; beklenir)."""


class CaptureManager:
    """Arka planda kare okur. type="auto" iken capture card'ı kendisi bulur,
    çıkarılınca bekler, takılınca yeniden bağlanır."""

    SCAN_INTERVAL = 1.5   # cihaz beklerken tarama aralığı (sn)
    READ_FAILS_TO_DROP = 8

    def __init__(self, lister=None, opener=None, on_negotiated=None):
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
        self.device: dict | None = None     # seçilen cihaz + format bilgisi
        self.signal = True
        self.log: list[str] = []
        self.connected_at = 0.0
        self._lister = lister or devices.list_devices
        self._opener = opener or cv2.VideoCapture
        self._on_negotiated = on_negotiated

    # -------------------------------------------------------------- yönetim
    def start(self, cfg: dict):
        self.stop()
        self.cfg = dict(cfg)
        self.error = None
        self.device = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="capture")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
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
        with self._lock:
            self._frame = None

    def _note(self, msg: str):
        log.info(msg)
        self.log = (self.log + [time.strftime("%H:%M:%S ") + msg])[-30:]

    def _open(self):
        c = self.cfg
        kind = c.get("type", "auto")
        if kind == "demo":
            self.backend_name = "demo"
            self.device = {"name": "Demo (sentetik)", "fourcc": "-",
                           "compressed": False}
            return DemoSource()
        if kind == "file":
            path = c.get("path") or ""
            if not Path(path).exists():
                raise RuntimeError(f"Dosya bulunamadı: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Video açılamadı: {path}")
            self.backend_name = "dosya"
            self.device = {"name": Path(path).name, "fourcc": "-",
                           "compressed": True}
            return cap
        if kind == "device":  # elle seçilmiş indeks
            dev = {"index": int(c.get("index", 0)),
                   "name": c.get("device_name") or f"Cihaz #{c.get('index', 0)}"}
        else:  # auto
            devs = self._lister()
            dev = devices.choose_device(devs, c.get("device_name", ""),
                                        c.get("ignore") or [])
            if dev is None:
                webcams = [d["name"] for d in devs if d.get("score", 0) <= -5]
                raise NoDevice("Capture card bulunamadı." + (
                    f" (Yok sayılan kamera: {', '.join(webcams)})" if webcams else ""))
        cached = (c.get("negotiated") or {}).get(dev["name"])
        self._note(f"Bağlanılıyor: {dev['name']} (#{dev['index']})")
        cap, info = devices.negotiate(dev["index"], c, opener=self._opener,
                                      cached=cached, log_fn=self._note)
        self.backend_name = info["backend"]
        self.device = {**dev, **info}
        self._note(f"Bağlandı: {dev['name']} · {info['width']}×{info['height']}"
                   f" · {info['fourcc']} · {info['fps']} fps")
        if self._on_negotiated and info.get("mode"):
            fc, w, h = info["mode"]
            try:
                self._on_negotiated(dev["name"], {"fourcc": fc, "width": w,
                                                  "height": h})
            except Exception:
                log.exception("Format bilgisi kaydedilemedi")
        return cap

    def _run(self):
        backoff = 1.0
        n, t_last, fails, k = 0, time.time(), 0, 0
        while not self._stop.is_set():
            if self._src is None:
                try:
                    self.status = "bağlanıyor" if self.status != "bekleniyor" \
                        else "bekleniyor"
                    self._src = self._open()
                    self.status = "açık"
                    self.error = None
                    self.signal = True
                    self.connected_at = time.time()
                    backoff, fails = 1.0, 0
                except NoDevice as e:
                    self.status = "bekleniyor"
                    self.error = str(e)
                    self.device = None
                    self._stop.wait(self.SCAN_INTERVAL)
                    continue
                except Exception as e:
                    self.status = "hata"
                    self.error = str(e)
                    self.device = None
                    log.warning("Kaynak açılamadı: %s", e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
            ok, frame = self._src.read()
            if not ok or frame is None:
                if self.cfg.get("type") == "file":
                    self._src.set(cv2.CAP_PROP_POS_FRAMES, 0)  # döngü
                    time.sleep(0.03)
                    continue
                fails += 1
                if fails >= self.READ_FAILS_TO_DROP:
                    # Kart çıkarıldı ya da sürücü takıldı: bırak, yeniden ara
                    self._note("Bağlantı koptu; capture card yeniden aranıyor.")
                    self.error = "Bağlantı koptu; yeniden aranıyor."
                    self.status = "bekleniyor"
                    self._release()
                    self.device = None
                    fails = 0
                    self._stop.wait(0.5)
                else:
                    time.sleep(0.02)
                continue
            fails = 0
            if self.cfg.get("type") == "file":
                time.sleep(1 / 30)
            k += 1
            if k % 10 == 1:
                sig = not devices.is_no_signal(frame)
                if sig != self.signal:
                    self._note("Görüntü sinyali geldi." if sig else
                               "Sinyal yok (tek renk görüntü).")
                self.signal = sig
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
        src = {k: v for k, v in self.cfg.items() if k != "negotiated"}
        return {"status": self.status, "error": self.error,
                "backend": self.backend_name, "fps": round(self.fps, 1),
                "source": src, "frame_id": fid, "device": self.device,
                "signal": self.signal, "connected_at": self.connected_at,
                "log": self.log[-8:],
                "resolution": None if f is None else [f.shape[1], f.shape[0]]}
