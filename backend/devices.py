"""Video cihazlarını bulma, capture card'ı seçme ve en iyi formatı ayarlama.

Amaç: capture card takılır takılmaz, ayar yapmadan doğru cihazdan en iyi
kalitede görüntü almak.

- Cihaz adları: Linux'ta /sys/class/video4linux, Windows'ta DirectShow
  (pygrabber). Ad alınamazsa indeksler denenir.
- Seçim: adı capture card'a benzeyen cihaz öne çıkar, dahili/USB webcam
  geri düşer. Kullanıcı bir cihazı "tercih edilen" ya da "yok say" olarak
  işaretleyebilir.
- Format: Önce sıkıştırmasız (YUY2/NV12) ve yeterli kare hızı (>=15 fps)
  denenir; olmazsa MJPG (sıkıştırmalı, uyarıyla). Bulunan ayar cihaz adına
  göre hatırlanır, sonraki takışta beklemeden açılır.
"""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)
SYSTEM = platform.system()

CAPTURE_HINTS = (
    "capture", "hdmi", "dvi", "vga", "sdi", "cam link", "camlink", "elgato",
    "magewell", "avermedia", "live gamer", "ugreen", "hagibis", "ms2109",
    "ms2130", "macrosilicon", "video grabber", "grabber", "epiphan",
    "blackmagic", "ezcap", "ripsaw", "acasis", "j5create", "startech",
    "usb video", "usb3.0 video", "usb 3.0 video", "usb3. 0", "hd video",
)
WEBCAM_HINTS = (
    "webcam", "web cam", "integrated", "facetime", "ir camera", "infrared",
    "front camera", "rear camera", "uvc camera", "virtual", "obs ",
    "obs-", "droidcam", "snap camera", "broadcast", "iriun", "dummy",
    "loopback", "v4l2loopback",
)

# Denenecek (fourcc, genişlik, yükseklik) sırası: önce sıkıştırmasız.
AUTO_MODES = (
    ("YUY2", 1920, 1080),
    ("NV12", 1920, 1080),
    ("YUY2", 1280, 1024),
    ("YUY2", 1280, 720),
    ("MJPG", 1920, 1080),
)
COMPRESSED = {"MJPG", "MJPEG", "H264", "HEVC"}
MIN_FPS = 15.0


def score_name(name: str) -> int:
    n = (name or "").lower()
    s = 0
    if any(h in n for h in CAPTURE_HINTS):
        s += 10
    if any(h in n for h in WEBCAM_HINTS):
        s -= 10
    elif "camera" in n or "kamera" in n or n.endswith(" cam"):
        s -= 3
    return s


def api_preference() -> list[int]:
    if SYSTEM == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if SYSTEM == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if SYSTEM == "Darwin":
        return [cv2.CAP_AVFOUNDATION]
    return [cv2.CAP_ANY]


def api_name(api: int) -> str:
    try:
        return cv2.videoio_registry.getBackendName(api)
    except Exception:
        return str(api)


# ------------------------------------------------------------- listeleme
def _linux_devices(root: Path = Path("/sys/class/video4linux")) -> list[dict]:
    out = []
    if not root.exists():
        return out

    def num(p):
        t = p.name[5:]
        return int(t) if t.isdigit() else 10_000

    for d in sorted(root.glob("video*"), key=num):
        n = num(d)
        if n == 10_000:
            continue
        try:
            name = (d / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        # UVC her cihaz için ikinci bir "metadata" düğümü açar (index 1);
        # görüntü veren düğüm index 0'dır.
        try:
            if int((d / "index").read_text().strip()) != 0:
                continue
        except (OSError, ValueError):
            pass
        out.append({"index": n, "name": name, "path": f"/dev/video{n}"})
    return out


def _windows_devices() -> list[dict] | None:
    """DirectShow cihaz adları; sıra OpenCV CAP_DSHOW indeksleriyle aynıdır."""
    try:
        import comtypes  # pygrabber bağımlılığı
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    try:
        comtypes.CoInitialize()
    except OSError:
        pass
    try:
        names = FilterGraph().get_input_devices()
        return [{"index": i, "name": n} for i, n in enumerate(names)]
    except Exception as e:
        log.warning("DirectShow cihaz listesi alınamadı: %s", e)
        return None
    finally:
        try:
            comtypes.CoUninitialize()
        except Exception:
            pass


def _probe_devices(max_index: int = 6) -> list[dict]:
    """Ad bilinmiyorsa indeksleri açarak dene (yavaş, son çare)."""
    out = []
    api = api_preference()[0]
    for i in range(max_index):
        cap = cv2.VideoCapture(i, api)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.append({"index": i, "name": f"Video cihazı #{i}",
                                "probe_size": [frame.shape[1], frame.shape[0]]})
        finally:
            cap.release()
    return out


def list_devices(probe: bool = True) -> list[dict]:
    if SYSTEM == "Linux":
        devs = _linux_devices()
    elif SYSTEM == "Windows":
        devs = _windows_devices()
        if devs is None:
            devs = _probe_devices() if probe else []
    else:
        devs = _probe_devices() if probe else []
    for d in devs:
        d["score"] = score_name(d["name"])
        d["looks_like_capture"] = d["score"] >= 10
        d["looks_like_webcam"] = d["score"] < 0
    return devs


def choose_device(devs: list[dict], preferred: str = "",
                  ignore: list[str] | tuple = ()) -> dict | None:
    ign = {x.lower() for x in ignore or ()}
    cands = [d for d in devs if d["name"].lower() not in ign]
    if preferred:
        for d in cands:
            if d["name"].lower() == preferred.lower():
                return d
    # Belirgin webcam'ler (-10) otomatik seçilmez; adı bilinmeyenler seçilir.
    cands = [d for d in cands if d.get("score", score_name(d["name"])) > -5]
    if not cands:
        return None
    return max(cands, key=lambda d: (d.get("score", 0), d["index"]))


# ----------------------------------------------------------- format seçimi
def fourcc_str(v: float) -> str:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return ""
    if i <= 0:
        return ""
    s = "".join(chr((i >> 8 * k) & 0xFF) for k in range(4))
    return s if s.isprintable() and s.strip() else ""


def _try_mode(index: int, api: int, fourcc: str, w: int, h: int, fps: int,
              opener: Callable, seconds: float = 1.2):
    """Bir modu açıp gerçek kare hızını ölçer. (cap, bilgi) ya da (None, hata)."""
    cap = opener(index, api)
    if not cap.isOpened():
        cap.release()
        return None, "açılamadı"
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(3):  # ısınma (ilk kareler gecikmeli/boş olabilir)
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        cap.release()
        return None, "kare okunamadı"
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds and n < 30:
        ok, f = cap.read()
        if not ok or f is None:
            break
        frame = f
        n += 1
    dt = max(1e-3, time.perf_counter() - t0)
    actual = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) or fourcc
    info = {"fourcc": actual or "?", "requested_fourcc": fourcc,
            "width": int(frame.shape[1]), "height": int(frame.shape[0]),
            "fps": round(n / dt, 1), "backend": api_name(api),
            "compressed": (actual or fourcc).upper() in COMPRESSED}
    return cap, info


def negotiate(index: int, cfg: dict, opener: Callable = cv2.VideoCapture,
              apis: list[int] | None = None, cached: dict | None = None,
              log_fn: Callable[[str], None] | None = None):
    """En iyi modu bulup açık VideoCapture döndürür: (cap, info)."""
    apis = apis or api_preference()
    say = log_fn or (lambda m: log.info(m))
    fourcc_cfg = (cfg.get("fourcc") or "auto").upper()
    fps = int(cfg.get("fps") or 60)

    if fourcc_cfg not in ("", "AUTO"):
        modes = [(fourcc_cfg, int(cfg.get("width") or 1920),
                  int(cfg.get("height") or 1080))]
    else:
        modes = list(AUTO_MODES)
    # Bu cihaz için daha önce bulunan mod varsa önce onu dene
    if cached and fourcc_cfg in ("", "AUTO"):
        c = (cached["fourcc"], cached["width"], cached["height"])
        modes = [c] + [m for m in modes if m != c]

    tried = []
    working_api = None
    for (fc, w, h) in modes:
        for api in ([working_api] if working_api is not None else apis):
            cap, info = _try_mode(index, api, fc, w, h, fps, opener)
            if cap is None:
                continue
            working_api = api
            say(f"{fc} {w}x{h} → {info['fourcc']} {info['width']}x"
                f"{info['height']} {info['fps']} fps ({info['backend']})")
            good = info["fps"] >= MIN_FPS
            if good and (not info["compressed"] or len(modes) == 1):
                info["mode"] = [fc, w, h]
                return cap, info
            cap.release()
            info["mode"] = [fc, w, h]
            tried.append(info)
            break
        if working_api is None:
            break  # cihaz hiçbir API ile açılmadı
    if not tried:
        raise RuntimeError(f"Cihaz #{index} açılamadı.")

    # Hiçbiri ideal değil: önce yeterli fps, sonra sıkıştırmasız, sonra çözünürlük
    def rank(i):
        return (i["fps"] >= MIN_FPS, not i["compressed"],
                i["width"] * i["height"], i["fps"])
    best = max(tried, key=rank)
    fc, w, h = best["mode"]
    cap, info = _try_mode(index, working_api, fc, w, h, fps, opener, seconds=0.3)
    if cap is None:
        raise RuntimeError(f"Cihaz #{index} yeniden açılamadı.")
    info["mode"] = best["mode"]
    return cap, info


def is_no_signal(frame) -> bool:
    """Neredeyse tek renk kare (siyah/mavi ekran) → sinyal yok."""
    if frame is None:
        return True
    h, w = frame.shape[:2]
    # Ortalama alarak küçültmek benek dokusunu siler ve homojen dokuyu
    # "düz" gösterir; bu yüzden ızgara üzerinden doğrudan piksel örneklenir.
    sy, sx = max(1, h // 48), max(1, w // 64)
    small = frame[::sy, ::sx]
    # Kanal başına değişkenlik: düz mavi ekranda da her kanal sabittir
    flat = small.reshape(-1, small.shape[2] if small.ndim == 3 else 1).astype(np.float32)
    return float(flat.std(axis=0).max()) < 4.0
