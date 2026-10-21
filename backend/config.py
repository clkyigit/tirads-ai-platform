"""Ayarlar: data/config.json (yoksa varsayılanlar). Arayüzden değiştirilen
ayarlar bu dosyaya yazılır."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8765},
    # type: "device" (capture card), "file" (video dosyası), "demo"
    "source": {"type": "demo", "index": 0, "path": "", "width": 1920,
               "height": 1080, "fps": 30, "fourcc": ""},
    # Görüntü alanı (normalize x, y, w, h). Model yalnızca burayı görür.
    "roi": None,
    # Kayıtlarda ve akışta siyaha boyanan alanlar (hasta adı vb.)
    "privacy_masks": [],
    "calibration": {"mm_per_px": None, "active_preset": None, "presets": {}},
    "model": {
        "backend": "auto",          # auto | torch | onnx | demo | none
        "path": "models/unetpp.pth",
        "encoder": "resnet34",      # eğitimdeki smp encoder adı
        "in_channels": 3,
        "classes": 1,
        "nodule_class": 1,          # classes > 1 ise nodül kanalı
        "input_size": [512, 512],   # [genişlik, yükseklik]
        "normalize": "imagenet",    # imagenet | zero_one | zscore
        "output_activation": "auto",
        "threshold": 0.5,
        "device": "auto",
    },
    "stream": {"jpeg_quality": 80, "max_fps": 25},
    "data_dir": "data",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, path: Path | None = None):
        self.path = path or ROOT / "data" / "config.json"
        self._lock = threading.Lock()
        user = {}
        if self.path.exists():
            user = json.loads(self.path.read_text(encoding="utf-8"))
        self.data = _merge(DEFAULTS, user)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)

    def update(self, patch: dict, replace: tuple[str, ...] = ()) -> dict:
        """patch'i birleştirir. replace'teki noktalı anahtarlar
        (ör. "calibration.presets") birleştirilmeden tamamen değiştirilir."""
        with self._lock:
            new = _merge(self.data, patch)
            for dotted in replace:
                keys = dotted.split(".")
                src, dst = patch, new
                for k in keys[:-1]:
                    src, dst = src[k], dst[k]
                dst[keys[-1]] = copy.deepcopy(src[keys[-1]])
            self.data = new
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)
            return copy.deepcopy(self.data)

    def data_dir(self) -> Path:
        p = Path(self.data["data_dir"])
        return p if p.is_absolute() else ROOT / p
