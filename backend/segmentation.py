"""UNet++ nodül segmentasyonu (PyTorch .pth veya ONNX).

Desteklenen modeller:
  * .pth / .pt : segmentation_models_pytorch.UnetPlusPlus state_dict'i
                 (encoder, in_channels, classes config'ten okunur)
  * .onnx      : Aynı giriş/çıkış sözleşmesine sahip herhangi bir model
                 (ör. kendi UNet++'ınızın veya nnU-Net modelinizin ONNX
                 dışa aktarımı). Giriş NCHW float32, çıkış N×C×H×W logit
                 veya olasılık.
  * "demo"     : Model dosyası olmadan arayüzü denemek için klasik eşikleme.
                 KLİNİK DEĞİLDİR.

Model girişi yalnızca görüntü alanıdır (ROI): cihazın menü/yazı alanları
kırpılır, çıkan maske tam kare koordinatlarına geri yerleştirilir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from . import measure

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class SegmenterError(RuntimeError):
    pass


class Segmenter:
    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = dict(cfg)
        self.base_dir = base_dir
        self.backend: str | None = None
        self.error: str | None = None
        self.device = "cpu"
        self._model = None
        self._sess = None
        self._input_name = None
        self.sidecar: dict = {}
        try:
            self._load()
        except Exception as e:  # yükleme hatası uygulamayı durdurmasın
            log.exception("Model yüklenemedi")
            self.backend = None
            self.error = f"Model yüklenemedi: {e}"

    # ------------------------------------------------------------------ yükleme
    def _load(self):
        kind = (self.cfg.get("backend") or "auto").lower()
        if kind == "demo":
            self.backend = "demo"
            return
        if kind == "none":
            self.error = "Model devre dışı (config: model.backend = none)."
            return
        path = self.cfg.get("path") or ""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        if path and not p.exists():
            # Eğitim hem .onnx hem .pth üretir; hangisi varsa onu kullan
            for ext in (".onnx", ".pth", ".pt"):
                if p.with_suffix(ext).exists():
                    p = p.with_suffix(ext)
                    break
        if not path or not p.exists():
            self.error = (f"Model dosyası bulunamadı: {p}. Ağırlıkları bu yola "
                          "koyun ya da ayarlardan yolu değiştirin.")
            return
        self._apply_sidecar(p)
        if kind == "auto":
            kind = "onnx" if p.suffix.lower() == ".onnx" else "torch"
        if kind == "onnx":
            self._load_onnx(p)
        elif kind == "torch":
            self._load_torch(p)
        else:
            raise SegmenterError(f"Bilinmeyen model.backend: {kind}")

    SIDECAR_KEYS = ("encoder", "in_channels", "classes", "nodule_class",
                    "input_size", "normalize", "output_activation", "threshold",
                    "class_names")

    def _apply_sidecar(self, p: Path):
        """Eğitim scriptinin model dosyasının yanına yazdığı <ad>.json:
        ön işleme ayarlarını eğitimdekiyle birebir aynı yapar."""
        side = p.with_suffix(".json")
        if not side.exists():
            return
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Model yan dosyası okunamadı: %s", e)
            return
        for k in self.SIDECAR_KEYS:
            if k in meta:
                self.cfg[k] = meta[k]
        self.sidecar = {k: meta.get(k) for k in ("trained_at", "val_dice",
                                                  "val_size_err_mm", "n_train")
                        if k in meta}

    def _load_onnx(self, p: Path):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        want = [x for x in ("CUDAExecutionProvider", "DmlExecutionProvider",
                            "CoreMLExecutionProvider") if x in avail]
        providers = want + ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(p), providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # [N, C, H, W]; dinamik boyutlar str/None olabilir
        if isinstance(shape[1], int):
            self.cfg["in_channels"] = shape[1]
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            self.cfg["input_size"] = [shape[3], shape[2]]
        self.device = self._sess.get_providers()[0]
        self.backend = "onnx"

    def _load_torch(self, p: Path):
        try:
            import torch
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise SegmenterError(
                "PyTorch modeli için 'torch' ve 'segmentation-models-pytorch' "
                "kurulu olmalı (requirements-ml.txt).") from e
        dev = self.cfg.get("device") or "auto"
        if dev == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif getattr(torch.backends, "mps", None) and \
                    torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        model = smp.UnetPlusPlus(
            encoder_name=self.cfg.get("encoder", "resnet34"),
            encoder_weights=None,
            in_channels=int(self.cfg.get("in_channels", 3)),
            classes=int(self.cfg.get("classes", 1)),
        )
        # weights_only=True: güvenilmeyen pickle kodu çalıştırılmaz.
        ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model", "network_weights"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        state = {}
        for k, v in ckpt.items():
            for pre in ("module.", "model.", "net."):
                if k.startswith(pre):
                    k = k[len(pre):]
            state[k] = v
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise SegmenterError(
                f"Ağırlıklar mimariyle uyuşmuyor ({len(missing)} eksik anahtar, "
                f"ör. {missing[:3]}). config'teki encoder/in_channels/classes "
                "değerlerini eğitimdekiyle aynı yapın.")
        if unexpected:
            log.warning("Kullanılmayan %d anahtar: %s", len(unexpected),
                        unexpected[:5])
        model.eval().to(dev)
        self._model = model
        self.device = dev
        self.backend = "torch"

    # ----------------------------------------------------------------- çıkarım
    @property
    def ready(self) -> bool:
        return self.backend is not None

    def info(self) -> dict:
        return {"ready": self.ready, "backend": self.backend,
                "device": self.device, "error": self.error,
                "path": self.cfg.get("path"),
                "encoder": self.cfg.get("encoder"),
                "input_size": self.cfg.get("input_size"),
                "is_demo": self.backend == "demo",
                "trained": self.sidecar,
                "classes": int(self.cfg.get("classes", 1)),
                "multiclass": self.multiclass,
                "normalize": self.cfg.get("normalize"),
                "threshold": self.cfg.get("threshold")}

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        w, h = self.cfg.get("input_size") or [512, 512]
        cin = int(self.cfg.get("in_channels", 3))
        x = cv2.resize(img_bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        if cin == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        norm = self.cfg.get("normalize", "imagenet")
        if norm == "imagenet":
            if cin == 1:
                x = (x - IMAGENET_MEAN.mean()) / IMAGENET_STD.mean()
            else:
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
        elif norm == "zscore":
            x = (x - x.mean()) / (x.std() + 1e-6)
        # "zero_one": yalnızca /255
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)

    def _postprocess(self, out: np.ndarray) -> np.ndarray:
        out = np.asarray(out, np.float32)[0]  # C×H×W
        act = self.cfg.get("output_activation", "auto")
        classes = out.shape[0]
        if act == "auto":
            act = "sigmoid" if classes == 1 else "softmax"
        if act == "sigmoid":
            prob = 1.0 / (1.0 + np.exp(-out[0]))
        elif act == "softmax":
            e = np.exp(out - out.max(axis=0, keepdims=True))
            sm = e / e.sum(axis=0, keepdims=True)
            prob = sm[int(self.cfg.get("nodule_class", 1))]
        else:  # "none": model zaten olasılık veriyor
            prob = out[0] if classes == 1 else out[int(self.cfg.get("nodule_class", 1))]
        return prob

    @property
    def multiclass(self) -> bool:
        return self.ready and self.backend != "demo" and int(self.cfg.get("classes", 1)) > 1

    def _raw(self, img_bgr):
        x = self._preprocess(img_bgr)
        if self.backend == "onnx":
            return self._sess.run(None, {self._input_name: x})[0]
        import torch
        with torch.inference_mode():
            t = torch.from_numpy(x).to(self.device)
            return self._model(t).float().cpu().numpy()

    def predict_labels(self, img_bgr: np.ndarray, work_width: int = 0) -> np.ndarray:
        """ROI için etiket haritası (uint8, ROI boyutunda ya da work_width'e
        küçültülmüş). Tek sınıflı modelde 0/1 (nodül), çok sınıflıda 0..K-1."""
        if not self.ready:
            raise SegmenterError(self.error or "Model hazır değil.")
        h, w = img_bgr.shape[:2]
        tw, th_ = (w, h)
        if work_width and w > work_width:
            tw, th_ = work_width, int(round(h * work_width / w))
        if self.backend == "demo":
            prob = _demo_prob(img_bgr)
            lab = (prob >= float(self.cfg.get("threshold", 0.5))).astype(np.uint8)
            lab = cv2.morphologyEx(lab, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            # Demo: kenara değen koyu bantlar (cilt altı, derin zayıflama) nodül değil
            n, cc, stats, _ = cv2.connectedComponentsWithStats(lab, 8)
            hh, ww = lab.shape
            for i in range(1, n):
                x, y, bw, bh = stats[i, :4]
                if x <= 1 or y <= 1 or x + bw >= ww - 1 or y + bh >= hh - 1:
                    lab[cc == i] = 0
            return cv2.resize(lab, (tw, th_), interpolation=cv2.INTER_NEAREST)
        out = np.asarray(self._raw(img_bgr), np.float32)[0]
        if out.shape[0] == 1:
            prob = 1.0 / (1.0 + np.exp(-out[0])) if self.cfg.get("output_activation", "auto") != "none" else out[0]
            prob = cv2.resize(prob, (tw, th_), interpolation=cv2.INTER_LINEAR)
            return (prob >= float(self.cfg.get("threshold", 0.5))).astype(np.uint8)
        # Çok sınıflı: kanal başına yeniden boyutlandırıp argmax
        up = np.stack([cv2.resize(c, (tw, th_), interpolation=cv2.INTER_LINEAR) for c in out])
        return up.argmax(0).astype(np.uint8)

    def predict_prob(self, img_bgr: np.ndarray) -> np.ndarray:
        """img_bgr (ROI) için ROI boyutunda olasılık haritası."""
        if not self.ready:
            raise SegmenterError(self.error or "Model hazır değil.")
        h, w = img_bgr.shape[:2]
        if self.backend == "demo":
            return _demo_prob(img_bgr)
        prob = self._postprocess(self._raw(img_bgr))
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    def segment(self, frame_bgr: np.ndarray, roi: list | None,
                click: tuple[int, int] | None = None) -> dict:
        """Tam karede segmentasyon. roi: normalize [x, y, w, h] veya None."""
        H, W = frame_bgr.shape[:2]
        x0, y0, x1, y1 = roi_to_px(roi, W, H)
        crop = frame_bgr[y0:y1, x0:x1]
        prob = self.predict_prob(crop)
        local_click = None
        if click is not None:
            local_click = (click[0] - x0, click[1] - y0)
        thr = float(self.cfg.get("threshold", 0.5))
        m = measure.clean_mask(prob, thr, local_click)
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = m
        conf = float(prob[m > 0].mean()) if m.any() else 0.0
        return {"mask": full, "confidence": conf}


def roi_to_px(roi, W, H):
    if not roi:
        return 0, 0, W, H
    x, y, w, h = roi
    x0 = int(round(max(0.0, min(1.0, x)) * W))
    y0 = int(round(max(0.0, min(1.0, y)) * H))
    x1 = int(round(max(0.0, min(1.0, x + w)) * W))
    y1 = int(round(max(0.0, min(1.0, y + h)) * H))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, W, H
    return x0, y0, x1, y1


def _demo_prob(img_bgr: np.ndarray) -> np.ndarray:
    """Model yokken arayüz denemesi için: hipoekoik bölgeleri eşikler.
    Gerçek bir segmentasyon yöntemi DEĞİLDİR."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 3)
    tissue = g[g > 10]
    if tissue.size == 0:
        return np.zeros(g.shape, np.float32)
    t = np.percentile(tissue, 20)
    prob = np.clip((t - g.astype(np.float32)) / 20.0 + 0.5, 0, 1)
    prob[g <= 3] = 0  # tamamen siyah (görüntü dışı) alanlar
    return prob.astype(np.float32)
