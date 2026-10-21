"""UNet++ nodül segmentasyonu eğitimi (segmentation-models-pytorch).

Önce etiketleri dışa aktarın (arayüzde "Eğitim için dışa aktar" ya da
`python -m tools.export_dataset`), sonra:

    python -m tools.train_unetpp --data data/export --out models

Çıktılar (models/):
    unetpp.pth   en iyi doğrulama Dice'lı ağırlıklar (state_dict)
    unetpp.onnx  aynı modelin ONNX hali (uygulama torch olmadan çalıştırır)
    unetpp.json  ön işleme ayarları + metrikler; uygulama bunu otomatik okur
    train_log.csv, test_report.json (--eval-test ile)

Ön işleme uygulamadakiyle birebir aynıdır: ROI'ye kırpılmış görüntü en-boy
oranı korunmadan input_size'a yeniden boyutlandırılır, RGB, ImageNet
normalizasyonu. Bu yüzden ayarları değiştirirseniz json'a yazılır ve
uygulama da aynısını kullanır.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

from backend import measure

try:  # numpy kısımları torch olmadan da test edilebilsin
    import torch
except ImportError:  # pragma: no cover
    torch = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------- veri
def load_index(data: Path) -> list[dict]:
    with (data / "index.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["negative"] = r["negative"] == "1"
        r["mm_per_px"] = float(r["mm_per_px"]) if r["mm_per_px"] else None
    return rows


def augment(img: np.ndarray, msk: np.ndarray, rng: random.Random):
    """USG'ye uygun hafif artırma. Dikey çevirme yok (derinlik yönü anlamlı)."""
    h, w = msk.shape
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.8:
        ang = rng.uniform(-10, 10)
        sc = rng.uniform(0.85, 1.15)
        tx, ty = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        msk = cv2.warpAffine(np.ascontiguousarray(msk), M, (w, h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    f = img.astype(np.float32)
    if rng.random() < 0.8:  # parlaklık / kontrast (gain, dinamik aralık)
        f = f * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.5:  # gama
        g = rng.uniform(0.75, 1.35)
        f = 255.0 * np.clip(f / 255.0, 0, 1) ** g
    if rng.random() < 0.25:  # hafif bulanıklık (farklı odak/ çözünürlük)
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.2))
    if rng.random() < 0.3:  # çarpımsal benek gürültüsü
        f = f * np.random.default_rng(rng.randrange(1 << 30)).normal(
            1.0, rng.uniform(0.03, 0.10), f.shape[:2])[..., None].astype(np.float32)
    img = np.clip(f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def to_input(img_bgr: np.ndarray, size: tuple[int, int], in_channels: int,
             normalize: str) -> np.ndarray:
    """backend.segmentation.Segmenter._preprocess ile aynı (C×H×W float32)."""
    x = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    if in_channels == 1:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    if normalize == "imagenet":
        x = (x - (IMAGENET_MEAN.mean() if in_channels == 1 else IMAGENET_MEAN)) / \
            (IMAGENET_STD.mean() if in_channels == 1 else IMAGENET_STD)
    elif normalize == "zscore":
        x = (x - x.mean()) / (x.std() + 1e-6)
    return np.ascontiguousarray(x.transpose(2, 0, 1), np.float32)


class SegDataset:
    """Harita tipi veri seti (torch.utils.data.DataLoader ile uyumlu)."""

    def __init__(self, data: Path, rows: list[dict], size, in_channels, normalize,
                 train: bool, seed: int = 0):
        self.data, self.rows = data, rows
        self.size, self.cin, self.norm = tuple(size), in_channels, normalize
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.data / "images" / f"{r['id']}.png"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(self.data / "masks" / f"{r['id']}.png"), cv2.IMREAD_GRAYSCALE)
        if self.train:
            # DataLoader işçileri veri setinin kopyasını tutar; her çağrıda
            # taze tohum, böylece her epoch farklı artırma görülür.
            rng = random.Random(int.from_bytes(os.urandom(4), "little"))
            img, msk = augment(img, msk, rng)
        x = to_input(img, self.size, self.cin, self.norm)
        y = cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST)
        return x, (y > 0).astype(np.float32)[None]


# --------------------------------------------------------------- metrikler
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def eval_case(prob: np.ndarray, gt: np.ndarray, mm_per_px, thr: float) -> dict:
    """Uygulamadaki gibi: radyolog nodüle tıklar (GT ağırlık merkezi),
    tıklanan bileşen seçilir. prob ve gt orijinal ROI boyutunda."""
    out = {}
    if gt.any():
        ys, xs = np.nonzero(gt)
        click = (int(xs.mean()), int(ys.mean()))
        pred = measure.clean_mask(prob, thr, click)
        out["dice"] = dice(pred, gt)
        if pred.any() and mm_per_px:
            mp = measure.contour_metrics(measure.mask_to_contour(pred))
            mg = measure.contour_metrics(measure.mask_to_contour(gt))
            pm = max(mp["width_px"], mp["height_px"], mp["feret_px"]) * mm_per_px
            gm = max(mg["width_px"], mg["height_px"], mg["feret_px"]) * mm_per_px
            out["size_err_mm"] = abs(pm - gm)
            out["width_err_mm"] = abs(mp["width_px"] - mg["width_px"]) * mm_per_px
            out["ap_err_mm"] = abs(mp["height_px"] - mg["height_px"]) * mm_per_px
    else:  # negatif örnek: tıklamasız en büyük bileşen var mı?
        out["false_positive"] = float(measure.clean_mask(prob, thr).any())
    return out


def summarize(results: list[dict]) -> dict:
    def mean(k):
        v = [r[k] for r in results if k in r]
        return (float(np.mean(v)) if v else None), len(v)
    s = {}
    for k in ("dice", "size_err_mm", "width_err_mm", "ap_err_mm", "false_positive"):
        s[k], s[f"n_{k}"] = mean(k)
    v = [r["size_err_mm"] for r in results if "size_err_mm" in r]
    s["size_err_mm_p90"] = float(np.percentile(v, 90)) if v else None
    return s


# ------------------------------------------------------------------ eğitim
def build_model(encoder, weights, in_channels):
    import segmentation_models_pytorch as smp
    try:
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights,
                                in_channels=in_channels, classes=1)
    except Exception as e:
        if weights is None:
            raise
        print(f"UYARI: '{weights}' ön eğitimli ağırlıklar indirilemedi ({e}); "
              "rastgele başlatılıyor. İnternetsiz makinede --encoder-weights none "
              "kullanın ya da ağırlıkları önceden indirin.")
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None,
                                in_channels=in_channels, classes=1)


def predict_full(model, img_bgr, size, cin, norm, device) -> np.ndarray:
    x = torch.from_numpy(to_input(img_bgr, size, cin, norm)[None]).to(device)
    with torch.inference_mode():
        p = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    h, w = img_bgr.shape[:2]
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def evaluate(model, data, rows, args, device) -> dict:
    model.eval()
    res = []
    for r in rows:
        img = cv2.imread(str(data / "images" / f"{r['id']}.png"), cv2.IMREAD_COLOR)
        gt = (cv2.imread(str(data / "masks" / f"{r['id']}.png"),
                         cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        prob = predict_full(model, img, tuple(args.size), args.in_channels,
                            args.normalize, device)
        e = eval_case(prob, gt, r["mm_per_px"], args.threshold)
        e["id"] = r["id"]
        res.append(e)
    return {"summary": summarize(res), "cases": res}


def loss_fn(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    has = (target.sum((1, 2, 3)) > 0).float()
    d = 1 - (2 * inter + 1) / (den + 1)
    # Negatif örneklerde Dice kaybı anlamsız; yalnızca BCE ile öğrenilir
    dl = (d * has).sum() / has.sum().clamp(min=1)
    return bce + dl


def export_onnx(model, path: Path, size, cin):
    model.eval().cpu()
    dummy = torch.randn(1, cin, size[1], size[0])
    torch.onnx.export(model, dummy, str(path), input_names=["image"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "N"}, "logits": {0: "N"}})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/export"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--name", default="unetpp")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="imagenet | none")
    ap.add_argument("--size", type=int, nargs=2, default=[512, 384],
                    metavar=("G", "Y"), help="model giriş boyutu (genişlik yükseklik)")
    ap.add_argument("--in-channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--normalize", default="imagenet",
                    choices=["imagenet", "zero_one", "zscore"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=30,
                    help="doğrulama Dice'ı bu kadar epoch iyileşmezse dur")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eval-test", action="store_true",
                    help="eğitim yerine en iyi modeli TEST setinde değerlendir "
                         "(test setine yalnızca en sonda bakın)")
    args = ap.parse_args()

    if any(v % 32 for v in args.size):
        raise SystemExit("--size değerleri 32'nin katı olmalı (ör. 512 384).")
    if torch is None:
        raise SystemExit("PyTorch kurulu değil: pip install -r requirements-ml.txt")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu")
    rows = load_index(args.data)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"Cihaz: {dev} · eğitim {len(tr)} · doğrulama {len(va)} · test {len(te)} görüntü")
    if not tr or not va:
        raise SystemExit("Eğitim ve doğrulama setinde en az birer görüntü gerekli.")
    args.out.mkdir(parents=True, exist_ok=True)
    pth = args.out / f"{args.name}.pth"
    weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights

    # Test değerlendirmesinde ön eğitimli ağırlık indirmeye gerek yok
    model = build_model(args.encoder, None if args.eval_test else weights,
                        args.in_channels)

    if args.eval_test:
        model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
        model.to(dev)
        rep = evaluate(model, args.data, te, args, dev)
        (args.out / "test_report.json").write_text(json.dumps(rep, indent=1))
        print("TEST:", json.dumps(rep["summary"], indent=1))
        return

    model.to(dev)
    ds = SegDataset(args.data, tr, args.size, args.in_channels, args.normalize,
                    train=True, seed=args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=len(tr) > args.batch,
                                     pin_memory=dev == "cuda",
                                     persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(dl)), pct_start=0.05)
    use_amp = dev == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_p = args.out / "train_log.csv"
    with log_p.open("w", encoding="utf-8") as f:
        f.write("epoch,loss,val_dice,val_size_err_mm,val_fp,sec\n")
    best, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        s = evaluate(model, args.data, va, args, dev)["summary"]
        vd = s["dice"] if s["dice"] is not None else 0.0
        dt = time.time() - t0
        fmt = lambda v: "" if v is None else f"{v:.4f}"
        with log_p.open("a", encoding="utf-8") as f:
            f.write(f"{ep},{tot / max(1, n):.4f},{fmt(s['dice'])},"
                    f"{fmt(s['size_err_mm'])},{fmt(s['false_positive'])},{dt:.0f}\n")
        msg = (f"[{ep:3d}/{args.epochs}] kayıp {tot / max(1, n):.4f} · val Dice {vd:.4f}"
               + (f" · boyut hatası {s['size_err_mm']:.2f} mm" if s["size_err_mm"] is not None else "")
               + f" · {dt:.0f} sn")
        if vd > best:
            best, bad = vd, 0
            torch.save(model.state_dict(), pth)
            meta = {
                "architecture": "UnetPlusPlus", "encoder": args.encoder,
                "in_channels": args.in_channels, "classes": 1,
                "input_size": list(args.size), "normalize": args.normalize,
                "output_activation": "sigmoid", "threshold": args.threshold,
                "trained_at": time.strftime("%Y-%m-%d %H:%M"), "epoch": ep,
                "val_dice": vd, "val_size_err_mm": s["size_err_mm"],
                "val_size_err_mm_p90": s["size_err_mm_p90"],
                "n_train": len(tr), "n_val": len(va),
                "data": str(args.data),
            }
            (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
            msg += "  ← en iyi"
        else:
            bad += 1
        print(msg, flush=True)
        if bad >= args.patience:
            print(f"{args.patience} epoch iyileşme yok, durduruluyor.")
            break

    print(f"En iyi doğrulama Dice: {best:.4f} → {pth}")
    model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    try:
        export_onnx(model, args.out / f"{args.name}.onnx", args.size, args.in_channels)
        print(f"ONNX: {args.out / (args.name + '.onnx')}")
    except Exception as e:
        print(f"ONNX dışa aktarılamadı ({e}); uygulama .pth ile de çalışır.")
    print("Uygulamada Ayarlar → Segmentasyon modeli → yol: "
          f"{args.out / (args.name + '.onnx')} (veya .pth) → Modeli yükle")


if __name__ == "__main__":
    main()
