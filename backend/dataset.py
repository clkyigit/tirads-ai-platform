"""Eğitim veri seti: etiket kaydı, istatistik ve dışa aktarım.

Kayıt yapısı (data/dataset/):
    <hasta_kodu>/<nodül_no>/<etiket_id>/
        image.png      tam kare (gizlilik maskeli), capture'dan geldiği gibi
        mask.png       radyoloğun son maskesi (0/255); negatif örnekte boş
        mask_ai.png    model kullanıldıysa düzeltilmemiş AI maskesi
        meta.json      görünüm, ROI (px), mm/px, TI-RADS, etiketleyen, kaynak...

Dışa aktarımda görüntüler ROI'ye kırpılır, çünkü uygulama çıkarımda da
yalnızca ROI'yi modele verir. Bölme hasta düzeyindedir: aynı hastanın
görüntüleri asla hem eğitimde hem testte bulunmaz.
"""

from __future__ import annotations

import json
import random
import re
import secrets
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

VIEWS = ("transverse", "longitudinal")


def safe_id(s: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip())[:n].strip("_")


def save_annotation(root: Path, *, image: np.ndarray, mask: np.ndarray | None,
                    ai_mask: np.ndarray | None, meta: dict) -> Path:
    pid = safe_id(meta.get("patient_id", ""))
    nid = safe_id(meta.get("nodule_id", "")) or "N1"
    if not pid:
        raise ValueError("Hasta takma kodu gerekli (hasta düzeyinde bölme için).")
    if meta.get("view") not in VIEWS:
        raise ValueError("Görünüm transverse veya longitudinal olmalı.")
    negative = bool(meta.get("negative"))
    if not negative and (mask is None or not mask.any()):
        raise ValueError("Maske yok. Konturu çizin ya da 'nodül yok (negatif)' "
                         "olarak işaretleyin.")
    aid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = root / pid / nid / aid
    d.mkdir(parents=True, exist_ok=False)
    H, W = image.shape[:2]
    cv2.imwrite(str(d / "image.png"), image)
    m = np.zeros((H, W), np.uint8) if negative or mask is None else mask
    cv2.imwrite(str(d / "mask.png"), (m > 0).astype(np.uint8) * 255)
    if ai_mask is not None:
        cv2.imwrite(str(d / "mask_ai.png"), (ai_mask > 0).astype(np.uint8) * 255)
    meta = {**meta, "patient_id": pid, "nodule_id": nid, "annotation_id": aid,
            "width": W, "height": H, "negative": negative,
            "mask_area_px": int((m > 0).sum()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ai_mask is not None and not negative and m.any():
        a, b = ai_mask > 0, m > 0
        meta["ai_vs_final_dice"] = float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return d


def iter_annotations(root: Path):
    if not root.exists():
        return
    for meta_p in sorted(root.glob("*/*/*/meta.json")):
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("deleted"):
            continue
        yield meta_p.parent, meta


def stats(root: Path) -> dict:
    patients, nodules = set(), set()
    views, annot, levels = Counter(), Counter(), Counter()
    n = neg = with_ai = 0
    dices = []
    for _, m in iter_annotations(root):
        n += 1
        patients.add(m["patient_id"])
        nodules.add((m["patient_id"], m["nodule_id"]))
        views[m.get("view")] += 1
        annot[m.get("annotator") or "-"] += 1
        neg += bool(m.get("negative"))
        lv = (m.get("tirads") or {}).get("level")
        levels[f"TR{lv}" if lv else "yok"] += 1
        if "ai_vs_final_dice" in m:
            with_ai += 1
            dices.append(m["ai_vs_final_dice"])
    return {"images": n, "patients": len(patients), "nodules": len(nodules),
            "negatives": neg, "views": dict(views), "annotators": dict(annot),
            "tirads_levels": dict(levels), "with_ai": with_ai,
            "ai_vs_final_dice_mean": float(np.mean(dices)) if dices else None}


def recent(root: Path, limit: int = 30) -> list[dict]:
    items = [(d, m) for d, m in iter_annotations(root)]
    items.sort(key=lambda x: x[1].get("annotation_id", ""), reverse=True)
    return [{"path": f"{m['patient_id']}/{m['nodule_id']}/{m['annotation_id']}",
             "view": m.get("view"), "negative": m.get("negative"),
             "annotator": m.get("annotator"), "created_at": m.get("created_at")}
            for _, m in items[:limit]]


def mark_deleted(root: Path, rel: str) -> bool:
    parts = rel.split("/")
    if len(parts) != 3 or not all(re.fullmatch(r"[A-Za-z0-9_-]{1,60}", p) for p in parts):
        raise ValueError("Geçersiz etiket yolu")
    p = root.joinpath(*parts) / "meta.json"
    if not p.exists():
        return False
    m = json.loads(p.read_text(encoding="utf-8"))
    m["deleted"] = time.strftime("%Y-%m-%d %H:%M:%S")  # yumuşak silme, iz kalır
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


# ------------------------------------------------------------------ dışa aktar
def _crop(meta: dict):
    W, H = meta["width"], meta["height"]
    r = meta.get("roi_px")
    if not r:
        return 0, 0, W, H
    x0, y0, x1, y1 = [int(v) for v in r]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, W, H
    return x0, y0, x1, y1


def split_patients(patients: list[str], val: float, test: float, seed: int) -> dict:
    ps = sorted(set(patients))
    random.Random(seed).shuffle(ps)
    n = len(ps)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    if n >= 3:
        n_test = max(n_test, 1) if test > 0 else 0
        n_val = max(n_val, 1) if val > 0 else 0
    return {"test": ps[:n_test], "val": ps[n_test:n_test + n_val],
            "train": ps[n_test + n_val:]}


def export(root: Path, out: Path, *, val=0.15, test=0.15, seed=42,
           include_negatives=True, nnunet_id: int | None = None,
           split_file: Path | None = None) -> dict:
    """Etiketleri eğitim formatına aktarır.

    out/
      images/<id>.png, masks/<id>.png   (ROI'ye kırpılmış; maske 0/1)
      index.csv                          id, hasta, nodül, görünüm, mm/px, split...
      splits.json                        hasta düzeyinde train/val/test
    nnunet_id verilirse ayrıca out/nnUNet_raw/DatasetXXX_Thyroid/ (nnU-Net v2).

    split_file verilirse önceki bölme korunur; yeni hastalar eğitime eklenir.
    Böylece test seti zamanla değişmez.
    """
    anns = list(iter_annotations(root))
    if not anns:
        raise ValueError(f"Etiket bulunamadı: {root}")
    if not include_negatives:
        anns = [(d, m) for d, m in anns if not m.get("negative")]
    patients = [m["patient_id"] for _, m in anns]
    if split_file and split_file.exists():
        prev = json.loads(split_file.read_text(encoding="utf-8"))
        known = {p: s for s, ps in prev.items() for p in ps}
        splits = {"train": [], "val": [], "test": []}
        for p in sorted(set(patients)):
            splits[known.get(p, "train")].append(p)
    else:
        splits = split_patients(patients, val, test, seed)
    which = {p: s for s, ps in splits.items() for p in ps}

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    rows = ["id,patient_id,nodule_id,view,split,negative,mm_per_px,"
            "width,height,tirads_level,annotator"]
    counts = Counter()
    for d, m in anns:
        img = cv2.imread(str(d / "image.png"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        x0, y0, x1, y1 = _crop(m)
        img, msk = img[y0:y1, x0:x1], (msk[y0:y1, x0:x1] > 127).astype(np.uint8)
        iid = f"{m['patient_id']}_{m['nodule_id']}_{m['annotation_id']}"
        cv2.imwrite(str(out / "images" / f"{iid}.png"), img)
        cv2.imwrite(str(out / "masks" / f"{iid}.png"), msk)
        sp = which[m["patient_id"]]
        counts[sp] += 1
        lvl = (m.get("tirads") or {}).get("level") or ""
        rows.append(",".join(str(x) for x in (
            iid, m["patient_id"], m["nodule_id"], m.get("view"), sp,
            int(bool(m.get("negative"))), m.get("mm_per_px") or "",
            img.shape[1], img.shape[0], lvl,
            safe_id(m.get("annotator") or ""))))
    (out / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = {"images": sum(counts.values()), "by_split": dict(counts),
               "patients": {k: len(v) for k, v in splits.items()}}
    if nnunet_id is not None:
        summary["nnunet"] = str(_export_nnunet(out, int(nnunet_id)))
    return summary


def _export_nnunet(out: Path, did: int) -> Path:
    """nnU-Net v2 'natural image' 2D formatı (PNG, 3 kanal -> 3 dosya)."""
    base = out / "nnUNet_raw" / f"Dataset{did:03d}_ThyroidUS"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    lines = (out / "index.csv").read_text(encoding="utf-8").splitlines()[1:]
    ntr = 0
    for ln in lines:
        iid, _, _, _, sp = ln.split(",")[:5]
        img = cv2.imread(str(out / "images" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(out / "masks" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        tr = sp in ("train", "val")
        ntr += tr
        cv2.imwrite(str(base / ("imagesTr" if tr else "imagesTs") / f"{iid}_0000.png"), img)
        cv2.imwrite(str(base / ("labelsTr" if tr else "labelsTs") / f"{iid}.png"), msk)
    (base / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "US"}, "labels": {"background": 0, "nodule": 1},
        "numTraining": ntr, "file_ending": ".png"}, indent=2), encoding="utf-8")
    # nnU-Net kendi 5 katlı bölmesini görüntü düzeyinde yapar. Hasta düzeyinde
    # bölme için bu dosyayı nnUNet_preprocessed/DatasetXXX_ThyroidUS/ altına
    # kopyalayıp eğitimi fold 0 ile başlatın.
    splits = json.loads((out / "splits.json").read_text())
    by_patient = defaultdict(list)
    for ln in lines:
        iid, pid, _, _, sp = ln.split(",")[:5]
        if sp in ("train", "val"):
            by_patient[pid].append(iid)
    folds = [{"train": [i for p in splits["train"] for i in by_patient[p]],
              "val": [i for p in splits["val"] for i in by_patient[p]]}]
    (base / "splits_final.json").write_text(json.dumps(folds, indent=1))
    return base
