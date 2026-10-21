"""Vaka kaydı: data/cases/<vaka_id>/ altında görüntü, maske, JSON ve rapor.

KVKK: Görüntüler gizlilik maskeleri uygulanmış halde kaydedilir. Hasta adı/
TC kimlik no saklanmaz; yalnızca kullanıcının girdiği çalışma/protokol no
(takma ad) tutulur.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

import cv2
import numpy as np

from . import tirads

VIEW_LABELS = {"transverse": "Transvers", "longitudinal": "Longitudinal"}
LOCATION_LABELS = {
    "right": "Sağ lob", "left": "Sol lob", "isthmus": "İstmus",
    "upper": "üst", "mid": "orta", "lower": "alt",
}


def _safe(s: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")[:n].strip("_")


def new_case_id(study_no: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = _safe(study_no)
    return f"{stamp}-{tag + '-' if tag else ''}{secrets.token_hex(2)}"


def overlay(frame: np.ndarray, mask: np.ndarray | None, lines: list) -> np.ndarray:
    out = frame.copy()
    if mask is not None and mask.any():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        tint = out.copy()
        cv2.drawContours(tint, cs, -1, (0, 200, 255), cv2.FILLED)
        out = cv2.addWeighted(tint, 0.18, out, 0.82, 0)
        cv2.drawContours(out, cs, -1, (0, 200, 255), 2)
    for ln in lines:
        (x1, y1), (x2, y2) = ln["points"]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (80, 255, 80), 2)
        if ln.get("label"):
            cv2.putText(out, ln["label"], (int(x2) + 6, int(y2) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
    return out


def build_report(case: dict) -> str:
    f = case.get("features", {})
    s = case.get("score", {})
    r = case.get("recommendation", {})
    m = case.get("measurements", {})
    loc = case.get("location", {})
    lex = tirads.lexicon()

    def lab(cat, val):
        for o in lex[cat]:
            if o["value"] == val:
                return f"{o['label']} ({o['points']})"
        return "-"

    def mm(v):
        return "-" if v in (None, "") else f"{float(v):.1f} mm"

    side = LOCATION_LABELS.get(loc.get("side"), "")
    pole = LOCATION_LABELS.get(loc.get("pole"), "")
    lines = [
        "TİROİD NODÜLÜ – ACR TI-RADS DEĞERLENDİRMESİ",
        f"Vaka: {case['case_id']}   Çalışma no: {case.get('study_no') or '-'}",
        f"Tarih: {case['created_at']}",
        f"Lokalizasyon: {(side + ' ' + pole).strip() or '-'}",
        "",
        "Boyutlar:",
        f"  Transvers genişlik: {mm(m.get('width_mm'))}",
        f"  AP (ön-arka):       {mm(m.get('ap_mm'))}",
        f"  Longitudinal:       {mm(m.get('length_mm'))}",
        f"  En büyük boyut:     {mm(m.get('max_mm'))}",
        "",
        "ACR TI-RADS:",
        f"  Kompozisyon:  {lab('composition', f.get('composition'))}",
        f"  Ekojenite:    {lab('echogenicity', f.get('echogenicity'))}",
        f"  Şekil:        {lab('shape', f.get('shape'))}",
        f"  Kenar:        {lab('margin', f.get('margin'))}",
        "  Ekojenik odaklar: " + (", ".join(
            lab("foci", x) for x in (f.get("foci") or [])) or "-"),
        f"  Toplam: {s.get('total', '-')} puan → {s.get('level_text', '-')}",
        "",
        f"Öneri: {r.get('text', '-')}",
    ]
    for n in s.get("notes") or []:
        lines.append(f"Not: {n}")
    if case.get("notes"):
        lines += ["", f"Radyolog notu: {case['notes']}"]
    ai = case.get("ai", {})
    dices = [f"{VIEW_LABELS.get(k, k)} {a['ai_vs_final_dice']:.2f}"
             for k, a in (ai.get("views") or {}).items()
             if a and a.get("ai_vs_final_dice") is not None]
    lines += [
        "",
        f"Segmentasyon: {ai.get('summary', 'manuel')}",
    ] + ([f"AI önerisi ile son kontur uyumu (Dice): {', '.join(dices)}"]
         if dices else []) + [
        f"Onaylayan radyolog: {case.get('radiologist') or '-'}",
        "",
        "Bu çıktı araştırma amaçlı karar destek yazılımı tarafından "
        "üretilmiştir; nihai değerlendirme radyoloğa aittir.",
    ]
    return "\n".join(lines)


def save_case(cases_dir: Path, case: dict, frames: dict) -> Path:
    """frames: view -> {"image": np.ndarray, "mask": np.ndarray|None,
    "ai_mask": np.ndarray|None, "lines": [...]}"""
    d = cases_dir / case["case_id"]
    d.mkdir(parents=True, exist_ok=False)
    for view, fr in frames.items():
        cv2.imwrite(str(d / f"{view}.png"), fr["image"])
        if fr.get("mask") is not None:
            cv2.imwrite(str(d / f"{view}_mask.png"),
                        (fr["mask"] > 0).astype(np.uint8) * 255)
        if fr.get("ai_mask") is not None:  # AI'ın düzeltilmemiş önerisi
            cv2.imwrite(str(d / f"{view}_mask_ai.png"),
                        (fr["ai_mask"] > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(d / f"{view}_overlay.jpg"),
                    overlay(fr["image"], fr.get("mask"), fr.get("lines", [])),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = build_report(case)
    (d / "report.txt").write_text(report, encoding="utf-8")
    (d / "case.json").write_text(json.dumps(case, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return d


def list_cases(cases_dir: Path, limit: int = 100) -> list[dict]:
    if not cases_dir.exists():
        return []
    out = []
    for d in sorted(cases_dir.iterdir(), reverse=True)[:limit]:
        p = d / "case.json"
        if not p.exists():
            continue
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "case_id": c.get("case_id"), "created_at": c.get("created_at"),
            "study_no": c.get("study_no"),
            "level_text": c.get("score", {}).get("level_text"),
            "max_mm": c.get("measurements", {}).get("max_mm"),
            "action": c.get("recommendation", {}).get("action"),
        })
    return out
