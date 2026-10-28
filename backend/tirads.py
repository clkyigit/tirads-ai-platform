"""ACR TI-RADS (2017) puanlama ve boyut eşiğine göre yönetim önerisi.

Kaynak: Tessler FN ve ark. ACR Thyroid Imaging, Reporting and Data System
(TI-RADS): White Paper of the ACR TI-RADS Committee. JACR 2017;14:587-595.

Kurallar tek yerde (burada) tutulur; tarayıcı arayüzü puanı /api/score
üzerinden buradan alır, böylece arayüz ile rapor asla farklı hesaplamaz.
"""

from __future__ import annotations

COMPOSITION = {
    "cystic": ("Kistik veya neredeyse tamamen kistik", 0),
    "spongiform": ("Spongiform", 0),
    "mixed": ("Karışık kistik ve solid", 1),
    "solid": ("Solid veya neredeyse tamamen solid", 2),
}

ECHOGENICITY = {
    "anechoic": ("Anekoik", 0),
    "iso_hyper": ("Hiperekoik veya izoekoik", 1),
    "hypo": ("Hipoekoik", 2),
    "very_hypo": ("Belirgin hipoekoik", 3),
}

SHAPE = {
    "wider": ("Genişliği yüksekliğinden fazla (wider-than-tall)", 0),
    "taller": ("Yüksekliği genişliğinden fazla (taller-than-wide)", 3),
}

MARGIN = {
    "smooth": ("Düzgün", 0),
    "ill_defined": ("Belirsiz (ill-defined)", 0),
    "lobulated_irregular": ("Lobüle veya düzensiz", 2),
    "ete": ("Ekstratiroidal yayılım", 3),
}

# Ekojenik odaklar: birden fazla seçilebilir, puanlar toplanır.
FOCI = {
    "none": ("Yok veya büyük kuyruklu yıldız artefaktı", 0),
    "macro": ("Makrokalsifikasyon", 1),
    "rim": ("Periferik (rim) kalsifikasyon", 2),
    "punctate": ("Punktat ekojenik odaklar", 3),
}

CATEGORIES = {
    "composition": COMPOSITION,
    "echogenicity": ECHOGENICITY,
    "shape": SHAPE,
    "margin": MARGIN,
}

LEVEL_TEXT = {
    1: "TR1 – Benign",
    2: "TR2 – Şüpheli değil",
    3: "TR3 – Hafif şüpheli",
    4: "TR4 – Orta derecede şüpheli",
    5: "TR5 – Yüksek derecede şüpheli",
}

# (İİAB eşiği mm, takip eşiği mm, takip takvimi)
THRESHOLDS = {
    3: (25.0, 15.0, "1., 3. ve 5. yıllarda USG takibi"),
    4: (15.0, 10.0, "1., 2., 3. ve 5. yıllarda USG takibi"),
    5: (10.0, 5.0, "5 yıla kadar yıllık USG takibi"),
}


def lexicon() -> dict:
    """Arayüzün form seçeneklerini üretmesi için sözlük."""
    out = {}
    for key, table in {**CATEGORIES, "foci": FOCI}.items():
        out[key] = [
            {"value": v, "label": label, "points": pts}
            for v, (label, pts) in table.items()
        ]
    return out


def points_to_level(total: int) -> int:
    # ACR: 0 puan TR1, 2 puan TR2, 3 puan TR3, 4–6 TR4, >=7 TR5.
    # 1 puan ACR şemasında ayrıca tanımlanmamıştır; TR1 ile TR2'nin
    # yönetimi aynı (İİAB/takip yok) olduğundan TR1 kabul edilir.
    if total <= 1:
        return 1
    if total == 2:
        return 2
    if total == 3:
        return 3
    if total <= 6:
        return 4
    return 5


def score(features: dict) -> dict:
    """features: {composition, echogenicity, shape, margin: str, foci: [str]}.

    Eksik kategori varsa level None döner ve `missing` listelenir.
    """
    breakdown: dict[str, int] = {}
    missing: list[str] = []
    errors: list[str] = []
    notes: list[str] = []

    for key, table in CATEGORIES.items():
        val = features.get(key)
        if val in (None, ""):
            missing.append(key)
            continue
        if val not in table:
            errors.append(f"Geçersiz değer: {key}={val!r}")
            continue
        breakdown[key] = table[val][1]

    foci = features.get("foci") or []
    if isinstance(foci, str):
        foci = [foci]
    foci = list(dict.fromkeys(foci))  # tekrarları at, sırayı koru
    if not foci:
        missing.append("foci")
    else:
        bad = [f for f in foci if f not in FOCI]
        if bad:
            errors.append(f"Geçersiz ekojenik odak: {bad}")
        elif "none" in foci and len(foci) > 1:
            errors.append(
                "'Yok' seçeneği diğer ekojenik odaklarla birlikte seçilemez"
            )
        else:
            breakdown["foci"] = sum(FOCI[f][1] for f in foci)

    if errors:
        return {"ok": False, "errors": errors, "missing": missing,
                "breakdown": breakdown, "total": None, "level": None}

    comp = features.get("composition")
    if comp == "spongiform":
        # ACR: spongiform nodülde diğer kategorilere puan eklenmez.
        notes.append("Spongiform nodül: ACR'ye göre diğer kategorilere puan "
                     "eklenmez.")
        breakdown = {k: (0 if k != "composition" else v)
                     for k, v in breakdown.items()}
    if (features.get("echogenicity") == "anechoic"
            and comp not in (None, "", "cystic")):
        notes.append("Anekoik ekojenite yalnızca kistik/neredeyse tamamen "
                     "kistik nodüller için kullanılır; kontrol ediniz.")

    if missing:
        return {"ok": True, "errors": [], "missing": missing,
                "breakdown": breakdown, "total": None, "level": None,
                "notes": notes}

    total = sum(breakdown.values())
    level = points_to_level(total)
    return {"ok": True, "errors": [], "missing": [], "breakdown": breakdown,
            "total": total, "level": level, "level_text": LEVEL_TEXT[level],
            "notes": notes}


def recommendation(level: int | None, max_dim_mm: float | None) -> dict:
    """Seviye ve en büyük boyuta göre ACR yönetim önerisi."""
    if level is None:
        return {"action": None, "text": "TI-RADS kategorisi tamamlanmadı."}
    if level in (1, 2):
        return {"action": "none",
                "text": "İİAB veya takip gerekmez (ACR TI-RADS)."}
    fna, follow, schedule = THRESHOLDS[level]
    if max_dim_mm is None or max_dim_mm <= 0:
        return {"action": None,
                "text": (f"Boyut ölçülmeden öneri verilemez (TR{level}: "
                         f"İİAB ≥{fna / 10:g} cm, takip ≥{follow / 10:g} cm).")}
    cm = max_dim_mm / 10
    if max_dim_mm >= fna:
        return {"action": "fna",
                "text": (f"İİAB önerilir: en büyük boyut {cm:.1f} cm ≥ "
                         f"{fna / 10:g} cm (TR{level} eşiği).")}
    if max_dim_mm >= follow:
        return {"action": "follow",
                "text": (f"Takip önerilir: {cm:.1f} cm (TR{level}; İİAB eşiği "
                         f"{fna / 10:g} cm, takip eşiği {follow / 10:g} cm). "
                         f"{schedule}.")}
    return {"action": "none",
            "text": (f"İİAB veya takip gerekmez: {cm:.1f} cm < takip eşiği "
                     f"{follow / 10:g} cm (TR{level}).")}
