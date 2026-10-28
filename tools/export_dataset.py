"""Etiketleri eğitim formatına aktarır (arayüzdeki düğmenin komut satırı hali).

    python -m tools.export_dataset                       # data/dataset -> data/export
    python -m tools.export_dataset --nnunet 501          # + nnU-Net v2 klasörü
    python -m tools.export_dataset --new-split           # hasta bölmesini baştan yap

Varsayılan olarak önceki hasta bölmesi (data/export/splits.json) korunur;
yeni hastalar eğitime eklenir, test seti değişmez.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from backend import dataset
from backend.config import Config


def main():
    cfg = Config()
    base = cfg.data_dir()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=base / "dataset")
    ap.add_argument("--out", type=Path, default=base / "export")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nnunet", type=int, default=None, metavar="ID")
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--new-split", action="store_true")
    a = ap.parse_args()

    keep = None
    old = a.out / "splits.json"
    if old.exists() and not a.new_split:
        keep = Path(tempfile.mkdtemp()) / "splits.json"
        shutil.copy(old, keep)
    s = dataset.export(a.src, a.out, val=a.val, test=a.test, seed=a.seed,
                       include_negatives=not a.no_negatives,
                       nnunet_id=a.nnunet, split_file=keep)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
