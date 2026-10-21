import csv
import json
import random
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import dataset
from backend.segmentation import Segmenter
from tools import train_unetpp as T


def fake_image(seed=0, w=640, h=480):
    rng = np.random.default_rng(seed)
    img = (rng.rayleigh(1.0, (h, w)) * 90).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (320 + seed * 5, 240), (60, 40), 0, 0, 360, 1, -1)
    img[m > 0] //= 3
    return img, m


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "dataset"

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, pid, nid="N1", view="transverse", negative=False, seed=0, ai=False):
        img, m = fake_image(seed)
        meta = {"patient_id": pid, "nodule_id": nid, "view": view,
                "negative": negative, "annotator": "Dr. A",
                "roi_px": [40, 30, 600, 460], "mm_per_px": 0.1,
                "tirads": {"level": 4, "total": 4}}
        ai_mask = np.roll(m, 4, axis=1) if ai else None
        time.sleep(0.002)
        return dataset.save_annotation(self.root, image=img,
                                       mask=None if negative else m,
                                       ai_mask=ai_mask, meta=meta)

    def test_save_validation(self):
        img, m = fake_image()
        with self.assertRaises(ValueError):  # hasta kodu yok
            dataset.save_annotation(self.root, image=img, mask=m, ai_mask=None,
                                    meta={"patient_id": "", "view": "transverse"})
        with self.assertRaises(ValueError):  # maske yok, negatif değil
            dataset.save_annotation(self.root, image=img, mask=None, ai_mask=None,
                                    meta={"patient_id": "P1", "view": "transverse"})
        d = self.add("P 1/../x")  # zararlı karakterler temizlenir
        self.assertTrue(str(d).startswith(str(self.root)))
        self.assertNotIn("..", d.relative_to(self.root).parts)

    def test_stats_ai_dice_and_delete(self):
        self.add("P1", ai=True)
        self.add("P1", view="longitudinal")
        self.add("P2", negative=True)
        s = dataset.stats(self.root)
        self.assertEqual((s["images"], s["patients"], s["nodules"], s["negatives"]),
                         (3, 2, 2, 1))
        self.assertEqual(s["with_ai"], 1)
        self.assertGreater(s["ai_vs_final_dice_mean"], 0.8)
        rel = dataset.recent(self.root)[0]["path"]
        self.assertTrue(dataset.mark_deleted(self.root, rel))
        self.assertEqual(dataset.stats(self.root)["images"], 2)
        with self.assertRaises(ValueError):
            dataset.mark_deleted(self.root, "../../etc")

    def test_export_patient_level_split_and_crop(self):
        for i in range(10):
            self.add(f"P{i}", seed=i % 3)
            self.add(f"P{i}", view="longitudinal", seed=i % 3)
        out = Path(self.tmp.name) / "export"
        s = dataset.export(self.root, out, val=0.2, test=0.2, seed=1, nnunet_id=501)
        self.assertEqual(s["images"], 20)
        with (out / "index.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_p = {}
        for r in rows:
            by_p.setdefault(r["patient_id"], set()).add(r["split"])
        self.assertTrue(all(len(v) == 1 for v in by_p.values()), "hasta sızıntısı")
        self.assertEqual({r["split"] for r in rows}, {"train", "val", "test"})
        img = cv2.imread(str(out / "images" / f"{rows[0]['id']}.png"))
        self.assertEqual(img.shape[:2], (430, 560))  # ROI'ye kırpıldı
        msk = cv2.imread(str(out / "masks" / f"{rows[0]['id']}.png"), 0)
        self.assertEqual(set(np.unique(msk)), {0, 1})
        nn = Path(s["nnunet"])
        self.assertTrue((nn / "dataset.json").exists())
        self.assertEqual(len(list((nn / "imagesTr").glob("*_0000.png"))),
                         s["by_split"]["train"] + s["by_split"]["val"])

        # Yeni hastalar eklenince eski bölme korunur, yeniler eğitime girer
        prev = json.loads((out / "splits.json").read_text())
        keep = Path(self.tmp.name) / "prev.json"
        keep.write_text(json.dumps(prev))
        self.add("P99")
        dataset.export(self.root, out, split_file=keep)
        new = json.loads((out / "splits.json").read_text())
        self.assertEqual(new["test"], prev["test"])
        self.assertIn("P99", new["train"])

    def test_training_input_matches_app_preprocessing(self):
        img, _ = fake_image()
        for norm in ("imagenet", "zero_one", "zscore"):
            for cin in (1, 3):
                seg = Segmenter({"backend": "none", "input_size": [256, 192],
                                 "in_channels": cin, "normalize": norm}, Path("."))
                a = seg._preprocess(img)[0]
                b = T.to_input(img, (256, 192), cin, norm)
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_augment_keeps_alignment(self):
        img, m = fake_image()
        rng = random.Random(3)
        for _ in range(20):
            ai, am = T.augment(img, m, rng)
            self.assertEqual(ai.shape, img.shape)
            self.assertEqual(am.dtype, np.uint8)
            if am.any():  # maske altındaki görüntü hâlâ daha koyu olmalı
                self.assertLess(ai[am > 0].mean(), ai[am == 0].mean())

    def test_dataset_item_and_metrics(self):
        for i in range(4):
            self.add(f"P{i}")
        out = Path(self.tmp.name) / "export"
        dataset.export(self.root, out, val=0.25, test=0.25, seed=0)
        rows = T.load_index(out)
        ds = T.SegDataset(out, rows, (256, 192), 3, "imagenet", train=True)
        x, y = ds[0]
        self.assertEqual(x.shape, (3, 192, 256))
        self.assertEqual(y.shape, (1, 192, 256))
        self.assertGreater(y.sum(), 100)

        gt = cv2.imread(str(out / "masks" / f"{rows[0]['id']}.png"), 0)
        perfect = T.eval_case(gt.astype(np.float32), gt, 0.1, 0.5)
        # clean_mask'teki morfolojik açma/kapama birkaç pikseli değiştirebilir
        self.assertAlmostEqual(perfect["dice"], 1.0, delta=2e-3)
        self.assertLess(perfect["size_err_mm"], 0.3)
        shifted = T.eval_case(np.roll(gt, 10, 1).astype(np.float32), gt, 0.1, 0.5)
        self.assertLess(shifted["dice"], 1.0)
        neg = T.eval_case(np.zeros_like(gt, np.float32), np.zeros_like(gt), 0.1, 0.5)
        self.assertEqual(neg["false_positive"], 0.0)
        s = T.summarize([perfect, shifted, neg])
        self.assertEqual(s["n_dice"], 2)


class TestAnnotationApi(unittest.TestCase):
    def test_annotation_flow(self):
        from starlette.testclient import TestClient

        from backend.server import create_app
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text('{"data_dir": "%s"}' % Path(tmp).as_posix())
            app = create_app(cfg, demo=True)
            c = TestClient(app)
            try:
                for _ in range(100):
                    if app.state.st.capture.latest()[1] is not None:
                        break
                    time.sleep(0.05)
                c.post("/api/config", json={"roi": [0.15, 0.1, 0.7, 0.88]})
                fr = c.post("/api/freeze").json()
                base = {"frame_id": fr["frame_id"], "patient_id": "P0001",
                        "nodule_id": "N1", "view": "transverse",
                        "annotator": "Dr. A"}
                r = c.post("/api/annotations", json=base)
                self.assertEqual(r.status_code, 400)  # maske yok
                seg = c.post("/api/segment", json={"frame_id": fr["frame_id"],
                                                   "click": [600, 300]}).json()
                # radyolog düzeltmesi: konturu biraz büyüt
                cnt = np.array(seg["contour"], float)
                ctr = cnt.mean(0)
                edited = ((cnt - ctr) * 1.08 + ctr).round().astype(int).tolist()
                c.post("/api/contour", json={"frame_id": fr["frame_id"],
                                             "contour": edited})
                r = c.post("/api/annotations", json={
                    **base, "features": {"composition": "solid"}}).json()
                self.assertTrue(r["ok"], r)
                d = Path(tmp) / "dataset" / r["path"]
                meta = json.loads((d / "meta.json").read_text())
                self.assertTrue((d / "mask_ai.png").exists())
                self.assertTrue(meta["ai"]["edited"])
                self.assertTrue(0.7 < meta["ai_vs_final_dice"] < 0.99)
                self.assertEqual(len(meta["roi_px"]), 4)

                fr2 = c.post("/api/freeze").json()
                r = c.post("/api/annotations", json={
                    **base, "frame_id": fr2["frame_id"], "negative": True}).json()
                self.assertTrue(r["ok"])
                st = c.get("/api/dataset").json()["stats"]
                self.assertEqual((st["images"], st["negatives"]), (2, 1))
                ex = c.post("/api/dataset/export", json={}).json()
                self.assertEqual(ex["images"], 2)
            finally:
                app.state.st.capture.stop()


if __name__ == "__main__":
    unittest.main()
