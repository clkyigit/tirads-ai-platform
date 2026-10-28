"""Çok sınıflı anatomi etiketleme/dışa aktarma, canlı anatomi ve biyopsi
oturum ölçümleri."""

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import anatomy as A
from backend import dataset
from backend.segmentation import Segmenter
from tools import train_unetpp as T


def neck_image(h=480, w=640):
    rng = np.random.default_rng(3)
    g = (rng.rayleigh(1.0, (h, w)) * 80).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    masks = {}
    for cid, (c, ax) in {2: ((260, 240), (150, 80)), 3: ((470, 250), (40, 40)),
                         5: ((90, 200), (50, 45)), 1: ((260, 240), (40, 30))}.items():
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, c, ax, 0, 0, 360, 1, -1)
        masks[cid] = m
    return img, masks


class TestAnatomyCore(unittest.TestCase):
    def test_label_map_order_and_contours(self):
        img, masks = neck_image()
        lab = A.label_map(img.shape, masks)
        self.assertEqual(lab[240, 260], 1)          # nodül tiroidin üstünde
        self.assertEqual(lab[240, 160], 2)
        self.assertEqual(lab[250, 470], 3)
        polys = A.contours_from_labels(lab, offset=(10, 20), scale=0.5)
        self.assertEqual(set(polys), {"1", "2", "3", "5"})
        xs = [p[0] for p in polys["3"][0]]
        self.assertAlmostEqual(np.mean(xs), 470 / 0.5 + 10, delta=15)  # ölçek + kaydırma

    def test_class_dice(self):
        a = np.zeros((10, 10), int); a[2:6, 2:6] = 1; a[7:9, 7:9] = 3
        b = a.copy(); b[2:6, 2:4] = 0
        d = T.class_dice(b, a, [1, 2, 3])
        self.assertAlmostEqual(d[1], 2 * 8 / (8 + 16))
        self.assertEqual(d[3], 1.0)
        self.assertNotIn(2, d)

    def test_demo_predict_labels(self):
        seg = Segmenter({"backend": "demo", "threshold": 0.5}, Path("."))
        img, _ = neck_image()
        lab = seg.predict_labels(img, work_width=320)
        self.assertEqual(lab.shape, (240, 320))
        self.assertFalse(seg.multiclass)


class TestAnatomyDataset(unittest.TestCase):
    def test_save_and_export_labels(self):
        img, masks = neck_image()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "ds"
            meta = {"patient_id": "P1", "nodule_id": "N1", "view": "transverse",
                    "annotator": "A", "roi_px": [20, 20, 620, 460]}
            dataset.save_annotation(root, image=img, mask=masks[1], ai_mask=None,
                                    meta=dict(meta), structs={2: masks[2], 3: masks[3]})
            # yalnızca anatomi (nodül yok) → negatif örnek olarak kabul
            d2 = dataset.save_annotation(root, image=img, mask=None, ai_mask=None,
                                         meta={**meta, "patient_id": "P2"},
                                         structs={2: masks[2]})
            m2 = json.loads((d2 / "meta.json").read_text())
            self.assertTrue(m2["negative"])
            self.assertEqual(m2["structures"], ["thyroid"])
            st = dataset.stats(root)
            self.assertEqual(st["with_anatomy"], 2)
            out = Path(t) / "export"
            s = dataset.export(root, out, val=0.0, test=0.0)
            self.assertEqual(s["with_labels"], 2)
            with (out / "index.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(r["has_labels"] == "1" for r in rows))
            lab = cv2.imread(str(out / "labels" / f"{rows[0]['id']}.png"), 0)
            self.assertEqual(lab.shape, (440, 600))   # ROI'ye kırpıldı
            self.assertTrue(set(np.unique(lab)) <= {0, 1, 2, 3})
            # Eğitim veri seti anatomi görevinde etiket haritası döndürür
            rows_t = T.load_index(out)
            ds = T.SegDataset(out, rows_t, (256, 192), 3, "imagenet", train=True,
                              task="anatomy")
            x, y = ds[0]
            self.assertEqual(y.dtype, np.int64)
            self.assertEqual(y.shape, (192, 256))


class TestApiAnatomySession(unittest.TestCase):
    def test_struct_contour_annotation_session_anatomy(self):
        from starlette.testclient import TestClient

        from backend.server import create_app
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text('{"data_dir": "%s", "roi": [0.18, 0.12, 0.64, 0.83],'
                           ' "calibration": {"mm_per_px": 0.25}}' % Path(tmp).as_posix())
            app = create_app(cfg, demo=True)
            c = TestClient(app)
            try:
                time.sleep(0.8)
                self.assertEqual(len(c.get("/api/status").json()["anatomy"]), 5)
                fid = c.post("/api/freeze").json()["frame_id"]
                sq = [[300, 150], [900, 150], [900, 600], [300, 600]]
                r = c.post("/api/contour", json={"frame_id": fid, "contour": sq, "cls": 2}).json()
                self.assertEqual(r["cls"], 2)
                self.assertNotIn("metrics_px", r)     # ölçüm yalnızca nodül için
                self.assertEqual(c.post("/api/contour", json={"frame_id": fid, "contour": sq,
                                                              "cls": 9}).status_code, 400)
                nod = [[560, 300], [680, 300], [680, 400], [560, 400]]
                c.post("/api/contour", json={"frame_id": fid, "contour": nod,
                                             "source": "manual", "confirmed": True})
                a = c.post("/api/annotations", json={"frame_id": fid, "patient_id": "P9",
                                                     "view": "transverse", "annotator": "A"}).json()
                d = Path(tmp) / "dataset" / a["path"]
                lab = cv2.imread(str(d / "labels.png"), 0)
                self.assertEqual(lab[350, 620], 1)
                self.assertEqual(lab[200, 400], 2)

                # Biyopsi oturumu + canlı anatomi (demo segmentasyon)
                self.assertEqual(c.post("/api/biopsy/session", json={"action": "start"}).status_code, 409)
                c.post("/api/biopsy/start")
                c.post("/api/biopsy/plan", json={"target": [623, 355], "entry": [1037, 100]})
                self.assertTrue(c.post("/api/biopsy/anatomy", json={"on": True}).json()["on"])
                s0 = c.post("/api/biopsy/session", json={"action": "start"}).json()
                self.assertTrue(s0["active"])
                time.sleep(8.0)
                st = c.get("/api/biopsy/state").json()
                self.assertTrue(st["anatomy_on"])
                self.assertIn("1", st["anatomy"]["polys"])
                self.assertGreater(st["session"]["frames"], 20)
                r = c.post("/api/biopsy/session", json={"action": "stop", "study_no": "B7"}).json()
                self.assertFalse(r["active"])
                self.assertGreater(r["tip_in_view_pct"], 50)
                self.assertIsNotNone(r["time_to_target_s"])
                self.assertLess(r["min_tip_to_target"], 8)
                sd = Path(r["path"])
                summ = json.loads((sd / "summary.json").read_text())
                self.assertEqual(summ["study_no"], "B7")
                self.assertTrue(summ["tip_path"])
                self.assertTrue((sd / "last.jpg").exists())
                c.post("/api/biopsy/anatomy", json={"on": False})
                time.sleep(0.3)
                self.assertIsNone(c.get("/api/biopsy/state").json()["anatomy"])
            finally:
                app.state.st.biopsy.stop()
                app.state.st.capture.stop()


if __name__ == "__main__":
    unittest.main()
