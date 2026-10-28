"""Çalıştırma:  python -m unittest discover -s tests -v   (veya pytest)"""

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import measure, tirads


def F(comp, echo, shape, margin, foci):
    return {"composition": comp, "echogenicity": echo, "shape": shape,
            "margin": margin, "foci": foci}


class TestTirads(unittest.TestCase):
    def test_levels(self):
        cases = [
            (F("cystic", "anechoic", "wider", "smooth", ["none"]), 0, 1),
            (F("mixed", "iso_hyper", "wider", "smooth", ["none"]), 2, 2),
            (F("solid", "iso_hyper", "wider", "smooth", ["none"]), 3, 3),
            (F("solid", "hypo", "wider", "smooth", ["none"]), 4, 4),
            (F("solid", "hypo", "wider", "lobulated_irregular", ["none"]), 6, 4),
            (F("solid", "hypo", "wider", "smooth", ["punctate"]), 7, 5),
            (F("solid", "very_hypo", "taller", "ete", ["macro", "rim", "punctate"]),
             2 + 3 + 3 + 3 + 6, 5),
        ]
        for feats, total, level in cases:
            r = tirads.score(feats)
            self.assertEqual(r["total"], total, feats)
            self.assertEqual(r["level"], level, feats)

    def test_foci_additive_and_exclusive(self):
        r = tirads.score(F("solid", "hypo", "wider", "smooth", ["macro", "rim"]))
        self.assertEqual(r["breakdown"]["foci"], 3)
        r = tirads.score(F("solid", "hypo", "wider", "smooth", ["none", "macro"]))
        self.assertFalse(r["ok"])

    def test_spongiform_no_extra_points(self):
        r = tirads.score(F("spongiform", "hypo", "taller", "ete", ["punctate"]))
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["level"], 1)

    def test_missing(self):
        r = tirads.score({"composition": "solid"})
        self.assertIsNone(r["level"])
        self.assertIn("foci", r["missing"])

    def test_one_point_maps_to_tr1(self):
        self.assertEqual(tirads.points_to_level(1), 1)

    def test_recommendation_thresholds(self):
        R = tirads.recommendation
        self.assertEqual(R(2, 40)["action"], "none")
        self.assertEqual(R(3, 25)["action"], "fna")
        self.assertEqual(R(3, 24.9)["action"], "follow")
        self.assertEqual(R(3, 15)["action"], "follow")
        self.assertEqual(R(3, 14.9)["action"], "none")
        self.assertEqual(R(4, 15)["action"], "fna")
        self.assertEqual(R(4, 10)["action"], "follow")
        self.assertEqual(R(4, 9.9)["action"], "none")
        self.assertEqual(R(5, 10)["action"], "fna")
        self.assertEqual(R(5, 5)["action"], "follow")
        self.assertEqual(R(5, 4.9)["action"], "none")
        self.assertIsNone(R(4, None)["action"])


class TestMeasure(unittest.TestCase):
    def ellipse_mask(self, a=60, b=30, size=(300, 400)):
        m = np.zeros(size, np.uint8)
        cv2.ellipse(m, (200, 150), (a, b), 0, 0, 360, 1, -1)
        return m

    def test_metrics_ellipse(self):
        c = measure.mask_to_contour(self.ellipse_mask())
        m = measure.contour_metrics(c)
        self.assertAlmostEqual(m["width_px"], 121, delta=3)
        self.assertAlmostEqual(m["height_px"], 61, delta=3)
        self.assertAlmostEqual(m["feret_px"], 120, delta=3)
        self.assertAlmostEqual(m["area_px"], math.pi * 60 * 30, delta=150)

    def test_taller_than_wide(self):
        c = measure.mask_to_contour(self.ellipse_mask(a=20, b=50))
        m = measure.contour_metrics(c)
        self.assertGreater(m["height_px"], m["width_px"])

    def test_calibration(self):
        self.assertAlmostEqual(
            measure.mm_per_px_from_points([0, 0], [0, 40], 10), 0.25)
        with self.assertRaises(ValueError):
            measure.mm_per_px_from_points([0, 0], [1, 1], 10)

    def test_clean_mask_click_selects_component(self):
        p = np.zeros((200, 300), np.float32)
        cv2.circle(p, (60, 100), 30, 1.0, -1)    # büyük
        cv2.circle(p, (220, 100), 15, 1.0, -1)   # küçük
        big = measure.clean_mask(p, 0.5)
        self.assertEqual(big[100, 60], 1)
        self.assertEqual(big[100, 220], 0)
        small = measure.clean_mask(p, 0.5, click=(220, 100))
        self.assertEqual(small[100, 220], 1)
        self.assertEqual(small[100, 60], 0)
        near = measure.clean_mask(p, 0.5, click=(250, 100))  # dışarıda ama yakın
        self.assertEqual(near[100, 220], 1)


class TestApi(unittest.TestCase):
    """Uçtan uca: demo kaynak + demo model ile dondur, segmente et, kaydet."""

    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient

        from backend.server import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = Path(cls.tmp.name) / "config.json"
        cfg.write_text('{"data_dir": "%s"}' % Path(cls.tmp.name).as_posix())
        cls.app = create_app(cfg, demo=True)
        cls.c = TestClient(cls.app)
        import time
        for _ in range(100):
            if cls.app.state.st.capture.latest()[1] is not None:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.app.state.st.capture.stop()
        cls.tmp.cleanup()

    def test_full_flow(self):
        c = self.c
        s = c.get("/api/status").json()
        self.assertTrue(s["model"]["ready"])
        self.assertEqual(s["capture"]["status"], "açık")

        # ROI + gizlilik maskesi (üst bilgi şeridi)
        c.post("/api/config", json={"roi": [0.15, 0.1, 0.7, 0.88],
                                    "privacy_masks": [[0, 0, 1, 0.08]]})
        fr = c.post("/api/freeze").json()
        png = c.get(f"/api/frame/{fr['frame_id']}.png").content
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(img[: int(img.shape[0] * 0.07)].max()), 0,
                         "gizlilik maskesi uygulanmadı")

        # Demo görüntüde 1 cm = 40 px skala
        cal = c.post("/api/calibration", json={
            "points": [[0, 0], [0, 40]], "distance_mm": 10,
            "save_as": "demo"}).json()
        self.assertAlmostEqual(cal["mm_per_px"], 0.25)
        self.assertIn("demo", cal["presets"])

        # Nodül merkezi civarına tıkla (1280x720 demo karesi)
        r = c.post("/api/segment", json={"frame_id": fr["frame_id"],
                                         "click": [600, 300]}).json()
        self.assertNotIn("error", r)
        self.assertGreater(len(r["contour"]), 10)
        w, h = r["metrics_mm"]["width_mm"], r["metrics_mm"]["height_mm"]
        # Sentetik nodül ~46 x 35 mm
        self.assertAlmostEqual(w, 46, delta=6)
        self.assertAlmostEqual(h, 35, delta=6)

        feats = F("solid", "hypo", "wider", "smooth", ["none"])
        sc = c.post("/api/score", json={"features": feats, "max_mm": w}).json()
        self.assertEqual(sc["score"]["level"], 4)
        self.assertEqual(sc["recommendation"]["action"], "fna")

        body = {"views": {"transverse": {"frame_id": fr["frame_id"], "lines": []}},
                "features": feats,
                "measurements": {"width_mm": w, "ap_mm": h, "length_mm": None},
                "radiologist": "Test", "approved": False}
        self.assertEqual(c.post("/api/cases", json=body).status_code, 400)
        body["approved"] = True
        # AI konturu onaylanmadan kayıt yapılamaz
        r = c.post("/api/cases", json=body)
        self.assertEqual(r.status_code, 400)
        self.assertIn("onaylanmadı", r.json()["error"])
        self.assertTrue(c.post("/api/contour/confirm",
                               json={"frame_id": fr["frame_id"]}).json()["confirmed"])
        sv = c.post("/api/cases", json=body).json()
        self.assertIn("case_id", sv)
        self.assertIn("Dice", sv["report"])
        self.assertIn("TR4", sv["report"])
        d = Path(sv["path"])
        for f in ("transverse.png", "transverse_mask.png", "transverse_mask_ai.png",
                  "transverse_overlay.jpg", "case.json", "report.txt"):
            self.assertTrue((d / f).exists(), f)
        lst = c.get("/api/cases").json()["cases"]
        self.assertEqual(lst[0]["case_id"], sv["case_id"])
        rep = c.get(f"/api/cases/{sv['case_id']}/report")
        self.assertEqual(rep.status_code, 200)
        self.assertIn(c.get("/api/cases/..%2Fx/report").status_code, (400, 404))

    def test_manual_contour_and_upload(self):
        c = self.c
        img = np.zeros((200, 300, 3), np.uint8)
        ok, enc = cv2.imencode(".png", img)
        fr = c.post("/api/upload", content=enc.tobytes()).json()
        self.assertEqual(fr["width"], 300)
        r = c.post("/api/contour", json={
            "frame_id": fr["frame_id"],
            "contour": [[50, 50], [150, 50], [150, 120], [50, 120]]}).json()
        self.assertAlmostEqual(r["metrics_px"]["width_px"], 101, delta=1)
        self.assertAlmostEqual(r["metrics_px"]["height_px"], 71, delta=1)
        self.assertEqual(c.post("/api/upload", content=b"xx").status_code, 400)


    def test_ai_contour_review_flow(self):
        """AI önerisi → düzelt → onay; elle çizim → AI bırakılır."""
        c = self.c
        fr = c.post("/api/freeze").json()
        fid = fr["frame_id"]
        seg = c.post("/api/segment", json={"frame_id": fid, "click": [600, 300]}).json()
        ai_contour = seg["contour"]
        # Düzeltme (genişlet) → hâlâ onaysız, AI uyumu < 1
        cnt = np.array(ai_contour, float)
        ctr = cnt.mean(0)
        bigger = ((cnt - ctr) * 1.1 + ctr).round().astype(int).tolist()
        r = c.post("/api/contour", json={"frame_id": fid, "contour": bigger,
                                         "source": "edit", "confirmed": False}).json()
        self.assertFalse(r["confirmed"])
        self.assertLess(r["ai_dice"], 0.99)
        self.assertGreater(r["metrics_px"]["width_px"],
                           seg["metrics_px"]["width_px"])
        # Geri al: AI önerisine dönüş → uyum ~1
        r = c.post("/api/contour", json={"frame_id": fid, "contour": ai_contour,
                                         "source": "ai", "confirmed": False}).json()
        self.assertGreater(r["ai_dice"], 0.97)
        # Kendini kesen (kısmi düzeltme) kontur temizlenir
        loop = ai_contour[: len(ai_contour) // 2] + [[620, 320], [560, 280]] + \
            ai_contour[len(ai_contour) // 2:]
        r = c.post("/api/contour", json={"frame_id": fid, "contour": loop,
                                         "source": "edit", "confirmed": True}).json()
        self.assertTrue(r["confirmed"])
        self.assertGreaterEqual(len(r["contour"]), 3)
        # Sıfırdan elle çizim: AI önerisi bırakılır, onaylı sayılır
        sq = [[500, 250], [700, 250], [700, 380], [500, 380]]
        r = c.post("/api/contour", json={"frame_id": fid, "contour": sq,
                                         "source": "manual", "confirmed": True}).json()
        self.assertLess(r["ai_dice"], 0.95)   # AI önerisi saklanır
        st = self.app.state.st.get_frame(fid)
        self.assertIsNotNone(st["ai_mask"])
        self.assertTrue(st["ai"]["redrawn"])
        # Hiç AI kullanılmamış karede elle çizim = manuel
        fid2 = c.post("/api/freeze").json()["frame_id"]
        r = c.post("/api/contour", json={"frame_id": fid2, "contour": sq,
                                         "source": "manual", "confirmed": True}).json()
        self.assertNotIn("ai_dice", r)
        self.assertEqual(self.app.state.st.get_frame(fid2)["ai"]["backend"], "manuel")
        # Onay kaldırılırsa (yeni AI önerisi) kayıt engellenir
        c.post("/api/segment", json={"frame_id": fid, "click": [600, 300]})
        self.assertFalse(self.app.state.st.get_frame(fid)["confirmed"])

if __name__ == "__main__":
    unittest.main()
