"""Biyopsi iğnesi algılama, hedef takibi, rehber geometrisi ve API."""

import math
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import needle as N


def speckle(seed, h=600, w=820):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.rayleigh(1.0, (h, w)) * 70, (0, 0), 1.2)


def with_needle(g, e, t, val=235, thick=3):
    cv2.line(g, tuple(map(int, e)), tuple(map(int, t)), val, thick)
    for k in (7, 14):
        cv2.line(g, (int(e[0]), int(e[1]) + k), (int(t[0]), int(t[1]) + k), 120 - 4 * k, 1)
    return g


def u8(g):
    return np.clip(g, 0, 255).astype(np.uint8)


def needle_pts(ang, L, e=(800, 30)):
    return e, (e[0] - L * math.cos(math.radians(ang)), e[1] + L * math.sin(math.radians(ang)))


class TestDetect(unittest.TestCase):
    def test_angles_and_tip(self):
        for i, ang in enumerate((15, 30, 45, 60)):
            e, t = needle_pts(ang, 380)
            d = N.detect(u8(with_needle(speckle(i), e, t)))
            self.assertIsNotNone(d, ang)
            self.assertAlmostEqual(d["angle_deg"], ang, delta=2.0)
            self.assertLess(math.dist(d["tip"], t), 10)
            self.assertLess(math.dist(d["entry"], e), 10)

    def test_no_false_positive_on_tissue(self):
        fp = 0
        for i in range(20):
            g = speckle(100 + i)
            cv2.line(g, (0, 220 + i), (820, 232 + i), 205, 4)               # yatay fasya
            cv2.ellipse(g, (400, 380), (90, 60), 0, 0, 360, 30, -1)          # hipoekoik nodül
            fp += N.detect(u8(g)) is not None
        self.assertEqual(fp, 0)

    def test_horizontal_fascia_does_not_win(self):
        g = speckle(7)
        cv2.line(g, (0, 250), (820, 262), 220, 5)
        e, t = needle_pts(35, 300)
        d = N.detect(u8(with_needle(g, e, t, val=200, thick=2)))
        self.assertAlmostEqual(d["angle_deg"], 35, delta=3)

    def test_plan_prior_rejects_parallel_fascia(self):
        ok = 0
        for i in range(10):
            rng = np.random.default_rng(50 + i)
            ang, L = rng.uniform(20, 45), rng.uniform(200, 380)
            g = speckle(200 + i)
            cv2.line(g, (0, 300), (820, 300 - int(820 * math.tan(math.radians(12)))), 220, 5)
            e, t = needle_pts(ang, L)
            img = u8(with_needle(g, e, t, val=200, thick=2))
            tgt = (e[0] - (L + 60) * math.cos(math.radians(ang)),
                   e[1] + (L + 60) * math.sin(math.radians(ang)))
            d = N.detect(img, plan={"entry": list(e), "target": list(tgt)})
            ok += bool(d and abs(d["angle_deg"] - ang) < 4 and math.dist(d["tip"], t) < 15)
        self.assertGreaterEqual(ok, 9)

    def test_roi_coordinates(self):
        e, t = needle_pts(30, 300)
        g = np.zeros((720, 1280))
        g[60:660, 200:1020] = speckle(3)
        g = with_needle(g, (e[0] + 200, e[1] + 60), (t[0] + 200, t[1] + 60))
        d = N.detect(u8(g), roi_px=(200, 60, 1020, 660))
        self.assertLess(math.dist(d["tip"], (t[0] + 200, t[1] + 60)), 12)


class TestTrackerMotion(unittest.TestCase):
    def test_advancing_needle_beats_static_parallel_fascia_without_plan(self):
        """Plan yokken: ilerleyen iğne, aynı yönde ve daha parlak sabit bir
        fasyaya rağmen bulunmalı (hareket + kenar ön bilgisi)."""
        import threading

        from backend.biopsy import BiopsyTracker

        class FakeCap:
            def __init__(self):
                self.fid, self.frame, self.lock = 0, None, threading.Lock()

            def latest(self):
                with self.lock:
                    return self.fid, self.frame
        cap = FakeCap()
        base = speckle(42)
        cv2.line(base, (0, 300), (820, 300 - int(820 * math.tan(math.radians(12)))), 220, 5)
        ang, ok, tot = 30, 0, 0
        tr = BiopsyTracker(cap, lambda: {"roi": None, "calibration": {"mm_per_px": 0.25}})
        tr.start()
        try:
            for k in range(36):
                e, t = needle_pts(ang, 60 + k * 7)
                g = base + np.random.default_rng(k).normal(0, 6, base.shape)
                img = cv2.cvtColor(u8(with_needle(g, e, t, val=200, thick=2)), cv2.COLOR_GRAY2BGR)
                with cap.lock:
                    cap.frame, cap.fid = img, cap.fid + 1
                time.sleep(0.06)
                n = tr.snapshot().get("needle")
                if k >= 14:
                    tot += 1
                    ok += bool(n and n.get("found") and abs(n["angle_deg"] - ang) < 4)
        finally:
            tr.stop()
        self.assertGreaterEqual(ok / tot, 0.85)


class TestSmootherTargetGeometry(unittest.TestCase):
    def test_smoother_holds_and_drops(self):
        sm = N.NeedleSmoother(hold_s=0.5)
        det = {"entry": [800, 30], "tip": [600, 150], "angle_deg": 31, "conf": 0.9}
        self.assertTrue(sm.update(det, now=0.0)["found"])
        self.assertTrue(sm.update(None, now=0.3)["found"])     # kısa kayıp: korunur
        self.assertFalse(sm.update(None, now=1.0)["found"])    # uzun kayıp: bulunamadı
        self.assertIsNone(sm.update(None, now=3.0))             # sıfırlanır

    def test_smoother_rejects_single_frame_jump(self):
        sm = N.NeedleSmoother()
        a = {"entry": [800, 30], "tip": [600, 150], "angle_deg": 31, "conf": 0.9}
        b = {"entry": [800, 300], "tip": [500, 330], "angle_deg": 6, "conf": 0.5}
        sm.update(a, now=0)
        out = sm.update(b, now=0.05)
        self.assertLess(math.dist(out["tip"], a["tip"]), 1)    # tek karelik sıçrama yok sayılır

    def test_target_tracker_follows_shift(self):
        g = u8(speckle(9))
        cv2.ellipse(g, (400, 300), (60, 40), 0, 0, 360, 30, -1)
        tr = N.TargetTracker()
        tr.set(g, (400, 300))
        M = np.float32([[1, 0, 12], [0, 1, -8]])
        shifted = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]))
        for _ in range(4):
            pos = tr.update(shifted)
        self.assertLess(math.dist(pos, (412, 292)), 3)
        self.assertFalse(tr.lost)

    def test_guidance_states(self):
        e, tgt = [800, 30], [400, 270]
        mmpp = 0.25   # yarıçap 5 mm = 20 px
        u = np.subtract(tgt, e) / np.linalg.norm(np.subtract(tgt, e))
        nd = lambda tip: {"entry": e, "tip": list(tip), "angle_deg": N.line_angle_deg(e, tip), "found": True}  # noqa: E731
        g = N.guidance(nd(np.add(e, u * 200)), tgt, e, 5, mmpp)
        self.assertEqual(g["status"], "on_track")
        self.assertAlmostEqual(g["miss"], 0, delta=0.1)
        self.assertAlmostEqual(g["remaining"], (np.linalg.norm(np.subtract(tgt, e)) - 200) * mmpp, delta=0.2)
        g = N.guidance(nd(np.add(tgt, u * 5)), tgt, e, 5, mmpp)
        self.assertEqual(g["status"], "in_target")
        g = N.guidance(nd(np.add(tgt, u * 80)), tgt, e, 5, mmpp)
        self.assertEqual(g["status"], "passed")
        # 10° saparak: hedefi ıskalayan hat
        a = math.atan2(u[1], u[0]) + math.radians(10)
        g = N.guidance(nd((e[0] + 200 * math.cos(a), e[1] + 200 * math.sin(a))), tgt, e, 5, mmpp)
        self.assertEqual(g["status"], "off_track")
        self.assertGreater(g["miss"], 5)
        self.assertEqual(g["target_side"], "shallower")
        self.assertAlmostEqual(abs(g["angle_diff_deg"]), 10, delta=0.5)
        g = N.guidance(None, tgt, e, 5, mmpp)
        self.assertEqual(g["status"], "searching")


class TestBiopsyApi(unittest.TestCase):
    def test_live_tracking_on_demo(self):
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
                self.assertTrue(c.post("/api/biopsy/start").json()["active"])
                time.sleep(0.3)
                self.assertEqual(c.post("/api/biopsy/plan", json={"target": [1, 2, 3]}).status_code, 400)
                p = c.post("/api/biopsy/plan", json={"target": [623, 355], "entry": [1037, 100],
                                                     "radius_mm": 5}).json()["plan"]
                self.assertEqual(p["radius_mm"], 5)
                dists, statuses = [], set()
                t0 = time.time()
                while time.time() - t0 < 8.5:
                    s = c.get("/api/biopsy/state").json()
                    g = s.get("guide") or {}
                    if g.get("tip_to_target") is not None:
                        dists.append(g["tip_to_target"])
                        statuses.add(g["status"])
                    time.sleep(0.25)
                self.assertGreater(len(dists), 10, "iğne algılanmadı")
                self.assertGreater(max(dists[:6]), 40)      # başta uzak
                self.assertLess(min(dists), 8)              # nodüle ulaştı
                self.assertIn("in_target", statuses)
                r = c.post("/api/biopsy/snapshot", json={"study_no": "B-1"}).json()
                d = Path(r["path"])
                for f in ("frame.png", "overlay.jpg", "state.json"):
                    self.assertTrue((d / f).exists())
                c.post("/api/biopsy/plan", json={"clear": True})
                time.sleep(0.4)   # sonraki işlenen karede yansır
                self.assertIsNone(c.get("/api/biopsy/state").json()["plan"]["target"])
                self.assertFalse(c.post("/api/biopsy/stop").json()["active"])
                self.assertEqual(c.post("/api/biopsy/snapshot", json={}).status_code, 409)
            finally:
                app.state.st.biopsy.stop()
                app.state.st.capture.stop()


if __name__ == "__main__":
    unittest.main()
