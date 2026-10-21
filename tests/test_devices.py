"""Capture card algılama, format seçimi ve tak-çıkar davranışı (donanımsız,
sahte VideoCapture ile)."""

import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import devices
from backend.capture import CaptureManager


class FakeWorld:
    """Takılı cihazları ve desteklenen modları taklit eder."""

    def __init__(self):
        self.lock = threading.Lock()
        self.plugged = {}      # index -> {"name", "modes": {fourcc: (fps, w, h)}}
        self.signal = True
        self.opens = 0

    def devices(self):
        with self.lock:
            return [{"index": i, "name": d["name"],
                     "score": devices.score_name(d["name"])}
                    for i, d in sorted(self.plugged.items())]


class FakeCap:
    def __init__(self, world, index, api=None):
        self.w, self.index = world, index
        self.props = {}
        self.ok = index in world.plugged
        world.opens += 1

    def isOpened(self):
        return self.ok

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FOURCC:
            return self._mode()[3]
        return self.props.get(prop, 0)

    def _mode(self):
        d = self.w.plugged.get(self.index)
        req = devices.fourcc_str(self.props.get(cv2.CAP_PROP_FOURCC, 0)) or "YUY2"
        modes = d["modes"]
        fc = req if req in modes else next(iter(modes))
        fps, w, h = modes[fc]
        return fps, w, h, cv2.VideoWriter_fourcc(*fc)

    def read(self):
        if self.index not in self.w.plugged:
            return False, None
        fps, w, h, _ = self._mode()
        time.sleep(1.0 / fps)
        if self.w.signal:
            f = np.random.default_rng().integers(0, 255, (h // 8, w // 8, 3), np.uint8)
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            f = np.full((h, w, 3), (120, 0, 0), np.uint8)  # mavi "sinyal yok" ekranı
        return True, f

    def release(self):
        self.ok = False


class TestScoring(unittest.TestCase):
    def test_names(self):
        cap = ["USB3.0 Capture", "Cam Link 4K", "AVerMedia Live Gamer Portable",
               "USB Video", "UGREEN 25854", "Magewell USB Capture HDMI"]
        cam = ["Integrated Camera", "HD Webcam", "FaceTime HD Camera",
               "Integrated_Webcam_HD", "OBS Virtual Camera", "Dummy video device (0x0000)"]
        for n in cap:
            self.assertGreaterEqual(devices.score_name(n), 10, n)
        for n in cam:
            self.assertLess(devices.score_name(n), -5, n)
        self.assertEqual(devices.score_name("Chicony USB2.0 Camera"), -3)

    def test_choose(self):
        devs = [{"index": 0, "name": "Integrated Camera"},
                {"index": 2, "name": "USB3.0 Capture"}]
        for d in devs:
            d["score"] = devices.score_name(d["name"])
        self.assertEqual(devices.choose_device(devs)["index"], 2)
        # yalnızca dahili kamera varsa seçilmez
        self.assertIsNone(devices.choose_device(devs[:1]))
        # yok sayılan capture card
        self.assertIsNone(devices.choose_device(devs, ignore=["USB3.0 Capture"]))
        # tercih edilen her şeyin önündedir (kamera olsa bile)
        self.assertEqual(devices.choose_device(devs, preferred="integrated camera")["index"], 0)

    def test_linux_sysfs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for n, name, idx in [(0, "Integrated Camera", 0), (1, "Integrated Camera", 1),
                                 (2, "USB3.0 Capture", 0), (3, "USB3.0 Capture", 1)]:
                d = root / f"video{n}"
                d.mkdir()
                (d / "name").write_text(name + "\n")
                (d / "index").write_text(f"{idx}\n")
            devs = devices._linux_devices(root)
            self.assertEqual([d["index"] for d in devs], [0, 2])  # metadata düğümleri elendi

    def test_no_signal(self):
        self.assertTrue(devices.is_no_signal(np.zeros((720, 1280, 3), np.uint8)))
        self.assertTrue(devices.is_no_signal(np.full((720, 1280, 3), (200, 30, 30), np.uint8)))
        us = (np.random.default_rng(0).rayleigh(1, (720, 1280)) * 60).clip(0, 255).astype(np.uint8)
        self.assertFalse(devices.is_no_signal(cv2.cvtColor(us, cv2.COLOR_GRAY2BGR)))


class TestNegotiate(unittest.TestCase):
    def setUp(self):
        self.w = FakeWorld()
        self.op = lambda i, api=None: FakeCap(self.w, i, api)

    def test_prefers_uncompressed_when_fast(self):
        self.w.plugged[0] = {"name": "Cam Link", "modes": {
            "YUY2": (60, 1920, 1080), "MJPG": (60, 1920, 1080)}}
        cap, info = devices.negotiate(0, {"fourcc": "auto"}, opener=self.op, apis=[0])
        self.assertEqual(info["fourcc"], "YUY2")
        self.assertFalse(info["compressed"])
        cap.release()

    def test_usb2_card_falls_back(self):
        # MS2109 tipi: YUY2 1080p 5 fps, 720p 10 fps -> MJPG 30 fps seçilmeli
        self.w.plugged[0] = {"name": "USB Video", "modes": {
            "YUY2": (5, 1920, 1080), "MJPG": (30, 1920, 1080)}}
        cap, info = devices.negotiate(0, {"fourcc": "auto"}, opener=self.op, apis=[0])
        self.assertEqual(info["fourcc"], "MJPG")
        self.assertTrue(info["compressed"])
        self.assertGreaterEqual(info["fps"], 15)
        cap.release()

    def test_cached_mode_is_tried_first(self):
        self.w.plugged[0] = {"name": "Cam Link", "modes": {
            "YUY2": (60, 1920, 1080), "NV12": (60, 1920, 1080)}}
        t0 = time.time()
        cap, info = devices.negotiate(0, {"fourcc": "auto"}, opener=self.op, apis=[0],
                                      cached={"fourcc": "NV12", "width": 1920, "height": 1080})
        self.assertEqual(info["mode"][0], "NV12")
        self.assertLess(time.time() - t0, 2.0)
        cap.release()

    def test_missing_device(self):
        with self.assertRaises(RuntimeError):
            devices.negotiate(5, {"fourcc": "auto"}, opener=self.op, apis=[0])


class TestHotPlug(unittest.TestCase):
    def test_plug_unplug_replug_and_signal(self):
        w = FakeWorld()
        w.plugged[0] = {"name": "Integrated Camera", "modes": {"YUY2": (30, 640, 480)}}
        saved = {}
        cm = CaptureManager(lister=w.devices, opener=lambda i, api=None: FakeCap(w, i, api),
                            on_negotiated=lambda n, m: saved.update({n: m}))
        cm.SCAN_INTERVAL = 0.1
        cm.start({"type": "auto", "fourcc": "auto"})
        try:
            def wait(cond, t=6.0):
                end = time.time() + t
                while time.time() < end:
                    if cond():
                        return True
                    time.sleep(0.05)
                return False

            # Yalnızca dahili kamera: bekler, kamerayı açmaz
            self.assertTrue(wait(lambda: cm.status == "bekleniyor"))
            self.assertIsNone(cm.latest()[1])

            # Capture card takıldı
            with w.lock:
                w.plugged[2] = {"name": "USB3.0 Capture", "modes": {"YUY2": (60, 1920, 1080)}}
            self.assertTrue(wait(lambda: cm.status == "açık" and cm.latest()[1] is not None))
            self.assertEqual(cm.device["name"], "USB3.0 Capture")
            self.assertEqual(cm.latest()[1].shape[:2], (1080, 1920))
            self.assertIn("USB3.0 Capture", saved)

            # Sinyal kesildi
            w.signal = False
            self.assertTrue(wait(lambda: not cm.signal))
            w.signal = True
            self.assertTrue(wait(lambda: cm.signal))

            # Çıkarıldı
            with w.lock:
                del w.plugged[2]
            self.assertTrue(wait(lambda: cm.status == "bekleniyor"))
            self.assertIsNone(cm.latest()[1])

            # Yeniden takıldı (farklı indeksle)
            with w.lock:
                w.plugged[4] = {"name": "USB3.0 Capture", "modes": {"YUY2": (60, 1920, 1080)}}
            self.assertTrue(wait(lambda: cm.status == "açık" and cm.latest()[1] is not None))
            self.assertEqual(cm.device["index"], 4)
        finally:
            cm.stop()


if __name__ == "__main__":
    unittest.main()
