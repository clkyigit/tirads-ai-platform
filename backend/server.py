"""HTTP API + tarayıcı arayüzü (Starlette)."""

from __future__ import annotations

import asyncio
import contextlib
import re
import logging
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (FileResponse, JSONResponse, Response,
                                 StreamingResponse)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import measure, storage, tirads
from .capture import CaptureManager, probe_devices
from .config import ROOT, Config
from .segmentation import Segmenter

log = logging.getLogger(__name__)
FRONTEND = ROOT / "frontend"
MAX_FRAMES = 40
MAX_UPLOAD = 40 * 1024 * 1024


def apply_privacy(img: np.ndarray, masks: list) -> np.ndarray:
    if not masks:
        return img
    out = img.copy()
    H, W = out.shape[:2]
    for x, y, w, h in masks:
        x0, y0 = int(max(0, x) * W), int(max(0, y) * H)
        x1, y1 = int(min(1, x + w) * W), int(min(1, y + h) * H)
        out[y0:y1, x0:x1] = 0
    return out


class State:
    def __init__(self, config_path: Path | None = None, start_capture=True,
                 demo=False):
        self.cfg = Config(config_path)
        self.capture = CaptureManager()
        self.frames: OrderedDict[str, dict] = OrderedDict()
        self.frames_lock = threading.Lock()
        self.model_lock = threading.Lock()
        c = self.cfg.get()
        if demo:
            c["source"]["type"] = "demo"
            c["model"]["backend"] = "demo"
        self.segmenter = Segmenter(c["model"], ROOT)
        if start_capture:
            self.capture.start(c["source"])

    # ----------------------------------------------------------- kareler
    def put_frame(self, img: np.ndarray, origin: str) -> dict:
        fid = secrets.token_hex(6)
        img = apply_privacy(img, self.cfg.get()["privacy_masks"])
        with self.frames_lock:
            self.frames[fid] = {"image": img, "mask": None, "origin": origin,
                                "ai": None, "ts": time.time()}
            while len(self.frames) > MAX_FRAMES:
                self.frames.popitem(last=False)
        return {"frame_id": fid, "width": img.shape[1], "height": img.shape[0]}

    def get_frame(self, fid: str) -> dict:
        with self.frames_lock:
            fr = self.frames.get(fid)
        if fr is None:
            raise KeyError("Kare bulunamadı (süresi dolmuş olabilir). "
                           "Görüntüyü yeniden dondurun.")
        return fr

    def reload_model(self):
        with self.model_lock:
            self.segmenter = Segmenter(self.cfg.get()["model"], ROOT)


def _err(msg: str, code: int = 400):
    return JSONResponse({"error": msg}, status_code=code)


def _metrics_payload(contour, mm_per_px):
    m = measure.contour_metrics(contour)
    return {"contour": contour, "metrics_px": m,
            "metrics_mm": measure.to_mm(m, mm_per_px)}


def create_app(config_path: Path | None = None, start_capture=True,
               demo=False) -> Starlette:
    st = State(config_path, start_capture, demo)

    async def index(_):
        return FileResponse(FRONTEND / "index.html",
                            headers={"Cache-Control": "no-store"})

    async def status(_):
        c = st.cfg.get()
        return JSONResponse({
            "capture": st.capture.info(),
            "model": st.segmenter.info(),
            "calibration": c["calibration"],
            "roi": c["roi"], "privacy_masks": c["privacy_masks"],
            "lexicon": tirads.lexicon(),
        })

    async def devices(_):
        return JSONResponse({"devices": await run_in_threadpool(probe_devices)})

    async def set_source(req: Request):
        body = await req.json()
        allowed = {"type", "index", "path", "width", "height", "fps", "fourcc"}
        patch = {k: v for k, v in body.items() if k in allowed}
        if patch.get("type") not in ("device", "file", "demo"):
            return _err("type: device | file | demo olmalı")
        c = st.cfg.update({"source": patch})
        await run_in_threadpool(st.capture.start, c["source"])
        return JSONResponse({"ok": True, "source": c["source"]})

    async def stream(req: Request):
        sc = st.cfg.get()["stream"]
        q = int(sc.get("jpeg_quality", 80))
        period = 1.0 / float(sc.get("max_fps", 25))

        async def gen():
            last = -1
            while True:
                if await req.is_disconnected():
                    break
                fid, frame = st.capture.latest()
                if frame is None or fid == last:
                    await asyncio.sleep(0.02)
                    continue
                last = fid
                masks = st.cfg.get()["privacy_masks"]

                def enc():
                    img = apply_privacy(frame, masks)
                    return cv2.imencode(".jpg", img,
                                        [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
                jpg = await run_in_threadpool(enc)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(period)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"})

    async def freeze(_):
        _, frame = st.capture.latest()
        if frame is None:
            return _err("Henüz görüntü yok. Kaynağı kontrol edin.", 409)
        return JSONResponse(st.put_frame(frame.copy(), "capture"))

    async def upload(req: Request):
        data = await req.body()
        if not data:
            return _err("Boş dosya")
        if len(data) > MAX_UPLOAD:
            return _err("Dosya çok büyük", 413)
        img, dicom_mmpp = None, None
        if len(data) > 132 and data[128:132] == b"DICM":
            try:
                img, dicom_mmpp = await run_in_threadpool(_read_dicom, data)
            except Exception as e:
                return _err(f"DICOM okunamadı: {e}")
        else:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return _err("Görüntü çözülemedi (PNG/JPG/BMP/DICOM).")
        out = st.put_frame(img, "upload")
        out["dicom_mm_per_px"] = dicom_mmpp
        return JSONResponse(out)

    async def frame_png(req: Request):
        try:
            fr = st.get_frame(req.path_params["fid"])
        except KeyError as e:
            return _err(str(e), 404)
        png = await run_in_threadpool(
            lambda: cv2.imencode(".png", fr["image"])[1].tobytes())
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    async def segment(req: Request):
        body = await req.json()
        try:
            fr = st.get_frame(body.get("frame_id", ""))
        except KeyError as e:
            return _err(str(e), 404)
        seg = st.segmenter
        if not seg.ready:
            return _err(seg.error or "Model hazır değil", 503)
        click = body.get("click")
        click = (int(click[0]), int(click[1])) if click else None
        roi = st.cfg.get()["roi"]

        def run():
            with st.model_lock:
                t0 = time.time()
                r = seg.segment(fr["image"], roi, click)
                r["ms"] = (time.time() - t0) * 1000
                return r
        try:
            r = await run_in_threadpool(run)
        except Exception as e:
            log.exception("Segmentasyon hatası")
            return _err(f"Segmentasyon hatası: {e}", 500)
        if not r["mask"].any():
            fr["mask"] = None
            return JSONResponse({"empty": True, "ms": r["ms"],
                                 "message": "Nodül bulunamadı. Nodülün "
                                            "üzerine tıklayarak tekrar "
                                            "deneyin veya elle çizin."})
        contour = measure.mask_to_contour(r["mask"])
        fr["mask"] = r["mask"]
        fr["ai"] = {"backend": seg.backend, "confidence": r["confidence"],
                    "edited": False}
        mmpp = st.cfg.get()["calibration"]["mm_per_px"]
        out = _metrics_payload(contour, mmpp)
        out.update({"confidence": r["confidence"], "ms": r["ms"],
                    "backend": seg.backend})
        return JSONResponse(out)

    async def set_contour(req: Request):
        """Radyoloğun elle çizdiği/düzelttiği kontur."""
        body = await req.json()
        try:
            fr = st.get_frame(body.get("frame_id", ""))
        except KeyError as e:
            return _err(str(e), 404)
        contour = body.get("contour") or []
        if len(contour) < 3:
            fr["mask"] = None
            return JSONResponse({"cleared": True})
        H, W = fr["image"].shape[:2]
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.asarray(contour, np.int32).reshape(-1, 1, 2)], 1)
        fr["mask"] = mask
        prev = fr.get("ai") or {}
        fr["ai"] = {**prev, "edited": True} if prev else {"backend": "manuel",
                                                          "edited": True}
        mmpp = st.cfg.get()["calibration"]["mm_per_px"]
        try:
            return JSONResponse(_metrics_payload(contour, mmpp))
        except ValueError as e:
            return _err(str(e))

    async def score(req: Request):
        body = await req.json()
        s = tirads.score(body.get("features") or {})
        r = tirads.recommendation(s.get("level"), body.get("max_mm"))
        return JSONResponse({"score": s, "recommendation": r})

    async def config_get(_):
        return JSONResponse(st.cfg.get())

    async def config_set(req: Request):
        body = await req.json()
        patch, replace = {}, []
        if "roi" in body:
            roi = body["roi"]
            if roi is not None and (len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0):
                return _err("roi: [x, y, w, h] (0–1) olmalı")
            patch["roi"] = roi
        if "privacy_masks" in body:
            patch["privacy_masks"] = body["privacy_masks"] or []
        if "stream" in body:
            patch["stream"] = body["stream"]
        return JSONResponse(st.cfg.update(patch, tuple(replace)))

    async def calibration(req: Request):
        body = await req.json()
        cal = st.cfg.get()["calibration"]
        presets = dict(cal.get("presets") or {})
        if "points" in body:
            try:
                p1, p2 = body["points"]
                cal["mm_per_px"] = measure.mm_per_px_from_points(
                    p1, p2, float(body["distance_mm"]))
            except (ValueError, KeyError, TypeError) as e:
                return _err(f"Kalibrasyon hatası: {e}")
            cal["active_preset"] = None
        if "mm_per_px" in body:
            v = body["mm_per_px"]
            if v is not None and not (0 < float(v) < 5):
                return _err("mm/piksel 0–5 aralığında olmalı")
            cal["mm_per_px"] = v
            cal["active_preset"] = None
        if body.get("save_as"):
            if not cal.get("mm_per_px"):
                return _err("Kaydedilecek kalibrasyon yok")
            presets[body["save_as"]] = cal["mm_per_px"]
            cal["active_preset"] = body["save_as"]
        if body.get("activate"):
            name = body["activate"]
            if name not in presets:
                return _err("Ön ayar bulunamadı", 404)
            cal["mm_per_px"] = presets[name]
            cal["active_preset"] = name
        if body.get("delete"):
            presets.pop(body["delete"], None)
            if cal.get("active_preset") == body["delete"]:
                cal["active_preset"] = None
        cal["presets"] = presets
        c = st.cfg.update({"calibration": cal},
                          replace=("calibration.presets",))
        return JSONResponse(c["calibration"])

    async def model_set(req: Request):
        body = await req.json()
        allowed = set(st.cfg.get()["model"].keys())
        patch = {k: v for k, v in body.items() if k in allowed}
        st.cfg.update({"model": patch})
        await run_in_threadpool(st.reload_model)
        return JSONResponse(st.segmenter.info())

    async def cases_list(_):
        d = st.cfg.data_dir() / "cases"
        return JSONResponse({"cases": storage.list_cases(d)})

    async def case_get(req: Request):
        cid = req.path_params["cid"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", cid):
            return _err("Geçersiz vaka", 400)
        p = st.cfg.data_dir() / "cases" / cid / "report.txt"
        if not p.exists():
            return _err("Vaka bulunamadı", 404)
        return Response(p.read_text(encoding="utf-8"),
                        media_type="text/plain; charset=utf-8")

    async def case_save(req: Request):
        b = await req.json()
        if not b.get("approved"):
            return _err("Kayıt için radyolog onayı gerekli.")
        if not (b.get("radiologist") or "").strip():
            return _err("Onaylayan radyolog adı gerekli.")
        views = b.get("views") or {}
        frames, ai = {}, {}
        for view, v in views.items():
            if view not in storage.VIEW_LABELS or not v.get("frame_id"):
                continue
            try:
                fr = st.get_frame(v["frame_id"])
            except KeyError as e:
                return _err(f"{storage.VIEW_LABELS[view]}: {e}", 404)
            frames[view] = {"image": fr["image"], "mask": fr["mask"],
                            "lines": v.get("lines") or []}
            ai[view] = fr.get("ai")
        if not frames:
            return _err("En az bir görüntü (transvers/longitudinal) gerekli.")
        features = b.get("features") or {}
        meas = b.get("measurements") or {}
        s = tirads.score(features)
        if s.get("level") is None:
            return _err("TI-RADS formu eksik: " + ", ".join(
                s.get("missing", []) + s.get("errors", [])))
        dims = [meas.get(k) for k in ("width_mm", "ap_mm", "length_mm")]
        dims = [float(x) for x in dims if x not in (None, "")]
        meas["max_mm"] = max(dims) if dims else None
        rec = tirads.recommendation(s["level"], meas["max_mm"])
        used = [f"{storage.VIEW_LABELS[k]}: "
                + (("elle düzeltilmiş " if a.get("edited") else "")
                   + (a.get("backend") or "manuel")) if a else
                f"{storage.VIEW_LABELS[k]}: manuel"
                for k, a in ai.items()]
        case = {
            "case_id": storage.new_case_id(b.get("study_no", "")),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "study_no": (b.get("study_no") or "").strip(),
            "radiologist": b["radiologist"].strip(),
            "location": b.get("location") or {},
            "features": features, "score": s, "recommendation": rec,
            "measurements": meas,
            "calibration_mm_per_px": st.cfg.get()["calibration"]["mm_per_px"],
            "ai": {"views": ai, "summary": "; ".join(used),
                   "model": st.segmenter.info()},
            "notes": (b.get("notes") or "").strip(),
            "approved": True,
            "software": "tirads-ai-platform 0.2 (araştırma amaçlı)",
        }
        d = st.cfg.data_dir() / "cases"
        path = await run_in_threadpool(storage.save_case, d, case, frames)
        return JSONResponse({"case_id": case["case_id"], "path": str(path),
                             "report": storage.build_report(case),
                             "recommendation": rec, "score": s})

    routes = [
        Route("/", index),
        Route("/api/status", status),
        Route("/api/devices", devices),
        Route("/api/source", set_source, methods=["POST"]),
        Route("/api/stream", stream),
        Route("/api/freeze", freeze, methods=["POST"]),
        Route("/api/upload", upload, methods=["POST"]),
        Route("/api/frame/{fid}.png", frame_png),
        Route("/api/segment", segment, methods=["POST"]),
        Route("/api/contour", set_contour, methods=["POST"]),
        Route("/api/score", score, methods=["POST"]),
        Route("/api/config", config_get),
        Route("/api/config", config_set, methods=["POST"]),
        Route("/api/calibration", calibration, methods=["POST"]),
        Route("/api/model", model_set, methods=["POST"]),
        Route("/api/cases", cases_list),
        Route("/api/cases", case_save, methods=["POST"]),
        Route("/api/cases/{cid}/report", case_get),
        Mount("/static", StaticFiles(directory=FRONTEND), name="static"),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        st.capture.stop()

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.st = st
    return app


def _dicom_mm_per_px(ds) -> float | None:
    """US DICOM'da SequenceOfUltrasoundRegions içindeki fiziksel piksel
    aralığı (birim 3 = cm). Varsa kalibrasyon için önerilir."""
    for reg in getattr(ds, "SequenceOfUltrasoundRegions", []) or []:
        try:
            if int(reg.PhysicalUnitsXDirection) == 3 and \
                    int(reg.PhysicalUnitsYDirection) == 3:
                dx = abs(float(reg.PhysicalDeltaX)) * 10
                dy = abs(float(reg.PhysicalDeltaY)) * 10
                if dx > 0 and abs(dx - dy) / dx < 0.02:
                    return dx
        except (AttributeError, ValueError, TypeError):
            continue
    return None


def _read_dicom(data: bytes) -> tuple[np.ndarray, float | None]:
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError("DICOM için 'pydicom' kurun") from e
    import io
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array
    if arr.ndim == 4:      # çok kareli renkli
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] not in (3, 4):  # çok kareli gri
        arr = arr[0]
    arr = arr.astype(np.float32)
    arr = (255 * (arr - arr.min()) / max(1e-6, arr.max() - arr.min())).astype(np.uint8)
    mmpp = _dicom_mm_per_px(ds)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR), mmpp
    return cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR), mmpp
