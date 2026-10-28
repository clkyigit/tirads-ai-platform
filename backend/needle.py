"""In-plane biyopsi iğnesi algılama, hedef takibi ve rehber geometrisi.

Eğitilmiş model gerektirmez. İn-plane iğne USG'de parlak, düz, eğik bir
çizgi olarak görünür (çoğu zaman altında reverberasyon çizgileriyle). Yöntem:

1. ROI küçültülür, arka plan çıkarılır (beyaz top-hat benzeri): ince parlak
   yapılar öne çıkar, büyük parlak alanlar (kalsifikasyon gölgesi, kapsül
   dışı doku) bastırılır.
2. En parlak pikseller eşiklenir, olasılıksal Hough ile doğru parçaları
   bulunur. Neredeyse yatay çizgiler (fasya, kas sınırı, tiroid kapsülü)
   elenir.
3. Plan (giriş noktası → hedef) varsa açı ve konum ön bilgisi olarak
   kullanılır; plan yoksa en uzun/parlak eğik çizgi seçilir.
4. Seçilen çizgiye yakın pikseller cv2.fitLine ile hassaslaştırılır; çizgi
   boyunca parlaklık profiliyle iğnenin görünen ucu bulunur.
5. Kareler arasında yumuşatma yapılır; kısa kayıplarda son konum korunur.

Hedef, seçildiği andaki görüntü parçasıyla (şablon eşleme) takip edilir;
böylece solunum ve hafif prob hareketinde hedef işareti nodülle birlikte
kayar.

UYARI: Karar destek amaçlıdır. Görüntüdeki iğneyi gözle takip etmenin
yerini tutmaz; algılama hatalı ya da gecikmeli olabilir.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

WORK_WIDTH = 480          # algılama genişliği (px)
MIN_ANGLE = 8.0           # bundan yatay çizgiler iğne sayılmaz (derece)
MAX_ANGLE = 82.0


# ------------------------------------------------------------- yardımcılar
def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (1.0, 0.0)


def line_angle_deg(p, q) -> float:
    """Yataydan açı, 0–90 (yön bağımsız)."""
    return math.degrees(math.atan2(abs(q[1] - p[1]), abs(q[0] - p[0]) + 1e-9))


def _dir_diff_deg(d1, d2) -> float:
    """İki doğrunun yön farkı (0–90), yön işaretinden bağımsız."""
    a = math.degrees(math.atan2(d1[1], d1[0]))
    b = math.degrees(math.atan2(d2[1], d2[0]))
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _point_line_dist(p, a, u):
    return abs((p[0] - a[0]) * u[1] - (p[1] - a[1]) * u[0])


def _sample(img, pts):
    h, w = img.shape
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return img[ys, xs]


def _line_in_box(p, u, w, h):
    """p'den geçen u yönlü doğrunun [0,w)×[0,h) içindeki iki ucu."""
    ts = []
    for (pc, uc, lim) in ((p[0], u[0], w - 1), (p[1], u[1], h - 1)):
        if abs(uc) < 1e-9:
            if not (0 <= pc <= lim):
                return None
            continue
        ts += [(0 - pc) / uc, (lim - pc) / uc]
    if not ts:
        return None
    ts = sorted(ts)
    # Kutu içinde kalan aralık: tüm sınır kesişimlerinin ortadaki ikisi
    cands = [t for t in ts if 0 - 1e-6 <= p[0] + t * u[0] <= w - 1 + 1e-6
             and 0 - 1e-6 <= p[1] + t * u[1] <= h - 1 + 1e-6]
    if len(cands) < 2:
        return None
    t0, t1 = min(cands), max(cands)
    return (p[0] + t0 * u[0], p[1] + t0 * u[1]), (p[0] + t1 * u[0], p[1] + t1 * u[1])


# --------------------------------------------------------------- algılama
def enhance(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 7)
    th = g - bg
    np.maximum(th, 0, out=th)
    return cv2.GaussianBlur(th, (0, 0), 1.0)


def detect(gray_full: np.ndarray, roi_px=None, plan: dict | None = None,
           work_width: int = WORK_WIDTH, prev: dict | None = None,
           motion: np.ndarray | None = None) -> dict | None:
    """Tek karede iğne algılar. Koordinatlar tam kare pikselidir.

    plan:   {"entry": [x,y] | None, "target": [x,y] | None}
    prev:   önceki karede bulunan iğne ({"entry","tip"}) – takip ön bilgisi
    motion: küçültülmüş ROI boyutunda hareket haritası (iğne ilerlerken
            hareket eder, fasya/kapsül durağandır)
    Dönüş: {"entry", "tip", "angle_deg", "conf", "length_px"} ya da None.
    """
    H, W = gray_full.shape[:2]
    x0, y0, x1, y1 = roi_px if roi_px else (0, 0, W, H)
    crop = gray_full[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    s = min(1.0, work_width / crop.shape[1])
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
    h, w = small.shape
    th = enhance(small)
    thr = float(np.percentile(th, 98.5))
    if thr <= 0.5:
        return None
    binary = (th > thr).astype(np.uint8) * 255

    min_len = max(12, int(0.08 * w))
    segs = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=max(10, int(0.05 * w)),
                           minLineLength=min_len, maxLineGap=max(4, int(0.02 * w)))
    if segs is None or len(segs) == 0:
        return None
    segs = np.asarray(segs).reshape(-1, 4)   # OpenCV sürümüne göre (N,1,4) ya da (N,4)

    to_small = lambda p: ((p[0] - x0) * s, (p[1] - y0) * s)  # noqa: E731
    plan_len = 0.0
    # Ön bilgiler: (doğru üzerinde nokta, yön, koridor genişliği, açı sigma)
    priors = []
    plan_dir = plan_pt = None
    if plan and plan.get("target") is not None:
        tgt = to_small(plan["target"])
        if plan.get("entry") is not None:
            ent = to_small(plan["entry"])
            plan_dir, plan_pt = _unit((tgt[0] - ent[0], tgt[1] - ent[1])), ent
            priors.append((plan_pt, plan_dir, 0.08 * w, 12.0))
            plan_len = math.dist(ent, tgt)
        else:
            plan_pt = tgt   # yalnızca hedef: iğne hattı hedefe yakın olmalı
    prev_prior = None
    if prev is not None:
        pe, pt_ = to_small(prev["entry"]), to_small(prev["tip"])
        prev_prior = (pe, _unit((pt_[0] - pe[0], pt_[1] - pe[1])), 0.05 * w, 8.0)
    mnoise = None
    if motion is not None and motion.shape == th.shape:
        mnoise = float(np.median(motion)) + 0.5

    best, best_score = None, 0.0
    for x_1, y_1, x_2, y_2 in segs:
        p, q = (float(x_1), float(y_1)), (float(x_2), float(y_2))
        ang = line_angle_deg(p, q)
        if ang < MIN_ANGLE or ang > MAX_ANGLE:
            continue
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(8, int(L))
        pts = np.stack([np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)], 1)
        r_pts = _sample(th, pts)
        u = _unit((q[0] - p[0], q[1] - p[1]))
        # Nokta başına ağırlık: koridor dışındaki kısımlar katkı vermez.
        # (Plan hattını yalnızca kesen uzun fasya çizgisi böylece elenir.)
        wts = np.ones(n, np.float32)
        ang_prior = 1.0
        for (a0, du, corr, sig) in priors:
            dist = np.abs((pts[:, 0] - a0[0]) * du[1] - (pts[:, 1] - a0[1]) * du[0])
            wts *= np.exp(-(dist / corr) ** 2)
            ang_prior *= math.exp(-(_dir_diff_deg(u, du) / sig) ** 2)
        if plan_dir is not None:
            # İğne giriş ile hedefin biraz ötesi arasında olmalı: plan ekseni
            # boyunca bu aralığın dışına taşan kısımlar (ör. tüm görüntüyü
            # boydan boya geçen fasya) katkı vermez.
            along = (pts[:, 0] - plan_pt[0]) * plan_dir[0] + (pts[:, 1] - plan_pt[1]) * plan_dir[1]
            lo, hi = -0.06 * w, plan_len + 0.12 * w
            excess = np.maximum(lo - along, 0) + np.maximum(along - hi, 0)
            wts *= np.exp(-(excess / (0.04 * w)) ** 2)
        if not priors and plan_pt is not None:
            ang_prior *= 0.35 + 0.65 * math.exp(-(_point_line_dist(plan_pt, p, u) / (0.2 * w)) ** 2)
        if prev_prior is not None:
            # Önceki kare ön bilgisi yumuşak: yanlış çizgiye kilitlenince
            # kurtulabilsin
            a0, du, corr, sig = prev_prior
            dist = _point_line_dist(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), a0, du)
            ang_prior *= 0.35 + 0.65 * math.exp(-(_dir_diff_deg(u, du) / sig) ** 2
                                                 - (dist / corr) ** 2)
        # Görüntüyü iki yandan da kesen parlak çizgi anatomiktir (fasya,
        # kapsül); iğne tek kenardan girer ve görüntünün içinde biter.
        ext = _line_in_box(p, u, w, h)
        if ext is not None:
            e0, e1 = ext
            edge = max(4.0, 0.04 * w)
            near0 = np.stack([e0[0] + u[0] * np.arange(0, edge), e0[1] + u[1] * np.arange(0, edge)], 1)
            near1 = np.stack([e1[0] - u[0] * np.arange(0, edge), e1[1] - u[1] * np.arange(0, edge)], 1)
            if _sample(th, near0).mean() > thr and _sample(th, near1).mean() > thr:
                ang_prior *= 0.1
        score = float((r_pts * wts).sum()) * ang_prior
        if mnoise is not None:
            # İlerleyen iğnenin ucu belirgin hareket üretir; ortalama yerine
            # hat üzerindeki en güçlü hareket, gürültüye oranla
            mv = float(_sample(motion, pts).max()) / mnoise
            score *= 1.0 + 3.0 * min(1.0, max(0.0, (mv - 5.0) / 10.0))
        if score > best_score:
            best, best_score = (p, q, u), score
    if best is None:
        return None

    p, q, u = best
    # Aynı doğru üzerindeki parçalar + yakın parlak pikseller → fitLine
    ys, xs = np.nonzero(binary)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    d = np.abs((pts[:, 0] - p[0]) * u[1] - (pts[:, 1] - p[1]) * u[0])
    near = pts[d < 3.0]
    if len(near) >= 10:
        vx, vy, cx, cy = cv2.fitLine(near, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        u2 = _unit((float(vx), float(vy)))
        if _dir_diff_deg(u2, u) < 8:  # sapmışsa Hough doğrusunu koru
            u, p = u2, (float(cx), float(cy))
    if u[1] < 0:  # yönü derine doğru (y artan) sabitle
        u = (-u[0], -u[1])

    # Çizgi boyunca parlaklık profili → görünen iğne parçası
    diag = math.hypot(w, h)
    ss = np.arange(-diag, diag, 1.0)
    line_pts = np.stack([p[0] + ss * u[0], p[1] + ss * u[1]], 1)
    inside = (line_pts[:, 0] >= 0) & (line_pts[:, 0] < w) & \
             (line_pts[:, 1] >= 0) & (line_pts[:, 1] < h)
    ss, line_pts = ss[inside], line_pts[inside]
    if len(ss) < min_len:
        return None
    r = _sample(th, line_pts)
    r = np.convolve(r, np.ones(3) / 3, mode="same")
    on = r > max(0.45 * thr, 0.35 * float(np.percentile(r, 95)))
    # İğne incedir: hattın hemen üstü (prob tarafı) karanlıktır. Fasya/kapsül
    # gibi geniş parlak yapılarda üst taraf da parlaktır; uç bu bölgelere
    # taşmasın. (Alt taraf kullanılmaz: reverberasyon çizgileri iğnenin
    # altında oluşur.)
    n_up = (u[1], -u[0]) if u[0] > 0 else (-u[1], u[0])   # y'si negatif normal
    if n_up[1] > 0:
        n_up = (-n_up[0], -n_up[1])
    off = max(3.0, 0.011 * w)
    above = _sample(th, line_pts + np.array([n_up[0] * off, n_up[1] * off]))
    above = np.convolve(above, np.ones(3) / 3, mode="same")
    needle_like = r > 1.5 * above
    # En uzun "açık" bölüm (en fazla 'gap' boşlukla)
    gap = max(4, int(0.015 * w))
    runs, start, last = [], None, None
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            elif i - last > gap:
                runs.append((start, last))
                start = i
            last = i
    if start is not None:
        runs.append((start, last))
    if not runs:
        return None

    def run_score(a_b):
        a, b = a_b
        return float(r[a:b + 1].sum())
    a, b = max(runs, key=run_score)
    # Uçları kırp: iğne gibi olmayan (üstü de parlak) kısımlar uç sayılmaz.
    # Yalnızca uçlardan kırpılır; iğnenin fasyayı kestiği orta kısım bölünmez.
    while b > a and not needle_like[b]:
        b -= 1
    while a < b and not needle_like[a]:
        a += 1
    # Uçtaki kısa "ada": iğne ucundan küçük bir boşlukla ayrılan kısa parlak
    # parça çoğunlukla ucun değdiği kapsül/fasyadır; uç sayılmaz.
    island_max = max(6, int(0.03 * w))
    for _ in range(2):
        seg = on[a:b + 1]
        offs = np.flatnonzero(~seg)
        if offs.size == 0:
            break
        last_gap_end = int(offs[-1])            # derin uca en yakın boşluk
        if (b - a) - last_gap_end <= island_max and last_gap_end > island_max:
            gap_start = last_gap_end
            while gap_start > 0 and not seg[gap_start - 1]:
                gap_start -= 1
            b = a + gap_start - 1
        else:
            break
    length = float(ss[b] - ss[a])
    if length < min_len:
        return None
    mean_resp = float(r[a:b + 1].mean())
    if mean_resp < 1.25 * thr:
        return None
    shallow = (p[0] + ss[a] * u[0], p[1] + ss[a] * u[1])
    deep = (p[0] + ss[b] * u[0], p[1] + ss[b] * u[1])

    entry_s, tip_s = shallow, deep
    if plan and plan.get("entry") is not None:
        ent = to_small(plan["entry"])
        if math.dist(deep, ent) < math.dist(shallow, ent):
            entry_s, tip_s = deep, shallow

    conf = 0.5 * min(1.0, length / (0.25 * w)) + 0.5 * min(1.0, mean_resp / (3.0 * thr))
    if plan_dir is not None:
        conf *= 0.4 + 0.6 * math.exp(-(_dir_diff_deg(u, plan_dir) / 15.0) ** 2)
    to_full = lambda q_: [q_[0] / s + x0, q_[1] / s + y0]  # noqa: E731
    entry, tip = to_full(entry_s), to_full(tip_s)
    return {"entry": entry, "tip": tip, "angle_deg": line_angle_deg(entry, tip),
            "conf": float(min(1.0, conf)), "length_px": length / s}


# ------------------------------------------------------- zamansal takip
class NeedleSmoother:
    """Kareler arası yumuşatma ve kısa kayıp toleransı."""

    def __init__(self, alpha=0.55, hold_s=0.6, min_conf=0.25):
        self.alpha, self.hold_s, self.min_conf = alpha, hold_s, min_conf
        self.state: dict | None = None
        self.last_seen = 0.0
        self._pending = None   # ani sıçrama: iki kare doğrulama

    def reset(self):
        self.state, self._pending = None, None

    def update(self, det: dict | None, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        if det is not None and det["conf"] >= self.min_conf:
            if self.state is not None:
                du = _dir_diff_deg(_unit(np.subtract(det["tip"], det["entry"])),
                                   _unit(np.subtract(self.state["tip"], self.state["entry"])))
                if du > 15 and det["conf"] < 0.7:
                    if self._pending is not None and _dir_diff_deg(
                            _unit(np.subtract(det["tip"], det["entry"])),
                            _unit(np.subtract(self._pending["tip"], self._pending["entry"]))) < 6:
                        self.state = dict(det)       # doğrulandı: yeni konuma geç
                        self._pending = None
                    else:
                        self._pending = det
                        return self._out(now)
                else:
                    a = self.alpha
                    for k in ("entry", "tip"):
                        self.state[k] = [a * det[k][i] + (1 - a) * self.state[k][i] for i in (0, 1)]
                    self.state["conf"] = a * det["conf"] + (1 - a) * self.state["conf"]
                    self.state["angle_deg"] = line_angle_deg(self.state["entry"], self.state["tip"])
                    self._pending = None
            else:
                self.state = dict(det)
            self.last_seen = now
        return self._out(now)

    def _out(self, now):
        if self.state is None:
            return None
        age = now - self.last_seen
        if age > self.hold_s * 4:
            self.state = None
            return None
        return {**self.state, "age_s": age, "found": age <= self.hold_s}


# ---------------------------------------------------------- hedef takibi
class TargetTracker:
    """Hedef çevresindeki görüntü parçasını şablon eşlemeyle izler."""

    SCALE = 0.5

    def __init__(self, patch=48, search=50, min_score=0.45):
        self.patch, self.search, self.min_score = patch, search, min_score
        self.tpl = None
        self.pos = None       # tam kare koordinatı
        self.score = 0.0
        self.lost = False

    def set(self, gray_full, pos):
        self.pos = [float(pos[0]), float(pos[1])]
        self.lost, self.score = False, 1.0
        g = cv2.resize(gray_full, None, fx=self.SCALE, fy=self.SCALE,
                       interpolation=cv2.INTER_AREA)
        self.tpl = self._crop(g, self.pos[0] * self.SCALE, self.pos[1] * self.SCALE,
                              self.patch // 2)

    def clear(self):
        self.tpl, self.pos = None, None

    @staticmethod
    def _crop(g, cx, cy, r):
        h, w = g.shape
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        if x0 < 0 or y0 < 0 or x0 + 2 * r > w or y0 + 2 * r > h:
            return None
        return g[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()

    def update(self, gray_full):
        if self.tpl is None or self.pos is None:
            return self.pos
        g = cv2.resize(gray_full, None, fx=self.SCALE, fy=self.SCALE,
                       interpolation=cv2.INTER_AREA)
        r = self.patch // 2
        cx, cy = self.pos[0] * self.SCALE, self.pos[1] * self.SCALE
        win = self._crop(g, cx, cy, r + self.search)
        if win is None:  # kenara yakın: pencereyi kırp
            h, w = g.shape
            x0 = max(0, int(cx) - r - self.search)
            y0 = max(0, int(cy) - r - self.search)
            win = g[y0:min(h, int(cy) + r + self.search), x0:min(w, int(cx) + r + self.search)]
            off = (x0, y0)
        else:
            off = (int(round(cx)) - r - self.search, int(round(cy)) - r - self.search)
        if win.shape[0] < self.tpl.shape[0] or win.shape[1] < self.tpl.shape[1]:
            self.lost = True
            return self.pos
        res = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        self.score = float(mx)
        if mx >= self.min_score:
            nx = (off[0] + loc[0] + r) / self.SCALE
            ny = (off[1] + loc[1] + r) / self.SCALE
            self.pos = [0.5 * nx + 0.5 * self.pos[0], 0.5 * ny + 0.5 * self.pos[1]]
            self.lost = False
        else:
            self.lost = True
        return self.pos


# --------------------------------------------------------------- geometri
def guidance(needle: dict | None, target, entry, radius_mm: float,
             mm_per_px: float | None, skin_y: float = 0.0) -> dict:
    """İğne ve hedefe göre rehber ölçümleri. mm_per_px yoksa piksel."""
    k = mm_per_px or 1.0
    unit = "mm" if mm_per_px else "px"
    R = radius_mm / mm_per_px if mm_per_px else 25.0   # hedef yarıçapı (px)
    out = {"unit": unit, "status": "searching", "target_radius_px": R}
    if entry is not None and target is not None:
        out["plan_angle_deg"] = line_angle_deg(entry, target)
        out["plan_length"] = math.dist(entry, target) * k
    if not needle or not needle.get("found"):
        return out
    e, t = needle["entry"], needle["tip"]
    u = _unit((t[0] - e[0], t[1] - e[1]))
    out["tip_depth"] = max(0.0, t[1] - skin_y) * k
    out["needle_angle_deg"] = needle["angle_deg"]
    if entry is not None and target is not None:
        out["angle_diff_deg"] = needle["angle_deg"] - out["plan_angle_deg"]
    if target is None:
        out["status"] = "tracking"
        return out
    tv = (target[0] - e[0], target[1] - e[1])
    s_t = tv[0] * u[0] + tv[1] * u[1]                 # hedefin hat üzerindeki izdüşümü
    s_tip = math.dist(e, t)
    cross = u[0] * tv[1] - u[1] * tv[0]               # >0: hedef hattın "altında" (derin)
    perp = abs(cross)
    tip_to_target = math.dist(t, target)
    out.update({
        "tip_to_target": tip_to_target * k,
        "remaining": (s_t - s_tip) * k,
        "miss": perp * k if s_t > 0 else None,
        "target_side": "deeper" if cross > 0 else "shallower",
    })
    if tip_to_target <= R:
        out["status"] = "in_target"
    elif s_t - s_tip < -R:
        out["status"] = "passed"
    elif s_t > 0 and perp <= R:
        out["status"] = "on_track"
    else:
        out["status"] = "off_track"
    return out
