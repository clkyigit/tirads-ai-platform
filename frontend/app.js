"use strict";
// TI-RADS AI Platform – tarayıcı arayüzü (bağımlılıksız, internet gerektirmez)

const $ = (id) => document.getElementById(id);
const VIEWS = ["transverse", "longitudinal"];
const LINE_LABEL = { width: "G", ap: "AP", length: "U" };
const LINE_INPUT = { width: "mWidth", ap: "mAp", length: "mLength" };
const CAT_TITLE = {
  composition: "1. Kompozisyon", echogenicity: "2. Ekojenite", shape: "3. Şekil",
  margin: "4. Kenar", foci: "5. Ekojenik odaklar (birden fazla seçilebilir)",
};

const S = {
  status: null,
  view: "transverse",
  tool: null,
  views: {},           // view -> {frameId, w, h, img, contour, ai, lines:{}}
  manualDims: {},      // elle girilmiş mm değerleri (çizgiden bağımsız)
  drag: null,          // aktif sürükleme
  poly: null,          // elle çizim noktaları
  calibPts: null,
  shapeTouched: false,
  lastScore: null,
};
VIEWS.forEach((v) => (S.views[v] = emptyView()));

function emptyView() {
  // aiContour: AI'ın ilk önerisi; confirmed: radyolog onayı; hist/fut: geri al/yinele
  return { frameId: null, w: 0, h: 0, img: null, contour: null, ai: null, lines: {},
           aiContour: null, confirmed: false, aiDice: null, hist: [], fut: [] };
}
const cur = () => S.views[S.view];
const mmpp = () => S.status?.calibration?.mm_per_px || null;

// ------------------------------------------------------------------ API
async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.method = o.method || "POST";
    o.headers = { "Content-Type": "application/json" };
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  const r = await fetch(path, o);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}

let toastTimer;
function toast(msg, err = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), err ? 6000 : 3000);
}
function viewerMsg(msg) {
  const m = $("viewerMsg");
  m.textContent = msg || "";
  m.classList.toggle("show", !!msg);
}
function store(k, v) { try { localStorage.setItem(k, v); } catch (_) {} }
function load(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }

// --------------------------------------------------------------- durum
async function refreshStatus() {
  try {
    S.status = await api("/api/status");
  } catch (e) {
    $("chipSource").textContent = "Sunucuya bağlanılamıyor";
    $("chipSource").className = "chip bad";
    return;
  }
  const c = S.status.capture, m = S.status.model, cal = S.status.calibration;
  renderSource(c);
  const chipM = $("chipModel");
  chipM.textContent = m.ready ? `Model: ${m.is_demo ? "DEMO" : "UNet++ " + m.backend} (${m.device})` : "Model: yok";
  chipM.className = "chip " + (m.ready ? (m.is_demo ? "warn" : "ok") : "bad");
  chipM.title = m.error || m.path || "";
  const chipC = $("chipCal");
  chipC.textContent = cal.mm_per_px ? `Kalibrasyon: ${cal.mm_per_px.toFixed(4)} mm/px${cal.active_preset ? " · " + cal.active_preset : ""}` : "Kalibrasyon: yok";
  chipC.className = "chip " + (cal.mm_per_px ? "ok" : "warn");
  $("calWarn").hidden = !!cal.mm_per_px;
  document.querySelector('[data-tool="ai"]').disabled = !m.ready || !cur().frameId;
}

// Kaynak durumu: üst çip + canlı görüntü üzerindeki durum ekranı
let lastConnectedAt = 0;
function renderSource(c) {
  const d = c.device || {};
  const chip = $("chipSource");
  const open = c.status === "açık";
  let txt, cls;
  if (open) {
    txt = `${d.name || "Kaynak"} · ${c.resolution ? c.resolution[0] + "×" + c.resolution[1] : "–"} · ${d.fourcc || ""} · ${c.fps} fps`;
    cls = !c.signal ? "warn" : d.compressed && c.source.type !== "demo" && c.source.type !== "file" ? "warn" : "ok";
    if (!c.signal) txt += " · sinyal yok";
  } else if (c.status === "bekleniyor") { txt = "Capture card bekleniyor"; cls = "warn"; }
  else if (c.status === "bağlanıyor") { txt = "Capture card bağlanıyor…"; cls = "warn"; }
  else { txt = "Kaynak hatası"; cls = "bad"; }
  chip.textContent = txt;
  chip.className = "chip " + cls;
  chip.title = [c.error, d.compressed && open ? "Sıkıştırmalı format (MJPG): ince ayrıntılar bozulabilir." : ""].filter(Boolean).join("\n");

  // Durum ekranı (yalnızca canlı görünümde)
  const vs = $("viewerState");
  let state = null, title = "", text = "";
  if (c.status === "bekleniyor") {
    state = "wait"; title = "Capture card bekleniyor";
    text = "Capture card'ı USB'ye takın; görüntü kendiliğinden gelecek." + (c.error && c.error.includes("Yok sayılan") ? " " + c.error.replace("Capture card bulunamadı. ", "") : "");
  } else if (c.status === "bağlanıyor") {
    state = "connecting"; title = "Bağlanıyor…"; text = "En iyi görüntü formatı deneniyor (birkaç saniye).";
  } else if (c.status === "hata") {
    state = "error"; title = "Capture card açılamadı";
    text = (c.error || "") + " Kartı çıkarıp yeniden takmayı deneyin." +
      (navigator.platform.toLowerCase().includes("linux") ? " Linux'ta kullanıcı 'video' grubunda olmalı." : "");
  } else if (open && !c.signal) {
    state = "nosignal"; title = "Sinyal yok";
    text = "Capture card bağlı ama görüntü gelmiyor. USG cihazının video çıkışının (HDMI/DVI) açık ve kablonun takılı olduğunu kontrol edin.";
  }
  const showState = state && !cur().frameId;
  vs.hidden = !showState;
  if (showState) { vs.className = "viewer-state " + state; $("vsTitle").textContent = title; $("vsText").textContent = text; }

  if (open && c.connected_at && c.connected_at !== lastConnectedAt) {
    if (lastConnectedAt) toast(`Bağlandı: ${d.name} · ${c.resolution ? c.resolution.join("×") : ""} · ${d.fourcc}` + (d.compressed && c.source.type === "auto" ? " (sıkıştırmalı format)" : ""));
    lastConnectedAt = c.connected_at;
    if (!cur().frameId) live.src = "/api/stream?" + Date.now();  // akışı tazele
  }
}

// ------------------------------------------------------- TI-RADS formu
function buildForm(lex) {
  const f = $("tiradsForm");
  f.innerHTML = "";
  for (const cat of ["composition", "echogenicity", "shape", "margin", "foci"]) {
    const fs = document.createElement("fieldset");
    fs.innerHTML = `<legend>${CAT_TITLE[cat]}<span class="tag" id="tag_${cat}"></span></legend>`;
    const opts = document.createElement("div");
    opts.className = "opts";
    for (const o of lex[cat]) {
      const type = cat === "foci" ? "checkbox" : "radio";
      const lab = document.createElement("label");
      lab.className = "opt";
      lab.innerHTML = `<input type="${type}" name="${cat}" value="${o.value}"><span>${o.label}</span><span class="p">${o.points}</span>`;
      opts.appendChild(lab);
    }
    fs.appendChild(opts);
    f.appendChild(fs);
  }
  f.addEventListener("change", (e) => {
    if (e.target.name === "foci") {
      const boxes = [...f.querySelectorAll('input[name="foci"]')];
      if (e.target.value === "none" && e.target.checked) boxes.forEach((b) => b !== e.target && (b.checked = false));
      else if (e.target.checked) boxes.find((b) => b.value === "none").checked = false;
    }
    if (e.target.name === "shape") { S.shapeTouched = true; $("tag_shape").textContent = ""; }
    scheduleScore();
  });
}

function features() {
  const f = $("tiradsForm");
  const get = (n) => f.querySelector(`input[name="${n}"]:checked`)?.value || null;
  return {
    composition: get("composition"), echogenicity: get("echogenicity"),
    shape: get("shape"), margin: get("margin"),
    foci: [...f.querySelectorAll('input[name="foci"]:checked')].map((x) => x.value),
  };
}

let scoreTimer;
function scheduleScore() {
  clearTimeout(scoreTimer);
  scoreTimer = setTimeout(updateScore, 80);
}
async function updateScore() {
  const dims = currentDims();
  let r;
  try {
    r = await api("/api/score", { json: { features: features(), max_mm: dims.max } });
  } catch (e) { return toast(e.message, true); }
  S.lastScore = r;
  const s = r.score, rec = r.recommendation;
  const card = $("resultCard");
  for (let i = 1; i <= 5; i++) card.classList.toggle("lvl" + i, s.level === i);
  $("trLevel").textContent = s.level ? s.level_text : "TR –";
  $("trPoints").textContent = s.total ?? "–";
  const names = { composition: "Komp.", echogenicity: "Ekoj.", shape: "Şekil", margin: "Kenar", foci: "Odak" };
  $("breakdown").textContent = Object.entries(s.breakdown || {}).map(([k, v]) => `${names[k]} ${v}`).join(" · ");
  const recEl = $("recText");
  recEl.textContent = s.errors?.length ? s.errors.join(" ") : (s.missing?.length ? "Eksik: " + s.missing.map((m) => CAT_TITLE[m].replace(/^\d\. /, "").replace(/ \(.*/, "")).join(", ") : rec.text);
  recEl.className = "rec" + (rec.action && !s.missing?.length ? " " + rec.action : "");
  $("scoreNotes").textContent = (s.notes || []).join(" ");
  updateSaveEnabled();
}

// ------------------------------------------------------------ ölçümler
function lineLenPx(l) { return Math.hypot(l.points[1][0] - l.points[0][0], l.points[1][1] - l.points[0][1]); }

function currentDims() {
  // Her ölçüm hangi görünümde çizildiyse oradan alınır; elle girilen değer önceliklidir.
  const out = {};
  for (const k of ["width", "ap", "length"]) {
    if (S.manualDims[k] != null) { out[k] = S.manualDims[k]; continue; }
    let v = null;
    for (const view of VIEWS) {
      const l = S.views[view].lines[k];
      if (l && mmpp()) v = lineLenPx(l) * mmpp();
    }
    out[k] = v;
  }
  const vals = Object.values(out).filter((x) => x != null && x > 0);
  out.max = vals.length ? Math.max(...vals) : null;
  return out;
}

function renderDims() {
  const d = currentDims();
  for (const k of ["width", "ap", "length"]) {
    const inp = $(LINE_INPUT[k]);
    if (document.activeElement !== inp) inp.value = d[k] != null ? d[k].toFixed(1) : "";
    inp.classList.toggle("from-line", S.manualDims[k] == null && d[k] != null);
  }
  $("mMax").textContent = d.max != null ? `${d.max.toFixed(1)} mm (${(d.max / 10).toFixed(2)} cm)` : "–";
  // Şekil önerisi: transvers görüntüde AP > genişlik ise taller-than-wide
  const hint = $("shapeHint");
  if (d.width && d.ap) {
    const taller = d.ap > d.width;
    hint.textContent = `AP/G = ${(d.ap / d.width).toFixed(2)} → ${taller ? "taller-than-wide" : "wider-than-tall"}`;
    if (!S.shapeTouched) {
      const r = $("tiradsForm").querySelector(`input[name="shape"][value="${taller ? "taller" : "wider"}"]`);
      if (r && !r.checked) { r.checked = true; $("tag_shape").textContent = "ölçümden önerildi"; }
    }
  } else hint.textContent = "";
  scheduleScore();
}

function onDimInput(k) {
  const v = parseFloat($(LINE_INPUT[k]).value);
  S.manualDims[k] = Number.isFinite(v) && v > 0 ? v : null;
  renderDims();
}

// ----------------------------------------------------------- görüntüleyici
const cv = $("cv"), ctx = cv.getContext("2d"), live = $("live");
let T = { s: 1, ox: 0, oy: 0 };

function fit() {
  const r = $("viewer").getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  const v = cur();
  if (v.w) {
    const s = Math.min(cv.width / v.w, cv.height / v.h);
    T = { s, ox: (cv.width - v.w * s) / 2, oy: (cv.height - v.h * s) / 2, dpr };
  }
  draw();
}
const toImg = (ev) => {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  return [((ev.clientX - r.left) * dpr - T.ox) / T.s, ((ev.clientY - r.top) * dpr - T.oy) / T.s];
};
const toCv = (p) => [p[0] * T.s + T.ox, p[1] * T.s + T.oy];

function setLiveMode(isLive) {
  if (!isLive) $("viewerState").hidden = true;
  else if (S.status) setTimeout(() => renderSource(S.status.capture), 0);
  live.style.display = isLive ? "block" : "none";
  cv.style.display = isLive ? "none" : "block";
  const b = $("viewerBadge");
  b.textContent = isLive ? "CANLI" : "DONDURULDU";
  b.classList.toggle("frozen", !isLive);
  $("btnFreeze").innerHTML = isLive ? "Dondur <kbd>Space</kbd>" : "Canlıya dön <kbd>Space</kbd>";
  if (isLive) { if (!live.src.includes("/api/stream")) live.src = "/api/stream?" + Date.now(); }
  else live.removeAttribute("src");  // akışı durdur, bant genişliği harcamasın
  document.querySelectorAll("#tools button[data-tool], #btnClearView").forEach((b) => (b.disabled = isLive || (b.dataset.tool === "ai" && !S.status?.model?.ready)));
}

function draw() {
  const v = cur();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!v.img) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(v.img, T.ox, T.oy, v.w * T.s, v.h * T.s);
  const dpr = T.dpr || 1;
  const cfg = S.status || {};

  // ROI ve gizlilik alanları
  const rectN = (r, color, dash) => {
    const [x, y] = toCv([r[0] * v.w, r[1] * v.h]);
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr; ctx.setLineDash(dash.map((d) => d * dpr));
    ctx.strokeRect(x, y, r[2] * v.w * T.s, r[3] * v.h * T.s); ctx.restore();
  };
  if (cfg.roi) rectN(cfg.roi, "rgba(54,179,224,.55)", [6, 4]);
  (cfg.privacy_masks || []).forEach((m) => rectN(m, "rgba(248,81,73,.6)", [2, 3]));

  // Kontur
  if (v.contour?.length) {
    const pending = isPending(v);
    ctx.save();
    ctx.beginPath();
    v.contour.forEach((p, i) => { const q = toCv(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    ctx.closePath();
    ctx.fillStyle = pending ? "rgba(54,179,224,.12)" : "rgba(255,200,61,.14)"; ctx.fill();
    // Onay bekleyen AI önerisi kesikli mavi, onaylı kontur düz sarı
    ctx.strokeStyle = pending ? "#4cc3f0" : "#ffc83d"; ctx.lineWidth = 2 * dpr;
    if (pending) ctx.setLineDash([7 * dpr, 5 * dpr]);
    ctx.stroke();
    ctx.restore();
  }
  // Kumpas çizgileri
  const mm = mmpp();
  for (const [k, l] of Object.entries(v.lines)) {
    const a = toCv(l.points[0]), b = toCv(l.points[1]);
    ctx.save();
    ctx.strokeStyle = "#5cf27a"; ctx.fillStyle = "#5cf27a"; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    for (const p of [a, b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 4 * dpr, 0, 7); ctx.fill(); }
    const len = lineLenPx(l);
    const txt = `${LINE_LABEL[k]} ${mm ? (len * mm).toFixed(1) + " mm" : Math.round(len) + " px"}`;
    ctx.font = `${13 * dpr}px system-ui, sans-serif`;
    const tx = b[0] + 8 * dpr, ty = b[1] - 6 * dpr, tw = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(tx - 3 * dpr, ty - 13 * dpr, tw + 6 * dpr, 17 * dpr);
    ctx.fillStyle = "#5cf27a"; ctx.fillText(txt, tx, ty);
    ctx.restore();
  }
  // Etkileşim önizlemeleri
  ctx.save();
  ctx.lineWidth = 1.5 * dpr;
  if (S.drag) {
    const a = toCv(S.drag.start), b = toCv(S.drag.end);
    ctx.strokeStyle = S.drag.kind === "rect" ? "#36b3e0" : "#5cf27a";
    ctx.setLineDash([5 * dpr, 4 * dpr]);
    if (S.drag.kind === "rect") ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    else { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
  }
  if (S.poly?.length) {
    ctx.strokeStyle = "#ffc83d"; ctx.fillStyle = "#ffc83d";
    ctx.beginPath();
    S.poly.forEach((p, i) => { const q = toCv(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    if (S.poly.hover) { const q = toCv(S.poly.hover); ctx.lineTo(q[0], q[1]); }
    ctx.stroke();
    S.poly.forEach((p, i) => { const q = toCv(p); ctx.beginPath(); ctx.arc(q[0], q[1], (i ? 3 : 6) * dpr, 0, 7); i ? ctx.fill() : ctx.stroke(); });
  }
  // Düzenleme fırçası (etki alanı)
  if (S.tool === "edit" && S.mouse && !S.edit && v.contour) {
    const q = toCv(S.mouse);
    ctx.strokeStyle = "rgba(76,195,240,.75)"; ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath(); ctx.arc(q[0], q[1], S.brush * dpr, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (S.calibPts) {
    ctx.fillStyle = "#36b3e0";
    S.calibPts.forEach((p) => { const q = toCv(p); ctx.beginPath(); ctx.arc(q[0], q[1], 5 * dpr, 0, 7); ctx.fill(); });
  }
  ctx.restore();
}

function setTool(t) {
  if (t && !cur().frameId) { toast("Önce görüntüyü dondurun."); return; }
  S.tool = S.tool === t ? null : t;
  S.poly = null; S.drag = null; S.free = null; S.edit = null;
  if (S.tool !== "calib") S.calibPts = null;
  if ((S.tool === "edit" || S.tool === "splice") && !cur().contour) { toast("Düzenlenecek kontur yok."); S.tool = null; }
  document.querySelectorAll("#tools button[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === S.tool));
  const hints = {
    ai: "Nodülün içine tıklayın; model o noktadaki nodülü segmente eder.",
    width: "Transvers çapı çizin: tıklayıp sürükleyin.",
    ap: "Ön-arka (AP) çapı çizin: tıklayıp sürükleyin.",
    length: "Longitudinal uzunluğu çizin: tıklayıp sürükleyin.",
    draw: "Basılı tutup sınırı çizin (bırakınca kapanır) ya da noktalarla tıklayıp ilk noktaya tıklayın / Enter. Esc iptal.",
    edit: "Konturu tutup sürükleyin; daire içindeki noktalar birlikte hareket eder. Fare tekerleği veya [ ] ile daireyi büyütüp küçültün.",
    splice: "Konturun üzerinden başlayıp yeni sınırı çizin ve yine konturun üzerinde bitirin; aradaki kısım değiştirilir.",
    calib: "Derinlik skalasında iki çentiğe tıklayın.",
    roi: "Model için görüntü alanını sürükleyerek seçin (menü/yazılar dışarıda kalsın).",
    privacy: "Hasta bilgisi alanını sürükleyerek seçin; kayıt ve akışta siyaha boyanır.",
  };
  $("hint").textContent = S.tool ? hints[S.tool] : (cur().frameId ? "Bir araç seçin ya da canlıya dönün." : "Nodülü ekrana getirip Dondur'a basın.");
  cv.style.cursor = S.tool === "edit" ? "none" : "crosshair";
  renderContourBar();
  draw();
}

// Konturu eşit aralıklı noktalara böl (düzenleme yumuşak olsun)
function resample(c, step) {
  const out = [];
  for (let i = 0; i < c.length; i++) {
    const a = c[i], b = c[(i + 1) % c.length];
    const k = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
    for (let j = 0; j < k; j++) out.push([a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k]);
  }
  return out;
}
S.brush = Math.min(120, Math.max(8, parseInt(load("brush") || "28", 10) || 28));  // ekran pikseli

cv.addEventListener("wheel", (ev) => {
  if (S.tool !== "edit") return;
  ev.preventDefault();
  setBrush(S.brush * (ev.deltaY < 0 ? 1.15 : 1 / 1.15));
}, { passive: false });
function setBrush(r) {
  S.brush = Math.round(Math.min(120, Math.max(8, r)));
  store("brush", S.brush);
  draw();
}
cv.addEventListener("mouseleave", () => { S.mouse = null; draw(); });

cv.addEventListener("mousedown", (ev) => {
  if (ev.button !== 0 || !S.tool) return;
  const p = toImg(ev);
  if (["width", "ap", "length"].includes(S.tool)) S.drag = { kind: "line", start: p, end: p };
  else if (["roi", "privacy"].includes(S.tool)) S.drag = { kind: "rect", start: p, end: p };
  else if ((S.tool === "draw" && !(S.poly && S.poly.length)) || S.tool === "splice") S.free = { pts: [p], moved: false };
  else if (S.tool === "edit" && cur().contour) {
    const dpr = T.dpr || 1, q = toCv(p);
    const pts = resample(cur().contour, 3);
    let best = -1, bd = Infinity;
    pts.forEach((c, i) => { const r = toCv(c), d = Math.hypot(r[0] - q[0], r[1] - q[1]); if (d < bd) { bd = d; best = i; } });
    if (bd > S.brush * dpr) return;
    const before = snap(cur());   // geri al için düzenleme öncesi durum
    cur().contour = pts;
    S.edit = { orig: pts.map((c) => c.slice()), grab: pts[best].slice(), start: p, before };
  }
});
cv.addEventListener("mousemove", (ev) => {
  const p = toImg(ev);
  S.mouse = p;
  if (S.drag) { S.drag.end = p; draw(); }
  else if (S.free && (ev.buttons & 1)) {
    const last = S.free.pts[S.free.pts.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) * T.s >= 3 * (T.dpr || 1)) {
      S.free.pts.push(p);
      if (S.free.pts.length > 4) S.free.moved = true;
      S.poly = S.free.pts; draw();
    }
  }
  else if (S.edit && (ev.buttons & 1)) {
    const dx = p[0] - S.edit.start[0], dy = p[1] - S.edit.start[1];
    const R = S.brush * (T.dpr || 1) / T.s, g = S.edit.grab;
    cur().contour = S.edit.orig.map((c) => {
      const w = Math.exp(-((c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2) / (R * R));
      return [c[0] + w * dx, c[1] + w * dy];
    });
    draw();
  }
  else if (S.poly) { S.poly.hover = p; draw(); }
  else if (S.tool === "edit") draw();
});
window.addEventListener("mouseup", async (ev) => {
  if (S.free) {
    const f = S.free; S.free = null;
    if (S.tool === "splice") {
      S.poly = null; S.suppressClick = true;
      if (!f.moved) { draw(); return; }
      const res = spliceContour(cur().contour, f.pts, 14 * (T.dpr || 1) / T.s);
      if (res.error) { draw(); return toast(res.error, true); }
      await applyContour(res.contour.map((q) => q.map(Math.round)), "edit");
      return;  // araç açık kalır: birden fazla bölge düzeltilebilir
    }
    if (f.moved && f.pts.length >= 8) {
      S.suppressClick = true; S.poly = null;
      await applyContour(f.pts.map((q) => q.map(Math.round)), "manual");
      setTool(null);
    } else if (S.poly === f.pts) S.poly = null;
    return;
  }
  if (S.edit) {
    const before = S.edit.before;
    S.edit = null;
    const moved = cur().contour.some((c, i) => Math.hypot(c[0] - S_editOrig(i, before)[0], c[1] - S_editOrig(i, before)[1]) > 0.5);
    if (!moved) { cur().contour = before.contour; draw(); return; }
    await applyContour(cur().contour.map((q) => q.map(Math.round)), "edit", { before });
    return;
  }
  if (!S.drag) return;
  const d = S.drag; S.drag = null;
  const v = cur();
  if (d.kind === "line") {
    if (Math.hypot(d.end[0] - d.start[0], d.end[1] - d.start[1]) < 4) return draw();
    v.lines[S.tool] = { points: [d.start.map(Math.round), d.end.map(Math.round)], source: "manual" };
    S.manualDims[S.tool] = null;
    // Aynı ölçüm diğer görünümde varsa kaldır (tek kaynak)
    VIEWS.forEach((o) => o !== S.view && delete S.views[o].lines[S.tool]);
    draw(); renderDims(); markDots();
  } else {
    const x0 = Math.max(0, Math.min(d.start[0], d.end[0])) / v.w, y0 = Math.max(0, Math.min(d.start[1], d.end[1])) / v.h;
    const x1 = Math.min(v.w, Math.max(d.start[0], d.end[0])) / v.w, y1 = Math.min(v.h, Math.max(d.start[1], d.end[1])) / v.h;
    if (x1 - x0 < 0.02 || y1 - y0 < 0.02) return draw();
    const r = [x0, y0, x1 - x0, y1 - y0].map((n) => +n.toFixed(4));
    try {
      if (S.tool === "roi") await api("/api/config", { json: { roi: r } });
      else await api("/api/config", { json: { privacy_masks: [...(S.status.privacy_masks || []), r] } });
      await refreshStatus();
      toast(S.tool === "roi" ? "Görüntü alanı kaydedildi." : "Gizlilik maskesi eklendi (sonraki dondurmalardan itibaren uygulanır).");
    } catch (e) { toast(e.message, true); }
    setTool(null);
  }
});
cv.addEventListener("click", async (ev) => {
  if (S.suppressClick) { S.suppressClick = false; return; }
  const p = toImg(ev);
  if (S.tool === "ai") return runSegment(p);
  if (S.tool === "draw") {
    if (!S.poly) S.poly = [];
    if (S.poly.length >= 3) {
      const f = toCv(S.poly[0]), q = toCv(p);
      if (Math.hypot(f[0] - q[0], f[1] - q[1]) < 10 * (T.dpr || 1)) return finishPoly();
    }
    S.poly.push(p.map(Math.round));
    return draw();
  }
  if (S.tool === "calib") {
    S.calibPts = [...(S.calibPts || []), p];
    draw();
    if (S.calibPts.length === 2) {
      const mm = parseFloat(prompt("İki nokta arası gerçek mesafe (mm):", "10"));
      const pts = S.calibPts;
      S.calibPts = null; setTool(null);
      if (!(mm > 0)) return;
      try {
        await api("/api/calibration", { json: { points: pts, distance_mm: mm } });
        await refreshStatus(); renderSettings(); renderDims(); draw();
        toast(`Kalibrasyon: ${mmpp().toFixed(4)} mm/px. Ayarlar'dan ön ayar olarak kaydedebilirsiniz.`);
      } catch (e) { toast(e.message, true); }
    }
  }
});
cv.addEventListener("dblclick", (ev) => { if (S.tool === "draw") { ev.preventDefault(); finishPoly(); } });

async function finishPoly() {
  const pts = S.poly;
  S.poly = null;
  if (!pts || pts.length < 3) { draw(); return toast("En az 3 nokta gerekli."); }
  await applyContour(pts, "manual");
  setTool(null);
}

// ------------------------------------------------- kontur durumu / geçmiş
const isPending = (v) => !!(v.contour && v.aiContour && !v.confirmed);
const snap = (v) => ({ contour: v.contour, ai: v.ai, confirmed: v.confirmed, aiDice: v.aiDice });
function pushHistory(v) {
  v.hist.push(snap(v));
  if (v.hist.length > 60) v.hist.shift();
  v.fut = [];
}

// how: "manual" (sıfırdan çizim, radyoloğun kendi çizimi → onaylı)
//      "edit"   (düzeltme; onay durumu korunur)
//      "ai"     (AI önerisine geri dönüş)
async function applyContour(contour, how, { history = true, confirmed, before } = {}) {
  const v = cur();
  const prev = before || snap(v);
  const conf = confirmed !== undefined ? confirmed : how === "manual" ? true : how === "ai" ? false : prev.confirmed;
  try {
    const r = await api("/api/contour", { json: { frame_id: v.frameId, contour, source: how, confirmed: conf } });
    if (history) { v.hist.push(prev); if (v.hist.length > 60) v.hist.shift(); v.fut = []; }
    // AI önerisine dönüşte aynı nesne korunur (sonraki geri almalarda tanınsın)
    v.contour = how === "ai" && contour === v.aiContour ? v.aiContour : (r.contour || contour);
    v.confirmed = conf;
    v.aiDice = r.ai_dice ?? null;
    if (how === "manual") v.ai = v.aiContour ? { ...v.ai, edited: true, redrawn: true } : { backend: "manuel", edited: true };
    else if (how === "edit") v.ai = v.ai ? { ...v.ai, edited: true } : { backend: "manuel", edited: true };
    autoLines(v, r.metrics_px);
    afterContourChange();
  } catch (e) { toast(e.message, true); v.contour = prev.contour; draw(); }
}

// Sürükleme başında konturun yeniden örneklenmiş hali (hareket var mı?)
function S_editOrig(i, before) {
  const o = S._editOrigCache && S._editOrigCache.src === before.contour ? S._editOrigCache.pts
    : (S._editOrigCache = { src: before.contour, pts: resample(before.contour, 3) }).pts;
  return o[i] || [Infinity, Infinity];
}

function afterContourChange() {
  renderAiInfo(); draw(); renderDims(); markDots(); renderContourBar();
}

async function confirmContour() {
  const v = cur();
  if (!v.contour) return;
  try {
    await api("/api/contour/confirm", { json: { frame_id: v.frameId, confirmed: true } });
    pushHistory(v);
    v.confirmed = true;
    setTool(null);
    afterContourChange();
    toast("Kontur onaylandı.");
  } catch (e) { toast(e.message, true); }
}

async function restore(state) {
  const v = cur();
  if (!state.contour) {
    await api("/api/contour", { json: { frame_id: v.frameId, contour: [] } });
    v.contour = null; v.confirmed = false; v.aiDice = null; v.ai = state.ai;
    for (const k of Object.keys(v.lines)) if (v.lines[k].source === "auto") delete v.lines[k];
    afterContourChange();
    return;
  }
  const how = state.contour === v.aiContour ? "ai"
    : (state.ai?.redrawn || state.ai?.backend === "manuel") ? "manual" : "edit";
  await applyContour(state.contour, how, { history: false, confirmed: state.confirmed });
  v.ai = state.ai;
  afterContourChange();
}

async function undo() {
  const v = cur();
  if (!v.frameId || !v.hist.length) return toast("Geri alınacak işlem yok.");
  const prev = v.hist.pop();
  v.fut.push(snap(v));
  await restore(prev);
}
async function redo() {
  const v = cur();
  if (!v.frameId || !v.fut.length) return toast("Yinelenecek işlem yok.");
  const next = v.fut.pop();
  v.hist.push(snap(v));
  await restore(next);
}

async function deleteContour() {
  const v = cur();
  if (!v.contour) return;
  pushHistory(v);
  await restore({ contour: null, ai: v.ai });
  toast("Kontur silindi (Ctrl+Z ile geri alabilirsiniz).");
}

// Kısmi düzeltme: konturun üzerinde başlayıp biten çizgi, iki uç arasındaki
// kontur parçasının yerine geçer. Hangi parçanın değişeceği, çizgiye daha
// yakın olan parçadır.
function spliceContour(contour, stroke, tol) {
  const C = resample(contour, 2);
  const n = C.length;
  const near = (p) => {
    let bi = -1, bd = Infinity;
    C.forEach((c, i) => { const d = Math.hypot(c[0] - p[0], c[1] - p[1]); if (d < bd) { bd = d; bi = i; } });
    return [bi, bd];
  };
  const [i0, d0] = near(stroke[0]);
  const [i1, d1] = near(stroke[stroke.length - 1]);
  if (d0 > tol || d1 > tol) return { error: "Çizgiyi konturun üzerinde başlatıp yine konturun üzerinde bitirin." };
  const S_ = stroke.map((p) => [p[0], p[1]]);
  if (i0 === i1 || Math.min((i1 - i0 + n) % n, (i0 - i1 + n) % n) < 3) {
    // Aynı noktada başlayıp biten çizgi: konturu o noktadan genişletir/daraltır
    return { contour: [...C.slice(0, i0 + 1), ...S_, ...C.slice(i0 + 1)] };
  }
  const arc = (a, b) => { const out = []; for (let i = a; ; i = (i + 1) % n) { out.push(C[i]); if (i === b) break; } return out; };
  const meanDist = (pts) => {
    let t = 0;
    for (const p of pts) { let m = Infinity; for (const q of S_) m = Math.min(m, Math.hypot(p[0] - q[0], p[1] - q[1])); t += m; }
    return t / Math.max(1, pts.length);
  };
  const fwd = arc(i0, i1);   // i0 → i1 yönündeki parça
  const bwd = arc(i1, i0);   // i1 → i0 yönündeki parça
  // fwd değişirse: çizgi (i0→i1) + bwd'nin içi; bwd değişirse: fwd + ters çizgi
  const replaceFwd = meanDist(fwd) <= meanDist(bwd);
  const out = replaceFwd ? [...S_, ...bwd.slice(1, -1)] : [...fwd, ...S_.slice().reverse().slice(1, -1)];
  if (out.length < 3) return { error: "Kontur düzeltilemedi; tekrar deneyin." };
  return { contour: out };
}

// Görüntü üstündeki kontur eylem çubuğu
function renderContourBar() {
  const v = cur(), bar = $("contourBar");
  const show = !!(v.frameId && v.contour);
  bar.hidden = !show;
  if (!show) return;
  const pending = isPending(v);
  bar.classList.toggle("pending", pending);
  const st = $("cbState");
  if (pending) st.textContent = v.ai?.edited ? "AI önerisi (düzeltildi) · onay bekliyor" : "AI önerisi · kontrol edip onaylayın";
  else if (v.ai?.redrawn) st.textContent = "✓ Elle çizildi (AI önerisi yerine)";
  else if (v.aiContour) st.textContent = v.ai?.edited ? "✓ Düzeltildi ve onaylandı" : "✓ AI önerisi onaylandı";
  else st.textContent = "✓ Elle çizildi";
  $("cbConfirm").hidden = !pending;
  $("cbAi").hidden = !S.status?.model?.ready;
  $("cbUndo").disabled = !v.hist.length;
  $("cbRedo").disabled = !v.fut.length;
  bar.querySelectorAll("[data-cb]").forEach((b) => b.classList.toggle("active", b.dataset.cb === S.tool));
}

function autoLines(v, m) {
  // Transvers: genişlik + AP; longitudinal: uzunluk (en uzun eksen)
  const set = (k, pts) => {
    v.lines[k] = { points: pts, source: "auto" };
    S.manualDims[k] = null;
    VIEWS.forEach((o) => S.views[o] !== v && delete S.views[o].lines[k]);
  };
  if (S.view === "transverse") { set("width", m.width_line); set("ap", m.height_line); }
  else set("length", m.feret_line);
}

async function runSegment(p) {
  const v = cur();
  viewerMsg("Segmentasyon çalışıyor…");
  try {
    const r = await api("/api/segment", { json: { frame_id: v.frameId, click: p.map(Math.round) } });
    if (r.empty) { toast(r.message); return; }
    pushHistory(v);
    v.contour = r.contour;
    v.aiContour = r.contour;       // geri al ile bu öneriye dönülebilir
    v.confirmed = false;
    v.aiDice = 1;
    v.ai = { backend: r.backend, confidence: r.confidence, ms: r.ms, edited: false };
    autoLines(v, r.metrics_px);
    setTool("edit");               // doğrudan düzenlemeye hazır
    afterContourChange();
    $("hint").textContent = "AI konturu çizdi. Kontrol edin: gerekirse sürükleyerek düzeltin (E), bir bölümünü yeniden çizin (K) ya da baştan çizin (D). Uygunsa Enter ile onaylayın.";
  } catch (e) { toast(e.message, true); }
  finally { viewerMsg(""); }
}

function renderAiInfo() {
  const parts = VIEWS.map((k) => {
    const a = S.views[k].ai;
    if (!a) return null;
    const name = k === "transverse" ? "Transvers" : "Longitudinal";
    if (a.backend === "manuel") return `${name}: elle çizildi`;
    const v = S.views[k];
    const fit = a.edited && v.aiDice != null ? ` · AI ile uyum ${(v.aiDice * 100).toFixed(0)}%` : "";
    const st = isPending(v) ? " · onay bekliyor" : v.contour ? " · onaylı" : "";
    return `${name}: ${a.backend === "demo" ? "DEMO" : "UNet++"} · güven ${(a.confidence * 100).toFixed(0)}%${a.redrawn ? " · elle yeniden çizildi" : a.edited ? " · düzeltildi" : ""}${fit}${st}`;
  }).filter(Boolean);
  $("aiInfo").textContent = parts.join("  |  ");
}

function markDots() {
  $("dotTransverse").classList.toggle("on", !!S.views.transverse.frameId);
  $("dotLongitudinal").classList.toggle("on", !!S.views.longitudinal.frameId);
  updateSaveEnabled();
}

// --------------------------------------------------------- dondur / görünüm
async function loadFrame(view, info) {
  const img = new Image();
  img.src = `/api/frame/${info.frame_id}.png`;
  await img.decode();
  S.views[view] = { ...emptyView(), frameId: info.frame_id, w: info.width, h: info.height, img };
}

async function freezeToggle() {
  const v = cur();
  if (v.frameId) {
    const has = v.contour || Object.keys(v.lines).length;
    if (has && !confirm("Bu görünümdeki kontur ve ölçümler silinecek. Canlıya dönülsün mü?")) return;
    clearViewData(S.view, true);
    setLiveMode(true); setTool(null);
    return;
  }
  try {
    const info = await api("/api/freeze", { method: "POST" });
    await loadFrame(S.view, info);
    setLiveMode(false); fit(); setTool(S.status?.model?.ready ? "ai" : "draw"); markDots();
  } catch (e) { toast(e.message, true); }
}

function clearViewData(view, dropFrame) {
  const v = S.views[view];
  for (const k of Object.keys(v.lines)) S.manualDims[k] = null;
  if (dropFrame) S.views[view] = emptyView();
  else { pushHistory(v); v.contour = null; v.lines = {}; v.confirmed = false; v.aiDice = null; }
  if (v.frameId && !dropFrame) api("/api/contour", { json: { frame_id: v.frameId, contour: [] } }).catch(() => {});
  renderAiInfo(); renderDims(); markDots(); draw(); renderContourBar();
}

function switchView(view) {
  S.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $("annView").textContent = VIEW_TR[view];
  const frozen = !!cur().frameId;
  setLiveMode(!frozen);
  fit();
  setTool(frozen && S.tool ? S.tool : null);
  renderContourBar();
}

async function uploadFile(file) {
  try {
    const r = await fetch("/api/upload", { method: "POST", body: file });
    const info = await r.json();
    if (!r.ok) throw new Error(info.error);
    await loadFrame(S.view, info);
    setLiveMode(false); fit(); setTool(S.status?.model?.ready ? "ai" : "draw"); markDots();
    if (info.dicom_mm_per_px && confirm(`DICOM piksel aralığı bulundu: ${info.dicom_mm_per_px.toFixed(4)} mm/px. Kalibrasyon olarak kullanılsın mı?`)) {
      await api("/api/calibration", { json: { mm_per_px: info.dicom_mm_per_px } });
      await refreshStatus(); renderDims(); draw();
    }
  } catch (e) { toast(e.message, true); }
}

// --------------------------------------------------------------- kayıt
function updateSaveEnabled() {
  const s = S.lastScore?.score;
  const anyView = VIEWS.some((k) => S.views[k].frameId);
  const pending = VIEWS.filter((k) => isPending(S.views[k]));
  $("btnSave").disabled = !(anyView && s?.level && $("approved").checked && $("radiologist").value.trim() && !pending.length);
  $("saveBlock").textContent = pending.length
    ? `${pending.map((k) => VIEW_TR[k]).join(" ve ")} kesitindeki AI konturu onay bekliyor.` : "";
}

async function save() {
  const d = currentDims();
  const views = {};
  for (const k of VIEWS) {
    const v = S.views[k];
    if (!v.frameId) continue;
    views[k] = {
      frame_id: v.frameId,
      lines: Object.entries(v.lines).map(([n, l]) => ({ label: LINE_LABEL[n], points: l.points })),
    };
  }
  const body = {
    views, features: features(),
    measurements: { width_mm: d.width, ap_mm: d.ap, length_mm: d.length, manual_override: { ...S.manualDims } },
    study_no: $("studyNo").value, radiologist: $("radiologist").value,
    location: { side: $("locSide").value, pole: $("locPole").value },
    notes: $("notes").value, approved: $("approved").checked,
  };
  $("btnSave").disabled = true;
  try {
    const r = await api("/api/cases", { json: body });
    store("radiologist", body.radiologist);
    $("report").textContent = r.report;
    $("report").hidden = false;
    toast(`Kaydedildi: ${r.case_id}`);
  } catch (e) { toast(e.message, true); updateSaveEnabled(); }
}

function resetAll() {
  VIEWS.forEach((k) => (S.views[k] = emptyView()));
  S.manualDims = {}; S.shapeTouched = false;
  $("tiradsForm").reset();
  ["studyNo", "notes"].forEach((id) => ($(id).value = ""));
  $("locSide").value = ""; $("locPole").value = "";
  $("approved").checked = false;
  $("report").hidden = true;
  $("tag_shape").textContent = "";
  switchView("transverse"); renderAiInfo(); renderDims(); markDots();
}

// ------------------------------------------------------- etiketleme modu
const VIEW_TR = { transverse: "Transvers", longitudinal: "Longitudinal" };

function setMode(m) {
  const prev = S.mode;
  S.mode = m;
  document.body.classList.toggle("mode-annot", m === "annot");
  document.body.classList.toggle("mode-biopsy", m === "biopsy");
  $("modeClinical").classList.toggle("active", m === "clinical");
  $("modeAnnot").classList.toggle("active", m === "annot");
  $("modeBiopsy").classList.toggle("active", m === "biopsy");
  $("viewerBadge").dataset.mode = m;
  if (m !== "biopsy") store("mode", m);   // biyopsi modu açılışta hatırlanmaz
  if (m === "annot") refreshDataset();
  if (m === "biopsy" && prev !== "biopsy") enterBiopsy();
  else if (m !== "biopsy" && prev === "biopsy") exitBiopsy();
}

// --------------------------------------------------------- biyopsi rehberi
const B = { es: null, st: null, pick: null, raf: 0, snapBusy: false };
const ov = $("ov"), octx = ov.getContext("2d");
const BX_STATUS = {
  searching: "İğne aranıyor", tracking: "İğne izleniyor",
  on_track: "Rotada", off_track: "Sapma var", in_target: "Uç hedefte", passed: "Uç hedefi geçti",
};
const BX_COLOR = {
  searching: "#8b98a8", tracking: "#4cc3f0", on_track: "#3fb950",
  off_track: "#f2a93b", in_target: "#7ee787", passed: "#f85149",
};

async function enterBiopsy() {
  setTool(null);
  setLiveMode(true);                       // biyopsi her zaman canlı görüntüde
  $("cv").style.display = "none";
  try { await api("/api/biopsy/start", { method: "POST" }); }
  catch (e) { toast(e.message, true); }
  if (B.es) B.es.close();
  B.es = new EventSource("/api/biopsy/events");
  B.es.onmessage = (ev) => { try { B.st = JSON.parse(ev.data); } catch (_) { return; } onBiopsyState(); };
  B.es.onerror = () => { $("bxPerf").textContent = "Bağlantı yeniden kuruluyor…"; };
  const p = B.st?.plan;
  if (!p || !p.target) setPick("target");
  window.addEventListener("resize", drawOverlay);
}

async function exitBiopsy() {
  setPick(null);
  if (B.es) { B.es.close(); B.es = null; }
  B.st = null;
  try { await api("/api/biopsy/stop", { method: "POST" }); } catch (_) {}
  octx.clearRect(0, 0, ov.width, ov.height);
  switchView(S.view);                      // donmuş kare varsa geri getir
}

function setPick(what) {
  B.pick = what;
  ov.classList.toggle("picking", !!what);
  $("bxPickTarget").classList.toggle("active", what === "target");
  $("bxPickEntry").classList.toggle("active", what === "entry");
  const h = $("bxHint");
  h.hidden = !what;
  if (what === "target") h.textContent = "Hedefe (nodülün iğnenin ulaşacağı noktasına) tıklayın";
  else if (what === "entry") h.textContent = "İğnenin görüntüye gireceği noktaya tıklayın (prob kenarı) · atlamak için Esc";
}

// Canlı görüntü ile aynı "contain" yerleşimi
function ovTransform() {
  const r = $("viewer").getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const fs = B.st?.frame_size || S.status?.capture?.resolution;
  if (!fs) return null;
  const s = Math.min(r.width / fs[0], r.height / fs[1]);
  return { s: s * dpr, ox: (r.width - fs[0] * s) / 2 * dpr, oy: (r.height - fs[1] * s) / 2 * dpr, dpr, r };
}

ov.addEventListener("click", async (ev) => {
  if (!B.pick) return;
  const t = ovTransform();
  if (!t) return toast("Görüntü bekleniyor.");
  const p = [((ev.clientX - t.r.left) * t.dpr - t.ox) / t.s, ((ev.clientY - t.r.top) * t.dpr - t.oy) / t.s];
  const fs = B.st.frame_size;
  if (p[0] < 0 || p[1] < 0 || p[0] > fs[0] || p[1] > fs[1]) return;
  const what = B.pick;
  try {
    await api("/api/biopsy/plan", { json: what === "target" ? { target: p, radius_mm: +$("bxRadius").value || 5 } : { entry: p } });
    if (what === "target" && !B.st?.plan?.entry) setPick("entry");
    else setPick(null);
  } catch (e) { toast(e.message, true); }
});

function fmt(v, unit, digits = 1) { return v == null || Number.isNaN(v) ? "–" : `${v.toFixed(digits)} ${unit}`; }

function onBiopsyState() {
  const st = B.st;
  if (!st || !st.active) return;
  const g = st.guide || {}, unit = g.unit || "mm";
  let status = g.status || "searching";
  const el = $("bxStatus");
  el.dataset.s = status;
  let txt = BX_STATUS[status] || status;
  if (status === "on_track" && g.tip_to_target != null) txt += ` · hedefe ${g.tip_to_target.toFixed(1)} ${unit}`;
  if (status === "off_track" && g.miss != null)
    txt += ` · ${g.miss.toFixed(1)} ${unit} (hedef hattın ${g.target_side === "deeper" ? "altında" : "üstünde"})`;
  if (status === "tracking") txt += " · hedef seçin (T)";
  el.textContent = txt;
  const warns = [];
  if (!st.mm_per_px) warns.push("Kalibrasyon yok: mesafeler piksel cinsinden.");
  if (st.target_lost) warns.push("Hedef takibi zayıfladı; gerekirse hedefi yeniden seçin (T).");
  if (st.error) warns.push("Takip hatası: " + st.error);
  $("bxWarn").textContent = warns.join(" ");
  $("bxTipT").textContent = fmt(g.tip_to_target, unit);
  $("bxMiss").textContent = fmt(g.miss, unit);
  $("bxRemain").textContent = g.remaining != null && g.remaining > 0 ? fmt(g.remaining, unit) : (g.remaining != null ? "0" : "–");
  $("bxAngle").textContent = g.needle_angle_deg != null ? `${g.needle_angle_deg.toFixed(0)}°` : "–";
  $("bxAngDiff").textContent = g.angle_diff_deg != null ? `${g.angle_diff_deg > 0 ? "+" : ""}${g.angle_diff_deg.toFixed(0)}°` : "–";
  $("bxDepth").textContent = fmt(g.tip_depth, unit);
  $("bxPerf").textContent = `Takip: ${st.fps ?? "–"} kare/sn · ${st.proc_ms != null ? st.proc_ms.toFixed(0) : "–"} ms`;
  if (!B.raf) B.raf = requestAnimationFrame(() => { B.raf = 0; drawOverlay(); });
}

function drawOverlay() {
  const r = $("viewer").getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (ov.width !== Math.round(r.width * dpr) || ov.height !== Math.round(r.height * dpr)) {
    ov.width = Math.round(r.width * dpr); ov.height = Math.round(r.height * dpr);
  }
  octx.setTransform(1, 0, 0, 1, 0, 0);
  octx.clearRect(0, 0, ov.width, ov.height);
  const st = B.st, t = ovTransform();
  if (!st || !st.active || !t || S.status?.capture?.status !== "açık") return;
  const P = (p) => [p[0] * t.s + t.ox, p[1] * t.s + t.oy];
  const g = st.guide || {}, plan = st.plan || {}, nd = st.needle;
  const R = (g.target_radius_px || 25) * t.s;
  const status = g.status || "searching", col = BX_COLOR[status] || "#ccc";
  const mm = st.mm_per_px;
  octx.lineCap = "round";

  // Planlanan yol: koridor + kesikli çizgi + hedefe kalan cm işaretleri
  if (plan.target && plan.entry) {
    const a = P(plan.entry), b = P(plan.target);
    octx.save();
    octx.strokeStyle = "rgba(76,195,240,.13)"; octx.lineWidth = 2 * R;
    octx.beginPath(); octx.moveTo(a[0], a[1]); octx.lineTo(b[0], b[1]); octx.stroke();
    octx.strokeStyle = "rgba(76,195,240,.85)"; octx.lineWidth = 1.5 * dpr; octx.setLineDash([8 * dpr, 6 * dpr]);
    octx.beginPath(); octx.moveTo(a[0], a[1]); octx.lineTo(b[0], b[1]); octx.stroke();
    octx.setLineDash([]);
    if (mm) {
      const L = Math.hypot(plan.target[0] - plan.entry[0], plan.target[1] - plan.entry[1]);
      const ux = (plan.entry[0] - plan.target[0]) / L, uy = (plan.entry[1] - plan.target[1]) / L;
      octx.fillStyle = "rgba(191,233,251,.95)"; octx.font = `${11 * dpr}px system-ui, sans-serif`;
      for (let cm = 1; cm * 10 / mm < L; cm++) {
        const d = cm * 10 / mm, q = P([plan.target[0] + ux * d, plan.target[1] + uy * d]);
        const nx = -uy * 7 * dpr, ny = ux * 7 * dpr;
        octx.strokeStyle = "rgba(191,233,251,.9)"; octx.lineWidth = 1.5 * dpr;
        octx.beginPath(); octx.moveTo(q[0] - nx, q[1] - ny); octx.lineTo(q[0] + nx, q[1] + ny); octx.stroke();
        octx.fillText(`${cm}`, q[0] + nx * 1.6, q[1] + ny * 1.6);
      }
    }
    // giriş noktası
    octx.fillStyle = "rgba(76,195,240,.9)";
    octx.beginPath(); octx.arc(a[0], a[1], 4 * dpr, 0, 7); octx.fill();
    octx.restore();
  }
  // Hedef
  if (plan.target) {
    const c = P(plan.target), tc = st.target_lost ? "#f2a93b" : status === "in_target" ? "#7ee787" : "#4cc3f0";
    octx.save();
    octx.strokeStyle = tc; octx.lineWidth = 2 * dpr;
    octx.beginPath(); octx.arc(c[0], c[1], R, 0, 7); octx.stroke();
    octx.beginPath();
    octx.moveTo(c[0] - R - 6 * dpr, c[1]); octx.lineTo(c[0] - R * 0.4, c[1]);
    octx.moveTo(c[0] + R * 0.4, c[1]); octx.lineTo(c[0] + R + 6 * dpr, c[1]);
    octx.moveTo(c[0], c[1] - R - 6 * dpr); octx.lineTo(c[0], c[1] - R * 0.4);
    octx.moveTo(c[0], c[1] + R * 0.4); octx.lineTo(c[0], c[1] + R + 6 * dpr);
    octx.stroke();
    octx.restore();
  }
  // İğne + tahmini yol
  if (nd && nd.found) {
    const e = P(nd.entry), tp = P(nd.tip);
    const ux = tp[0] - e[0], uy = tp[1] - e[1], L = Math.hypot(ux, uy) || 1;
    const roi = st.roi_px ? [P([st.roi_px[0], st.roi_px[1]]), P([st.roi_px[2], st.roi_px[3]])] : null;
    let far = [tp[0] + ux / L * 4000, tp[1] + uy / L * 4000];
    if (roi) {  // görüntü alanında kes
      let k = 4000;
      if (ux / L > 1e-6) k = Math.min(k, (roi[1][0] - tp[0]) / (ux / L));
      if (ux / L < -1e-6) k = Math.min(k, (roi[0][0] - tp[0]) / (ux / L));
      if (uy / L > 1e-6) k = Math.min(k, (roi[1][1] - tp[1]) / (uy / L));
      far = [tp[0] + ux / L * Math.max(0, k), tp[1] + uy / L * Math.max(0, k)];
    }
    octx.save();
    octx.strokeStyle = col; octx.globalAlpha = 0.85; octx.lineWidth = 1.5 * dpr; octx.setLineDash([5 * dpr, 6 * dpr]);
    octx.beginPath(); octx.moveTo(tp[0], tp[1]); octx.lineTo(far[0], far[1]); octx.stroke();
    octx.setLineDash([]); octx.globalAlpha = 1;
    octx.lineWidth = 3 * dpr;
    octx.shadowColor = "rgba(0,0,0,.7)"; octx.shadowBlur = 4 * dpr;
    octx.beginPath(); octx.moveTo(e[0], e[1]); octx.lineTo(tp[0], tp[1]); octx.stroke();
    octx.beginPath(); octx.arc(tp[0], tp[1], 7 * dpr, 0, 7); octx.stroke();
    octx.shadowBlur = 0;
    if (g.tip_to_target != null) {
      const txt = `${g.tip_to_target.toFixed(1)} ${g.unit}`;
      octx.font = `600 ${14 * dpr}px system-ui, sans-serif`;
      const tw = octx.measureText(txt).width, tx = tp[0] + 12 * dpr, ty = tp[1] - 12 * dpr;
      octx.fillStyle = "rgba(0,0,0,.7)"; octx.fillRect(tx - 4 * dpr, ty - 15 * dpr, tw + 8 * dpr, 20 * dpr);
      octx.fillStyle = col; octx.fillText(txt, tx, ty);
    }
    octx.restore();
  }
}

async function biopsySnapshot() {
  if (B.snapBusy) return;
  B.snapBusy = true;
  try {
    const r = await api("/api/biopsy/snapshot", { json: { study_no: $("bxStudy").value, operator: $("radiologist").value } });
    const g = B.st?.guide || {};
    toast(`Kaydedildi: ${r.name} · ${BX_STATUS[g.status] || ""}${g.tip_to_target != null ? " · uç–hedef " + g.tip_to_target.toFixed(1) + " " + g.unit : ""}`);
  } catch (e) { toast(e.message, true); }
  finally { B.snapBusy = false; }
}

async function refreshDataset() {
  try {
    const r = await api("/api/dataset");
    const st = r.stats;
    const v = st.views || {};
    $("dsStats").innerHTML =
      `Veri seti: <b>${st.images}</b> görüntü · <b>${st.patients}</b> hasta · <b>${st.nodules}</b> nodül` +
      `<br>Transvers ${v.transverse || 0} · Longitudinal ${v.longitudinal || 0} · Negatif ${st.negatives}` +
      (st.ai_vs_final_dice_mean != null ? `<br>AI–radyolog uyumu (Dice): <b>${st.ai_vs_final_dice_mean.toFixed(3)}</b> (${st.with_ai} görüntü)` : "");
    const box = $("dsRecent"); box.innerHTML = "";
    for (const it of r.recent) {
      const row = document.createElement("div"); row.className = "item";
      row.innerHTML = `<span>${it.path.split("/").slice(0, 2).join(" / ")} · ${VIEW_TR[it.view] || it.view}${it.negative ? " · negatif" : ""}</span>`;
      const del = document.createElement("button"); del.type = "button"; del.className = "ghost"; del.textContent = "Sil";
      del.onclick = async () => {
        if (!confirm("Bu etiket veri setinden çıkarılsın mı? (Dosya silinmez, işaretlenir.)")) return;
        try { await api("/api/dataset/delete", { json: { path: it.path } }); refreshDataset(); } catch (e) { toast(e.message, true); }
      };
      row.appendChild(del); box.appendChild(row);
    }
    if (!r.recent.length) box.innerHTML = '<span class="muted">Henüz kayıt yok.</span>';
  } catch (e) { $("dsStats").textContent = e.message; }
}

async function saveAnnotation() {
  const v = cur();
  if (!v.frameId) return toast("Önce görüntüyü dondurun.");
  const negative = $("annNegative").checked;
  if (!negative && !v.contour) return toast("Konturu çizin ya da 'nodül yok' işaretleyin.", true);
  const patient = $("annPatient").value.trim(), annotator = $("annotator").value.trim();
  if (!patient) { $("annPatient").focus(); return toast("Hasta takma kodu gerekli.", true); }
  if (!annotator) { $("annotator").focus(); return toast("Etiketleyen adı gerekli.", true); }
  const f = features();
  const hasF = Object.values(f).some((x) => (Array.isArray(x) ? x.length : x));
  const d = currentDims();
  $("btnAnnSave").disabled = true;
  try {
    const r = await api("/api/annotations", { json: {
      frame_id: v.frameId, patient_id: patient, nodule_id: $("annNodule").value.trim() || "N1",
      view: S.view, annotator, negative, features: hasF ? f : null,
      measurements: { width_mm: d.width, ap_mm: d.ap, length_mm: d.length },
    } });
    store("annotator", annotator);
    toast(`Kaydedildi: ${r.path.split("/").slice(0, 2).join(" / ")} · ${VIEW_TR[S.view]} (toplam ${r.stats.images})`);
    $("annNegative").checked = false;
    clearViewData(S.view, true);
    setLiveMode(true); setTool(null);
    refreshDataset();
  } catch (e) { toast(e.message, true); }
  finally { $("btnAnnSave").disabled = false; }
}

function nextNodule() {
  const m = /^(.*?)(\d+)$/.exec($("annNodule").value.trim());
  $("annNodule").value = m ? m[1] + (parseInt(m[2], 10) + 1) : "N2";
  VIEWS.forEach((k) => (S.views[k] = emptyView()));
  S.manualDims = {}; S.shapeTouched = false;
  $("tiradsForm").reset(); $("tag_shape").textContent = "";
  switchView("transverse"); renderAiInfo(); renderDims(); markDots();
  toast(`Nodül: ${$("annNodule").value}`);
}

async function exportDataset() {
  if (!confirm("Etiketler eğitim formatına aktarılsın mı? (data/export klasörü yeniden oluşturulur; önceki hasta bölmesi korunur.)")) return;
  $("exportOut").textContent = "Aktarılıyor…";
  try {
    const r = await api("/api/dataset/export", { json: { nnunet_id: 501 } });
    $("exportOut").textContent = `${r.images} görüntü → ${r.path} · eğitim ${r.by_split.train || 0} / doğrulama ${r.by_split.val || 0} / test ${r.by_split.test || 0} (hasta: ${r.patients.train}/${r.patients.val}/${r.patients.test})`;
  } catch (e) { $("exportOut").textContent = e.message; }
}

// ------------------------------------------------------------ ayarlar
function renderSettings() {
  const st = S.status; if (!st) return;
  const src = st.capture.source;
  $("srcType").value = src.type; $("srcIndex").value = src.index ?? 0; $("srcPath").value = src.path || "";
  $("srcW").value = src.width || 1920; $("srcH").value = src.height || 1080;
  $("srcFourcc").value = (src.fourcc || "auto").toUpperCase() === "AUTO" ? "auto" : src.fourcc.toUpperCase();
  syncSourceFields();
  $("devLog").textContent = (st.capture.log || []).join("\n") || "–";
  const cal = st.calibration;
  $("calManual").value = cal.mm_per_px ? cal.mm_per_px.toFixed(5) : "";
  const pr = $("calPresets"); pr.innerHTML = "";
  for (const [name, v] of Object.entries(cal.presets || {})) {
    const el = document.createElement("span");
    el.className = "preset" + (cal.active_preset === name ? " active" : "");
    const a = document.createElement("button"); a.type = "button"; a.textContent = `${name} (${v.toFixed(4)})`;
    a.onclick = () => calCall({ activate: name });
    const x = document.createElement("button"); x.type = "button"; x.textContent = "×"; x.title = "Sil";
    x.onclick = () => confirm(`'${name}' silinsin mi?`) && calCall({ delete: name });
    el.append(a, x); pr.appendChild(el);
  }
  $("areaInfo").textContent = `ROI: ${st.roi ? "tanımlı" : "tüm kare"} · Gizlilik maskesi: ${st.privacy_masks.length}`;
  const m = st.model;
  $("mdlStatus").textContent = m.ready ? `Yüklü: ${m.backend} · ${m.device}${m.is_demo ? " · DEMO, klinik değil" : ""}` : (m.error || "Yüklü değil");
}

async function loadModelForm() {
  const c = await api("/api/config");
  const m = c.model;
  $("mdlPath").value = m.path || ""; $("mdlBackend").value = m.backend; $("mdlEncoder").value = m.encoder || "";
  $("mdlCin").value = String(m.in_channels); $("mdlClasses").value = m.classes;
  $("mdlSize").value = (m.input_size || [512, 512]).join("x"); $("mdlNorm").value = m.normalize; $("mdlThr").value = m.threshold;
}

async function calCall(body) {
  try { await api("/api/calibration", { json: body }); await refreshStatus(); renderSettings(); renderDims(); draw(); }
  catch (e) { toast(e.message, true); }
}

function syncSourceFields() {
  const t = $("srcType").value, manual = $("srcFourcc").value !== "auto";
  document.querySelectorAll(".src-device").forEach((e) => (e.hidden = t !== "device"));
  document.querySelectorAll(".src-file").forEach((e) => (e.hidden = t !== "file"));
  document.querySelectorAll(".src-manual").forEach((e) => (e.hidden = !manual || t === "demo" || t === "file"));
}

async function applySource(body, msg) {
  try {
    await api("/api/source", { json: body });
    if (!cur().frameId) live.src = "/api/stream?" + Date.now();
    toast(msg);
    setTimeout(async () => { await refreshStatus(); renderSettings(); loadDevices(); }, 1500);
  } catch (e) { toast(e.message, true); }
}

async function loadDevices() {
  const box = $("devList");
  box.innerHTML = '<span class="muted">Taranıyor…</span>';
  let r;
  try { r = await api("/api/devices"); } catch (e) { box.textContent = e.message; return; }
  box.innerHTML = "";
  if (!r.devices.length) {
    box.innerHTML = '<span class="muted">Video cihazı bulunamadı. Capture card\'ı takın.</span>';
    return;
  }
  const src = S.status.capture.source;
  for (const d of r.devices) {
    const row = document.createElement("div");
    row.className = "dev" + (d.active ? " active" : "");
    const tags = [];
    if (d.active) tags.push('<span class="tag ok">kullanılıyor</span>');
    if (d.preferred) tags.push('<span class="tag ok">tercih edilen</span>');
    else if (d.auto_choice && !d.active) tags.push('<span class="tag">otomatik seçim</span>');
    if (d.ignored) tags.push('<span class="tag warn">yok sayılıyor</span>');
    else if (d.looks_like_webcam) tags.push('<span class="tag warn">kamera</span>');
    else if (d.looks_like_capture) tags.push('<span class="tag">capture card</span>');
    const nm = document.createElement("span"); nm.className = "name";
    nm.textContent = `#${d.index} ${d.name}`;
    row.appendChild(nm);
    row.insertAdjacentHTML("beforeend", tags.join(""));
    const use = document.createElement("button"); use.type = "button"; use.className = "ghost";
    use.textContent = d.preferred ? "Tercihi kaldır" : "Bunu kullan";
    use.onclick = () => applySource(d.preferred ? { type: "auto", device_name: "" } :
      { type: "auto", device_name: d.name, ignore: (src.ignore || []).filter((x) => x.toLowerCase() !== d.name.toLowerCase()) },
      d.preferred ? "Otomatik seçime dönüldü." : `Tercih: ${d.name}`);
    const ign = document.createElement("button"); ign.type = "button"; ign.className = "ghost";
    ign.textContent = d.ignored ? "Yok saymayı kaldır" : "Yok say";
    ign.onclick = () => {
      const list = (src.ignore || []).filter((x) => x.toLowerCase() !== d.name.toLowerCase());
      if (!d.ignored) list.push(d.name);
      applySource({ type: "auto", ignore: list, ...(d.preferred && !d.ignored ? { device_name: "" } : {}) },
        d.ignored ? `${d.name} tekrar kullanılabilir.` : `${d.name} yok sayılacak.`);
    };
    row.append(use, ign);
    box.appendChild(row);
  }
}

function bindSettings() {
  $("btnSettings").onclick = async () => { await refreshStatus(); renderSettings(); await loadModelForm(); $("dlgSettings").showModal(); loadDevices(); };
  $("btnProbe").onclick = loadDevices;
  $("srcType").onchange = syncSourceFields;
  $("srcFourcc").onchange = syncSourceFields;
  $("btnSrcApply").onclick = () => applySource({
    type: $("srcType").value, index: +$("srcIndex").value, path: $("srcPath").value,
    width: +$("srcW").value, height: +$("srcH").value, fourcc: $("srcFourcc").value,
  }, "Kaynak değiştirildi.");
  document.querySelectorAll("[data-area]").forEach((b) => (b.onclick = () => {
    if (!cur().frameId) return toast("Önce görüntüyü dondurun.");
    $("dlgSettings").close(); setTool(b.dataset.area);
  }));
  $("btnClearAreas").onclick = async () => {
    if (!confirm("ROI ve tüm gizlilik maskeleri silinsin mi?")) return;
    await api("/api/config", { json: { roi: null, privacy_masks: [] } });
    await refreshStatus(); renderSettings(); draw();
  };
  $("btnCalib").onclick = () => {
    if (!cur().frameId) return toast("Önce görüntüyü dondurun.");
    $("dlgSettings").close(); setTool("calib");
  };
  $("btnCalManual").onclick = () => { const v = parseFloat($("calManual").value); calCall({ mm_per_px: v > 0 ? v : null }); };
  $("btnCalSave").onclick = () => { const n = $("calName").value.trim(); if (!n) return toast("Ön ayar adı girin."); calCall({ save_as: n }); $("calName").value = ""; };
  $("btnMdlApply").onclick = async () => {
    const size = $("mdlSize").value.toLowerCase().split(/[x×*, ]+/).map(Number);
    $("mdlStatus").textContent = "Yükleniyor…";
    try {
      await api("/api/model", { json: {
        path: $("mdlPath").value, backend: $("mdlBackend").value, encoder: $("mdlEncoder").value,
        in_channels: +$("mdlCin").value, classes: +$("mdlClasses").value,
        input_size: size.length === 2 && size.every((n) => n > 0) ? size : [512, 512],
        normalize: $("mdlNorm").value, threshold: +$("mdlThr").value,
      } });
      await refreshStatus(); renderSettings(); setLiveMode(!cur().frameId);
    } catch (e) { $("mdlStatus").textContent = e.message; }
  };
  $("btnCases").onclick = async () => {
    const list = $("caseList"); list.innerHTML = "Yükleniyor…"; $("caseReport").hidden = true;
    $("dlgCases").showModal();
    try {
      const r = await api("/api/cases");
      list.innerHTML = r.cases.length ? "" : '<span class="muted">Kayıt yok.</span>';
      for (const c of r.cases) {
        const b = document.createElement("button");
        b.innerHTML = `<span>${c.created_at} · ${c.study_no || "–"}</span><span>${c.level_text || ""} · ${c.max_mm ? c.max_mm.toFixed(1) + " mm" : "–"}</span>`;
        b.onclick = async () => { $("caseReport").textContent = await api(`/api/cases/${c.case_id}/report`); $("caseReport").hidden = false; };
        list.appendChild(b);
      }
    } catch (e) { list.textContent = e.message; }
  };
}

// --------------------------------------------------------------- başlat
async function init() {
  await refreshStatus();
  if (!S.status) { setTimeout(init, 2000); return; }
  buildForm(S.status.lexicon);
  bindSettings();
  $("modeClinical").onclick = () => setMode("clinical");
  $("modeAnnot").onclick = () => setMode("annot");
  $("modeBiopsy").onclick = () => setMode("biopsy");
  $("bxPickTarget").onclick = () => setPick(B.pick === "target" ? null : "target");
  $("bxPickEntry").onclick = () => setPick(B.pick === "entry" ? null : "entry");
  $("bxClear").onclick = async () => { await api("/api/biopsy/plan", { json: { clear: true } }).catch((e) => toast(e.message, true)); setPick("target"); };
  $("bxSnap").onclick = biopsySnapshot;
  $("bxRadius").addEventListener("change", () => api("/api/biopsy/plan", { json: { radius_mm: +$("bxRadius").value || 5 } }).catch(() => {}));
  $("bxTrack").addEventListener("change", () => api("/api/biopsy/plan", { json: { track_target: $("bxTrack").checked } }).catch(() => {}));
  $("btnAnnSave").onclick = saveAnnotation;
  $("btnNextNodule").onclick = nextNodule;
  $("btnExport").onclick = exportDataset;
  $("annotator").value = load("annotator") || "";
  $("annPatient").addEventListener("change", () => { $("annNodule").value = "N1"; });
  // Etiket alanlarında Enter: odağı bırak, kısayollar (Space/S) hemen çalışsın
  ["annPatient", "annNodule", "annotator", "studyNo", "radiologist"].forEach((id) =>
    $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } }));
  $("viewer").addEventListener("mousedown", () => document.activeElement?.blur?.());
  setMode(load("mode") === "annot" ? "annot" : "clinical");
  $("radiologist").value = load("radiologist") || "";

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchView(t.dataset.view)));
  document.querySelectorAll("#tools button[data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  $("btnFreeze").onclick = freezeToggle;
  $("btnClearView").onclick = () => clearViewData(S.view, false);
  $("fileInput").onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; };
  for (const k of ["width", "ap", "length"]) $(LINE_INPUT[k]).addEventListener("input", () => onDimInput(k));
  ["approved", "radiologist"].forEach((id) => $(id).addEventListener("input", updateSaveEnabled));
  $("btnSave").onclick = save;
  $("cbConfirm").onclick = confirmContour;
  $("cbUndo").onclick = undo;
  $("cbRedo").onclick = redo;
  $("cbDelete").onclick = deleteContour;
  $("cbAi").onclick = () => setTool("ai");
  $("contourBar").querySelectorAll("[data-cb]").forEach((b) => (b.onclick = () => setTool(b.dataset.cb)));
  $("btnNew").onclick = () => confirm("Mevcut değerlendirme temizlensin mi?") && resetAll();
  live.onerror = () => setTimeout(() => { if (!cur().frameId) live.src = "/api/stream?" + Date.now(); }, 2000);

  window.addEventListener("resize", fit);
  window.addEventListener("keydown", (e) => {
    // Metin alanlarında kısayol yok; radyo/onay kutularında kısayollar çalışır
    if (e.target.matches('input:not([type="radio"]):not([type="checkbox"]), textarea, select') ||
        document.querySelector("dialog[open]")) return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && !e.altKey) {
      if (k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      else if (k === "y" || (k === "z" && e.shiftKey)) { e.preventDefault(); redo(); }
      return;
    }
    if (e.altKey) return;
    if (S.mode === "biopsy") {
      if (k === "t") setPick("target");
      else if (k === "g") setPick("entry");
      else if (k === "f") biopsySnapshot();
      else if (k === "escape") setPick(null);
      else if (e.code === "Space") { e.preventDefault(); toast("Biyopsi modunda dondurma kapalı; iğne konumunu F ile kaydedin."); }
      return;
    }
    const map = { a: "ai", w: "width", h: "ap", l: "length", d: "draw", e: "edit", k: "splice" };
    if (k === "enter" && S.tool !== "draw" && isPending(cur())) { e.preventDefault(); return confirmContour(); }
    if ((k === "delete" || k === "backspace") && cur().contour) { e.preventDefault(); return deleteContour(); }
    if ((k === "[" || k === "]") && S.tool === "edit") return setBrush(S.brush * (k === "]" ? 1.15 : 1 / 1.15));
    if (k === "s" && S.mode === "annot") { e.preventDefault(); return saveAnnotation(); }
    if (e.code === "Space") { e.preventDefault(); e.target.blur?.(); freezeToggle(); }
    else if (k === "1") switchView("transverse");
    else if (k === "2") switchView("longitudinal");
    else if (map[k]) setTool(map[k]);
    else if (k === "escape") setTool(null);
    else if (k === "enter" && S.tool === "draw") finishPoly();
  });

  setLiveMode(true); fit(); renderDims();
  setInterval(refreshStatus, 1500);
}
init();
