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
  return { frameId: null, w: 0, h: 0, img: null, contour: null, ai: null, lines: {} };
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
  const srcName = { device: `Capture #${c.source.index}`, file: "Dosya", demo: "Demo" }[c.source.type] || "–";
  const chipS = $("chipSource");
  chipS.textContent = `${srcName}: ${c.status}` + (c.resolution ? ` · ${c.resolution[0]}×${c.resolution[1]} · ${c.fps} fps` : "");
  chipS.className = "chip " + (c.status === "açık" ? "ok" : c.status === "hata" ? "bad" : "warn");
  chipS.title = c.error || "";
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
  card.className = "card result" + (s.level ? " lvl" + s.level : "");
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
    ctx.save();
    ctx.beginPath();
    v.contour.forEach((p, i) => { const q = toCv(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    ctx.closePath();
    ctx.fillStyle = "rgba(255,200,61,.14)"; ctx.fill();
    ctx.strokeStyle = "#ffc83d"; ctx.lineWidth = 2 * dpr; ctx.stroke();
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
  if (S.calibPts) {
    ctx.fillStyle = "#36b3e0";
    S.calibPts.forEach((p) => { const q = toCv(p); ctx.beginPath(); ctx.arc(q[0], q[1], 5 * dpr, 0, 7); ctx.fill(); });
  }
  ctx.restore();
}

function setTool(t) {
  if (t && !cur().frameId) { toast("Önce görüntüyü dondurun."); return; }
  S.tool = S.tool === t ? null : t;
  S.poly = null; S.drag = null;
  if (S.tool !== "calib") S.calibPts = null;
  document.querySelectorAll("#tools button[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === S.tool));
  const hints = {
    ai: "Nodülün içine tıklayın; model o noktadaki nodülü segmente eder.",
    width: "Transvers çapı çizin: tıklayıp sürükleyin.",
    ap: "Ön-arka (AP) çapı çizin: tıklayıp sürükleyin.",
    length: "Longitudinal uzunluğu çizin: tıklayıp sürükleyin.",
    draw: "Sınır boyunca tıklayın; kapatmak için ilk noktaya tıklayın ya da Enter. Esc iptal.",
    calib: "Derinlik skalasında iki çentiğe tıklayın.",
    roi: "Model için görüntü alanını sürükleyerek seçin (menü/yazılar dışarıda kalsın).",
    privacy: "Hasta bilgisi alanını sürükleyerek seçin; kayıt ve akışta siyaha boyanır.",
  };
  $("hint").textContent = S.tool ? hints[S.tool] : (cur().frameId ? "Bir araç seçin ya da canlıya dönün." : "Nodülü ekrana getirip Dondur'a basın.");
  draw();
}

cv.addEventListener("mousedown", (ev) => {
  if (ev.button !== 0 || !S.tool) return;
  const p = toImg(ev);
  if (["width", "ap", "length"].includes(S.tool)) S.drag = { kind: "line", start: p, end: p };
  else if (["roi", "privacy"].includes(S.tool)) S.drag = { kind: "rect", start: p, end: p };
});
cv.addEventListener("mousemove", (ev) => {
  const p = toImg(ev);
  if (S.drag) { S.drag.end = p; draw(); }
  else if (S.poly) { S.poly.hover = p; draw(); }
});
window.addEventListener("mouseup", async (ev) => {
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

async function applyContour(contour, how) {
  const v = cur();
  try {
    const r = await api("/api/contour", { json: { frame_id: v.frameId, contour } });
    v.contour = contour;
    v.ai = how === "manual" ? { backend: "manuel", edited: true } : v.ai;
    autoLines(v, r.metrics_px);
    renderAiInfo(); draw(); renderDims(); markDots();
  } catch (e) { toast(e.message, true); }
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
    v.contour = r.contour;
    v.ai = { backend: r.backend, confidence: r.confidence, ms: r.ms, edited: false };
    autoLines(v, r.metrics_px);
    renderAiInfo(); draw(); renderDims(); markDots();
  } catch (e) { toast(e.message, true); }
  finally { viewerMsg(""); }
}

function renderAiInfo() {
  const parts = VIEWS.map((k) => {
    const a = S.views[k].ai;
    if (!a) return null;
    const name = k === "transverse" ? "Transvers" : "Longitudinal";
    if (a.backend === "manuel") return `${name}: elle çizildi`;
    return `${name}: ${a.backend === "demo" ? "DEMO" : "UNet++"} · güven ${(a.confidence * 100).toFixed(0)}% · ${Math.round(a.ms)} ms${a.edited ? " · düzeltildi" : ""}`;
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
  else { v.contour = null; v.lines = {}; v.ai = null; }
  if (v.frameId && !dropFrame) api("/api/contour", { json: { frame_id: v.frameId, contour: [] } }).catch(() => {});
  renderAiInfo(); renderDims(); markDots(); draw();
}

function switchView(view) {
  S.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  const frozen = !!cur().frameId;
  setLiveMode(!frozen);
  fit();
  setTool(frozen && S.tool ? S.tool : null);
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
  $("btnSave").disabled = !(anyView && s?.level && $("approved").checked && $("radiologist").value.trim());
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

// ------------------------------------------------------------ ayarlar
function renderSettings() {
  const st = S.status; if (!st) return;
  const src = st.capture.source;
  $("srcType").value = src.type; $("srcIndex").value = src.index ?? 0; $("srcPath").value = src.path || "";
  $("srcW").value = src.width || 1920; $("srcH").value = src.height || 1080; $("srcFourcc").value = src.fourcc || "";
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

function bindSettings() {
  $("btnSettings").onclick = async () => { await refreshStatus(); renderSettings(); await loadModelForm(); $("dlgSettings").showModal(); };
  $("btnProbe").onclick = async () => {
    $("probeOut").textContent = "Taranıyor…";
    try {
      const r = await api("/api/devices");
      $("probeOut").textContent = r.devices.length ? "Bulunan: " + r.devices.map((d) => `#${d.index} (${d.width}×${d.height}, ${d.backend})`).join(", ") : "Cihaz bulunamadı. Kaynak şu an açıksa önce Demo'ya alıp tekrar tarayın.";
    } catch (e) { $("probeOut").textContent = e.message; }
  };
  $("btnSrcApply").onclick = async () => {
    try {
      await api("/api/source", { json: {
        type: $("srcType").value, index: +$("srcIndex").value, path: $("srcPath").value,
        width: +$("srcW").value, height: +$("srcH").value, fourcc: $("srcFourcc").value,
      } });
      if (!cur().frameId) live.src = "/api/stream?" + Date.now();
      setTimeout(refreshStatus, 1500); toast("Kaynak değiştirildi.");
    } catch (e) { toast(e.message, true); }
  };
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
  $("radiologist").value = load("radiologist") || "";

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchView(t.dataset.view)));
  document.querySelectorAll("#tools button[data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  $("btnFreeze").onclick = freezeToggle;
  $("btnClearView").onclick = () => clearViewData(S.view, false);
  $("fileInput").onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; };
  for (const k of ["width", "ap", "length"]) $(LINE_INPUT[k]).addEventListener("input", () => onDimInput(k));
  ["approved", "radiologist"].forEach((id) => $(id).addEventListener("input", updateSaveEnabled));
  $("btnSave").onclick = save;
  $("btnNew").onclick = () => confirm("Mevcut değerlendirme temizlensin mi?") && resetAll();
  live.onerror = () => setTimeout(() => { if (!cur().frameId) live.src = "/api/stream?" + Date.now(); }, 2000);

  window.addEventListener("resize", fit);
  window.addEventListener("keydown", (e) => {
    // Metin alanlarında kısayol yok; radyo/onay kutularında kısayollar çalışır
    if (e.target.matches('input:not([type="radio"]):not([type="checkbox"]), textarea, select') ||
        document.querySelector("dialog[open]") || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const map = { a: "ai", w: "width", h: "ap", l: "length", d: "draw" };
    if (e.code === "Space") { e.preventDefault(); e.target.blur?.(); freezeToggle(); }
    else if (k === "1") switchView("transverse");
    else if (k === "2") switchView("longitudinal");
    else if (map[k]) setTool(map[k]);
    else if (k === "escape") setTool(null);
    else if (k === "enter" && S.tool === "draw") finishPoly();
  });

  setLiveMode(true); fit(); renderDims();
  setInterval(refreshStatus, 3000);
}
init();
