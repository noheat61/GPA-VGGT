// Side-by-side point-cloud viewer: GPA-VGGT on the left, a selectable baseline on the
// right, one shared camera so both views always show the same viewpoint.
// Data comes from docs/analysis/export_page_clouds.py (static/clouds/).
//
// Loading order matters more than decode speed (one cloud decodes in ~3 ms): three.js is
// imported dynamically so the first scene's clouds download in parallel with it, and once
// that scene is on screen every other cloud is prefetched in the background.
const threeReady = Promise.all([
  import("three"),
  import("three/addons/controls/OrbitControls.js"),
]);
let THREE, OrbitControls;

const BASE = "static/clouds/";
const POINT_SCALE = 0.0045;   // point size as a fraction of the scene span

// preview.html (bundle.py) inlines the cloud files as base64; the deployed page fetches them.
// A data: URL lets the browser's native base64 decoder do the work.
// Download progress per path, {loaded, total} in bytes; read by the loading bars. A
// prefetch that is already in flight reports here too, so its bar picks up mid-way.
const progress = new Map();

async function getBuffer(path) {
  const inline = window.__CLOUDS__ && window.__CLOUDS__[path];
  if (inline) return (await fetch(`data:application/octet-stream;base64,${inline}`)).arrayBuffer();
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return res.arrayBuffer();
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let loaded = 0;
  progress.set(path, { loaded, total });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, loaded);
    loaded += value.length;
    progress.set(path, { loaded, total });
  }
  return out.buffer;
}

async function getMeta() {
  if (window.__CLOUDS__ && window.__CLOUDS__["scenes.json"]) return JSON.parse(atob(window.__CLOUDS__["scenes.json"]));
  const res = await fetch(BASE + "scenes.json");
  if (!res.ok) throw new Error(`scenes.json: HTTP ${res.status}`);
  return res.json();
}

function thumbURL(id) {
  const key = `${id}/thumb.jpg`;
  return window.__CLOUDS__ && window.__CLOUDS__[key] ? `data:image/jpeg;base64,${window.__CLOUDS__[key]}` : BASE + key;
}

// Layout: n uint16 xyz triples quantized to the scene bbox, then n uint8 rgb triples.
function decode(buf, scene) {
  const n = buf.byteLength / 9;
  const q = new Uint16Array(buf, 0, n * 3);
  const col = new Uint8Array(buf, n * 6, n * 3);
  const [lo, hi] = scene.bbox;
  const sx = (hi[0] - lo[0]) / 65535, sy = (hi[1] - lo[1]) / 65535, sz = (hi[2] - lo[2]) / 65535;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i += 3) {
    pos[i] = lo[0] + q[i] * sx;
    pos[i + 1] = lo[1] + q[i + 1] * sy;
    pos[i + 2] = lo[2] + q[i + 2] * sz;
  }
  return { pos, col };
}

// Raw decoded arrays, keyed "scene/method". Needs no three.js, so it can start at once.
const cache = new Map();
function loadCloud(scene, method) {
  const key = `${scene.id}/${method}`;
  if (!cache.has(key)) {
    cache.set(key, getBuffer(`${key}.bin`).then(buf => decode(buf, scene)).catch(err => {
      cache.delete(key);
      throw err;
    }));
  }
  return cache.get(key);
}

const geoms = new Map();
function toGeometry(key, data) {
  if (!geoms.has(key)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(data.col, 3, true));
    geoms.set(key, g);
  }
  return geoms.get(key);
}

// Fetch one at a time so background prefetch never competes with what is on screen.
async function prefetch(meta, first) {
  const order = [first, ...meta.scenes.filter(s => s !== first)];
  for (const s of order) {
    for (const m of ["ours", "fastvggt", "vanilla", "avggt_s2"].filter(m => m in s.methods)) {
      try { await loadCloud(s, m); } catch { /* retried on demand */ }
    }
  }
}

function makePane(el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  // Colours are stored as display sRGB; skip three's linear->sRGB conversion.
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(0xffffff, 1);
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const material = new THREE.PointsMaterial({ vertexColors: true, sizeAttenuation: true, size: 0.01 });
  const points = new THREE.Points(new THREE.BufferGeometry(), material);
  scene.add(points);
  const loading = el.querySelector(".viewer-loading");
  loading.innerHTML = `
    <div class="load-box">
      <div class="load-text">Loading point cloud</div>
      <div class="load-track"><div class="load-fill"></div></div>
      <div class="load-num"></div>
    </div>`;
  return { el, renderer, scene, points, material, loading,
           fill: loading.querySelector(".load-fill"), num: loading.querySelector(".load-num") };
}

async function init() {
  const left = document.getElementById("viewer-left");
  const right = document.getElementById("viewer-right");
  if (!left || !right) return;

  let meta;
  try {
    meta = await getMeta();
    // Start the first scene's two clouds now, while three.js is still downloading.
    loadCloud(meta.scenes[0], "ours").catch(() => {});
    loadCloud(meta.scenes[0], document.getElementById("baseline").value).catch(() => {});
    const [three, controlsModule] = await threeReady;
    THREE = three;
    OrbitControls = controlsModule.OrbitControls;
  } catch (err) {
    for (const el of [left, right]) {
      const msg = document.createElement("div");
      msg.className = "viewer-error";
      msg.textContent = location.protocol === "file:"
        ? "The 3D viewer needs the page to be served over HTTP. Run `python -m http.server` in project_page/ or open preview.html."
        : "The 3D viewer could not load. Check your connection and reload the page.";
      el.append(msg);
    }
    console.error(err);
    return;
  }

  const panes = [makePane(left), makePane(right)];
  const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.001, 1000);
  // One OrbitControls per canvas, both driving the shared camera; keep their targets in step.
  const controls = panes.map(p => {
    const c = new OrbitControls(camera, p.renderer.domElement);
    c.enableDamping = false;
    c.screenSpacePanning = true;
    c.zoomSpeed = 0.8;
    return c;
  });
  const render = () => panes.forEach(p => p.renderer.render(p.scene, camera));
  controls.forEach((c, i) => c.addEventListener("change", () => {
    controls[1 - i].target.copy(c.target);
    render();
  }));

  function resize() {
    const w = left.clientWidth, h = left.clientHeight;
    panes.forEach(p => p.renderer.setSize(p.el.clientWidth, p.el.clientHeight, false));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }
  new ResizeObserver(resize).observe(left);

  const select = document.getElementById("baseline");
  const sizeInput = document.getElementById("pt-size");
  const statL = document.getElementById("stat-left");
  const statR = document.getElementById("stat-right");
  const strip = document.getElementById("scene-strip");
  let current = meta.scenes[0];

  function setSize() {
    const s = current.span * POINT_SCALE * parseFloat(sizeInput.value);
    panes.forEach(p => (p.material.size = s));
    render();
  }

  function resetView() {
    const c = current.camera;
    camera.fov = c.fov;
    camera.near = current.span * 0.002;
    camera.far = current.span * 200;
    camera.up.set(...c.up);
    camera.position.set(...c.eye);
    camera.lookAt(...c.target);
    camera.updateProjectionMatrix();
    controls.forEach(ctl => { ctl.target.set(...c.target); ctl.update(); });
    render();
  }

  function stat(el, m) {
    el.innerHTML = `AUC@3 <b>${m.auc3.toFixed(1)}</b>`;
  }

  let token = 0;
  async function show(pane, method, t) {
    const path = `${current.id}/${method}.bin`;
    pane.loading.hidden = false;
    pane.fill.style.width = "0%";
    pane.num.textContent = "";
    let live = true;
    (function tick() {
      if (!live) return;
      const p = progress.get(path);
      if (p) {
        pane.fill.style.width = `${(100 * p.loaded / p.total).toFixed(1)}%`;
        pane.num.textContent = `${Math.round(100 * p.loaded / p.total)}% · ` +
          `${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB`;
      }
      requestAnimationFrame(tick);
    })();
    try {
      const data = await loadCloud(current, method);
      if (t !== token) return;
      pane.points.geometry = toGeometry(`${current.id}/${method}`, data);
    } catch (err) {
      console.error(err);
    } finally {
      live = false;
      if (t === token) pane.loading.hidden = true;
    }
    render();
  }

  // A scene may lack a baseline (ScanNet has no AVGGT run): disable it, fall back to FastVGGT.
  function syncOptions() {
    for (const opt of select.options) {
      opt.disabled = !(opt.value in current.methods);
      opt.textContent = opt.disabled ? `${opt.dataset.name} (not run)` : opt.dataset.name;
    }
    if (select.selectedOptions[0].disabled) select.value = "fastvggt";
  }
  for (const opt of select.options) opt.dataset.name = opt.textContent;

  function update(resetCam) {
    syncOptions();
    const t = ++token;
    stat(statL, current.methods.ours);
    stat(statR, current.methods[select.value]);
    if (resetCam) resetView();
    setSize();
    return Promise.all([show(panes[0], "ours", t), show(panes[1], select.value, t)]);
  }

  strip.innerHTML = meta.scenes.map((s, i) => `
    <button class="scene" role="tab" aria-selected="${i === 0}" data-i="${i}">
      <img src="${thumbURL(s.id)}" alt="" loading="lazy">
      <span>${s.dataset} · ${s.label}<small>${s.seq}</small></span>
    </button>`).join("");
  strip.querySelectorAll(".scene").forEach(b => b.addEventListener("click", () => {
    current = meta.scenes[+b.dataset.i];
    strip.querySelectorAll(".scene").forEach(x => x.setAttribute("aria-selected", x === b));
    update(true);
  }));
  select.addEventListener("change", () => update(false));
  sizeInput.addEventListener("input", setSize);
  document.getElementById("reset-view").addEventListener("click", resetView);

  resize();
  await update(true);
  prefetch(meta, current);
}

init();
