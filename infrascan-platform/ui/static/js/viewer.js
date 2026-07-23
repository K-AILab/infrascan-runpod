/* viewer.js — three.js point-cloud viewer with verbose console logging
 * so you can tell what's happening at every step.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader }     from "three/addons/loaders/PLYLoader.js";

const SP = window.INFRASCAN_SPACE;
if (!SP) { throw new Error("INFRASCAN_SPACE missing"); }

const TAG = "[viewer]";
const ASSET = (path) => `/spaces/${encodeURIComponent(SP.slug)}/asset/${path}`;
console.log(TAG, "space =", SP);
console.log(TAG, "PLY url =", ASSET("downsampled_web.ply"));
console.log(TAG, "cameras url =", ASSET("cameras.json"));

const canvas        = document.getElementById("threeCanvas");
const placeholder   = document.getElementById("canvasPlaceholder");
const scanpointName = document.getElementById("scanpointName");
const scanpointSub  = document.getElementById("scanpointSub");

if (!canvas) { console.error(TAG, "no #threeCanvas element"); }

// ── scene + camera + renderer ────────────────────────────────────────────
const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x14110d);

const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 5000);
camera.position.set(4, 4, 4);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

scene.add(new THREE.AmbientLight(0xffffff, 0.8));

// Axes helper (50 m) — kill once everyone's used to the orientation.
const axes = new THREE.AxesHelper(5);
scene.add(axes);

function fitToContainer() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) { return; }
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", fitToContainer);
fitToContainer();

const cloudGroup  = new THREE.Group();
const pointsGroup = new THREE.Group();
scene.add(cloudGroup);
scene.add(pointsGroup);

// Per-axis sign vector (1 or -1). Pickable from the HUD; persisted per space.
// Key is versioned so default changes take effect on existing browsers.
const FLIP_KEY = `infrascan_flip_v2_${SP.slug}`;
const DEFAULT_FLIP = { x: 1, y: 1, z: -1 };  // flip Z for video-input
const flip = (() => {
  try {
    const cached = JSON.parse(localStorage.getItem(FLIP_KEY) || "null");
    if (cached && typeof cached.x === "number") return cached;
  } catch (_e) {}
  // Also nuke any older-version keys so they don't shadow this default.
  Object.keys(localStorage).filter(k => k.startsWith(`infrascan_flip_${SP.slug}`))
    .forEach(k => { if (k !== FLIP_KEY) localStorage.removeItem(k); });
  return { ...DEFAULT_FLIP };
})();
console.log(TAG, "flip =", flip);

function applyFlip() {
  cloudGroup.scale.set(flip.x, flip.y, flip.z);
  pointsGroup.scale.set(flip.x, flip.y, flip.z);
  localStorage.setItem(FLIP_KEY, JSON.stringify(flip));
}
applyFlip();

["flipX","flipY","flipZ"].forEach((id) => {
  const axis = id.slice(-1).toLowerCase();
  const btn = document.getElementById(id);
  if (!btn) return;
  const refresh = () => btn.classList.toggle("on", flip[axis] === -1);
  refresh();
  btn.addEventListener("click", () => {
    flip[axis] = -flip[axis];
    applyFlip();
    refresh();
    refit();
  });
});

document.getElementById("recenter")?.addEventListener("click", refit);

document.getElementById("resetFlip")?.addEventListener("click", () => {
  flip.x = DEFAULT_FLIP.x; flip.y = DEFAULT_FLIP.y; flip.z = DEFAULT_FLIP.z;
  applyFlip();
  ["flipX","flipY","flipZ"].forEach(id => {
    const axis = id.slice(-1).toLowerCase();
    document.getElementById(id)?.classList.toggle("on", flip[axis] === -1);
  });
  refit();
});

let __bs = null;
function refit() {
  if (!__bs) return;
  const c = __bs.center.clone();
  c.x *= flip.x; c.y *= flip.y; c.z *= flip.z;
  controls.target.copy(c);
  const dist = Math.max(__bs.radius * 1.8, 3);
  camera.position.set(c.x + dist, c.y + dist * 0.6, c.z + dist);
  camera.near = Math.max(__bs.radius / 1000, 0.05);
  camera.far  = __bs.radius * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

// ── Fetch asset sizes for real % progress ────────────────────────────────
const sizesPromise = fetch(`/api/spaces/${encodeURIComponent(SP.slug)}/assets`)
  .then(r => r.ok ? r.json() : {})
  .then(sizes => { console.log(TAG, "asset sizes (bytes):", sizes); return sizes; })
  .catch(() => ({}));

// ── PLY load — stream + parse so we can show a real % ───────────────────
const t0 = performance.now();
sizesPromise.then(async (sizes) => {
  const expectedBytes = sizes["downsampled_web.ply"] || 0;
  if (expectedBytes) {
    scanpointSub.textContent = `point cloud · 0% (${(expectedBytes/1024/1024).toFixed(1)} MB)`;
  } else {
    scanpointSub.textContent = "point cloud · loading…";
  }
  try {
    const resp = await fetch(ASSET("downsampled_web.ply"));
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const total = expectedBytes || parseInt(resp.headers.get("content-length") || "0", 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      if (total) {
        const pct = (received / total) * 100;
        scanpointSub.textContent = `point cloud · ${pct.toFixed(0)}% · ${(received/1024/1024).toFixed(1)} / ${(total/1024/1024).toFixed(1)} MB`;
      } else {
        scanpointSub.textContent = `point cloud · ${(received/1024/1024).toFixed(1)} MB received`;
      }
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
    onPLYLoaded(new PLYLoader().parse(buf.buffer));
  } catch (err) {
    console.error(TAG, "PLY load failed:", err);
    scanpointName.textContent = "Point cloud failed to load";
    scanpointSub.textContent  = "Open the console to see why.";
  }
});

function onPLYLoaded(geometry) {
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  geometry.computeBoundingSphere();
  __bs = geometry.boundingSphere;
  const nPts = geometry.attributes.position.count;
  const hasColor = !!geometry.getAttribute("color");
  console.log(TAG, `PLY parsed in ${dt}s · ${nPts.toLocaleString()} points · vertexColors=${hasColor}`);
  console.log(TAG, `bounding sphere center=${__bs.center.toArray().map(v=>v.toFixed(2))} radius=${__bs.radius.toFixed(2)}`);

  const ptSize = Math.max(2.0, Math.min(4.0, __bs.radius * 0.002 + 2));
  const points = new THREE.Points(geometry, new THREE.PointsMaterial({
    size: ptSize,
    vertexColors: hasColor,
    color: hasColor ? 0xffffff : 0xcc785c,
    sizeAttenuation: false,
  }));
  cloudGroup.add(points);
  refit();

  placeholder.style.display = "none";
  scanpointName.textContent = SP.title;
  scanpointSub.textContent  = `${nPts.toLocaleString()} points`;
}

// ── cameras.json → scanpoint markers + walk.scanpoints (single source) ───
fetch(ASSET("cameras.json")).then(r => {
  console.log(TAG, "cameras.json HTTP", r.status);
  return r.ok ? r.json() : null;
}).then((cams) => {
  if (!Array.isArray(cams)) {
    console.warn(TAG, "cameras.json is not an array; got", cams);
    return;
  }
  console.log(TAG, `cameras.json has ${cams.length} entries; sample:`, cams[0]);

  const byFrame = new Map();
  for (const c of cams) {
    const f = (typeof c.frame === "number") ? c.frame :
              (typeof c.id === "number") ? c.id : null;
    if (f !== null && !byFrame.has(f)) byFrame.set(f, c);
  }
  console.log(TAG, `→ ${byFrame.size} unique scanpoints`);

  // ① Populate the Walk-mode index immediately (before any three.js work).
  walk.scanpoints = [...byFrame.keys()].sort((a, b) => a - b);
  console.log(TAG, "walk.scanpoints =", walk.scanpoints.length,
              `(first=${walk.scanpoints[0]}, last=${walk.scanpoints.at(-1)})`);
  // If Walk mode happens to be visible now, refresh it.
  if (panes.walk && panes.walk.style.display !== "none") refreshWalk();
  if (panes.map  && panes.map.style.display  !== "none") refreshMap();

  // ② Build 3D markers (cheaper than waiting on them to populate Walk).
  const markerGeo = new THREE.SphereGeometry(0.15, 16, 12);
  for (const [frame, c] of byFrame.entries()) {
    const [x, y, z] = c.pos || [0, 0, 0];
    const m = new THREE.Mesh(
      markerGeo,
      new THREE.MeshBasicMaterial({ color: 0xcc785c, transparent: true, opacity: 0.9 }),
    );
    m.position.set(x, y, z);
    m.userData = { frame, name: `Scanpoint ${String(frame).padStart(3, "0")}` };
    pointsGroup.add(m);
  }
}).catch((e) => console.error(TAG, "cameras.json failed:", e));

// ── Click → teleport ──────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
canvas.addEventListener("pointerdown", (ev) => {
  const r = canvas.getBoundingClientRect();
  mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  mouse.y = -((ev.clientY - r.top)  / r.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(pointsGroup.children, false);
  if (hits.length) {
    const m = hits[0].object;
    controls.target.copy(m.position);
    scanpointName.textContent = m.userData.name;
  }
});

// ── render loop ──────────────────────────────────────────────────────────
function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}
loop();
console.log(TAG, "render loop started");


// ─── Mode switching: 3D ↔ Walk ↔ Map ────────────────────────────────────
const panes = {
  "3d":   document.querySelector('.mode-pane[data-mode="3d"]'),
  "walk": document.querySelector('.mode-pane[data-mode="walk"]'),
  "map":  document.querySelector('.mode-pane[data-mode="map"]'),
};
const flipBar = document.getElementById("flipBar");

function setMode(mode) {
  Object.entries(panes).forEach(([k, el]) => { if (el) el.style.display = (k === mode) ? "block" : "none"; });
  document.querySelectorAll('.seg button[data-mode]').forEach(b => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
  flipBar.style.display = (mode === "3d") ? "" : "none";
  if (mode === "walk") refreshWalk();
  if (mode === "map")  refreshMap();
  console.log(TAG, "mode →", mode);
}
document.querySelectorAll('.seg button[data-mode]').forEach(b => {
  b.addEventListener("click", () => setMode(b.dataset.mode));
});


// ─── Walk mode state ────────────────────────────────────────────────────
const walk = {
  scanpoints: [],      // sorted frame ids that exist on disk
  spIdx: 0,            // current index into scanpoints
  yawIdx: 0,           // 0..11 — the 12 standard yaws
  YAWS: [0,30,60,90,120,150,180,210,240,270,300,330],
};

const walkImg = document.getElementById("walkImg");
const walkSp  = document.getElementById("walkSp");
const walkYaw = document.getElementById("walkYaw");

function _viewName(frame, yaw) {
  const f = String(frame).padStart(6, "0");
  const y = String(yaw).padStart(3, "0");
  return `${f}_pz000_y${y}_normal.jpg`;
}

function refreshWalk() {
  if (!walk.scanpoints.length) {
    walkSp.textContent  = "no scanpoints loaded";
    walkYaw.textContent = "—";
    return;
  }
  const frame = walk.scanpoints[walk.spIdx];
  const yaw   = walk.YAWS[walk.yawIdx];
  const name  = _viewName(frame, yaw);
  walkImg.src = ASSET("views/" + name);
  walkImg.onerror = () => { console.warn(TAG, "missing view", name); };
  walkSp.textContent  = `scanpoint ${walk.spIdx + 1} of ${walk.scanpoints.length} · frame ${frame}`;
  walkYaw.textContent = String(yaw).padStart(3, "0") + "°";
  scanpointName.textContent = `Walk · ${walk.spIdx + 1}/${walk.scanpoints.length}`;
}

function _step(dir) { walk.spIdx = Math.max(0, Math.min(walk.scanpoints.length - 1, walk.spIdx + dir)); refreshWalk(); }
function _yaw(dir)  { walk.yawIdx  = (walk.yawIdx  + dir + walk.YAWS.length) % walk.YAWS.length; refreshWalk(); }

document.getElementById("walkPrev")?.addEventListener("click", () => _step(-1));
document.getElementById("walkNext")?.addEventListener("click", () => _step(+1));
document.getElementById("walkYawL")?.addEventListener("click", () => _yaw(-1));
document.getElementById("walkYawR")?.addEventListener("click", () => _yaw(+1));
document.addEventListener("keydown", (e) => {
  // Only when Walk is the visible mode
  if (panes.walk.style.display === "none") return;
  if (e.target.matches("input, textarea")) return;
  if (e.key === "ArrowUp")    { _step(-1); e.preventDefault(); }
  if (e.key === "ArrowDown")  { _step(+1); e.preventDefault(); }
  if (e.key === "ArrowLeft")  { _yaw(-1);  e.preventDefault(); }
  if (e.key === "ArrowRight") { _yaw(+1);  e.preventDefault(); }
});

// Click on the perspective image — stub for now. Real wiring needs a
// server endpoint that takes (slug, view_filename, click_xy_norm) and
// returns the nearest object's embedding + top-K matches.
walkImg.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  const r = walkImg.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = (e.clientY - r.top)  / r.height;
  console.log(TAG, "click @", { view: walkImg.src.split("/").pop(), x: x.toFixed(3), y: y.toFixed(3) });
  scanpointSub.textContent = `clicked at (${(x*100).toFixed(0)}%, ${(y*100).toFixed(0)}%) — FAISS wiring pending`;
});
// Stop browser drag-image behaviour that some users perceive as a phantom click.
walkImg.setAttribute("draggable", "false");


// ─── Map mode state ─────────────────────────────────────────────────────
const mapImg  = document.getElementById("mapImg");
const mapDots = document.getElementById("mapDots");
let mapBounds = null;

function refreshMap() {
  mapImg.src = ASSET("topdown.png");
  fetch(ASSET("bounds.json")).then(r => r.ok ? r.json() : null).then((b) => {
    mapBounds = b;
    drawMapDots();
  }).catch(() => {});
}

function drawMapDots() {
  if (!mapBounds || !walk.scanpoints.length) return;
  // bounds.json schema: {axis_u, axis_v, u_min, u_max, v_min, v_max, width, height, v_flipped}
  // axis_u / axis_v are 0/1/2 = x/y/z. v_flipped means the image's Y goes the other way.
  const axNames = ["x", "y", "z"];
  const uAxis = axNames[mapBounds.axis_u ?? 2];
  const vAxis = axNames[mapBounds.axis_v ?? 0];
  const { u_min, u_max, v_min, v_max, v_flipped } = mapBounds;

  mapDots.innerHTML = "";
  const w = mapImg.clientWidth  || mapDots.clientWidth;
  const h = mapImg.clientHeight || mapDots.clientHeight;
  pointsGroup.children.forEach((m) => {
    const u = m.position[uAxis];
    const v = m.position[vAxis];
    let nx = (u - u_min) / Math.max(u_max - u_min, 1e-6);
    let ny = (v - v_min) / Math.max(v_max - v_min, 1e-6);
    if (v_flipped) ny = 1 - ny;
    const x = nx * w, y = ny * h;
    mapDots.insertAdjacentHTML("beforeend",
      `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="#cc785c" stroke="#fff" stroke-width="1" />`
    );
  });
}


// walk.scanpoints is now populated inside the cameras.json .then() above —
// no polling needed.
