# Web Console 3D Arm Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a faithful, display-only 3D render of the live arm to the web console (`web/dashboard.html`), driven entirely client-side off the existing telemetry WebSocket.

**Architecture:** A one-time export script bakes each arm's URDF + visual meshes into self-contained glТF bundles under `web/models/<arm>/`. `api_server.py` serves those plus vendored three.js / urdf-loader libraries as static files. A small ES-module in the dashboard loads the selected arm's bundle with urdf-loader, renders it with view-only OrbitControls, and sets joint values from the `RobotState` already streaming over `WS /telemetry`.

**Tech Stack:** Python (trimesh, yourdfpy, robot_descriptions) for the offline export; FastAPI `StaticFiles` for serving; three.js 0.160.1 + `urdf-loader` 0.12.3 (vendored ESM, native import map) for the browser; no bundler/build step.

## Global Constraints

- Run all Python with the project venv: `./robot_control/bin/python <script>`. Tests are stdlib-`assert` scripts (no pytest), run the same way.
- The dashboard stays **vanilla HTML/JS, no build step** — ES modules wired via a native `<script type="importmap">`.
- **No CDN at runtime.** All JS libraries are vendored under `web/vendor/` (the console runs on a LAN / direct cable and must work offline).
- Viewer is **display-only**: OrbitControls move the *camera* only. No endpoint is ever called from the viewer; it never drives the robot.
- Visual meshes only, converted to glTF `.glb`, **meters scale preserved**; `<collision>` stripped from exported URDFs. For meshes with no native material (STL), bake the URDF link `<material><color>`.
- Both arms are 6-DOF. Joint order matches the controller's `q` exactly:
  - `ur15`: `["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]`
  - `gofa`: `["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]`
  - Hand-E finger joint: `"robotiq_hande_left_finger_joint"`, range `0` (closed) … `0.025` m (open). `gripper_frac` 0=open…1=closed ⇒ joint value `(1 - frac) * 0.025`.
- Every viewer hook in the classic script is guarded (`window.viewer3d && …`) so a failed/absent viewer never breaks the rest of the console.
- Git: commit at the end of each task. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  Do **not** push (the user pushes).

---

## File Structure

- **Create** `scripts/export_web_models.py` — offline asset baker (+ `--check`).
- **Create** `web/models/ur15/{ur15.urdf, hande.urdf, *.glb}`, `web/models/gofa/{gofa.urdf, *.glb}` — generated, committed.
- **Create** `web/vendor/three.module.js`, `web/vendor/jsm/**`, `web/vendor/urdf-loader/{URDFLoader.js,URDFClasses.js}` — vendored libs, committed.
- **Modify** `scripts/api_server.py` — mount `/vendor` and `/models` as static dirs.
- **Modify** `scripts/api_smoketest.py:413-454` (`test_e2e_subprocess`) — assert the static assets are served.
- **Modify** `web/dashboard.html` — import map, viewer panel + CSS, the ES-module viewer, and 4 hook calls.
- **Modify** `CLAUDE.md`, `API.md`, `README.md`, `usage.html` — docs.

---

## Task 1: Asset export script + generated model bundles

**Files:**
- Create: `scripts/export_web_models.py`
- Create (generated): `web/models/ur15/`, `web/models/gofa/`

**Interfaces:**
- Produces: committed `web/models/<arm>/<arm>.urdf` (self-contained, relative `.glb` refs, no `<collision>`); `web/models/ur15/hande.urdf`. CLI: default = generate; `--check` = verify only (exit non-zero on a missing/dangling reference).

- [ ] **Step 1: Create the export script**

Create `scripts/export_web_models.py`:

```python
#!/usr/bin/env python
"""Bake per-arm 3D model bundles for the web console.

For each arm, load its URDF, convert every VISUAL mesh to glTF (.glb) via trimesh
(meters scale preserved), strip <collision>, and write a self-contained <arm>.urdf
referencing the local .glb files into web/models/<arm>/. STL meshes (no native
material) get the URDF link <color> baked in; DAE/OBJ keep their own materials.

Run once on a box that has the source meshes (the dev machine); the output is
committed so the server serves it everywhere without the robot_descriptions cache.

  ./robot_control/bin/python scripts/export_web_models.py          # (re)generate
  ./robot_control/bin/python scripts/export_web_models.py --check  # verify only
"""
import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc  # noqa: E402

MODELS_DIR = os.path.join(_ROOT, "web", "models")


def _vis_color(vis):
    """RGBA list from a <visual>'s <material><color>, or None."""
    mat = vis.find("material")
    if mat is not None:
        col = mat.find("color")
        if col is not None and col.get("rgba"):
            return [float(x) for x in col.get("rgba").split()]
    return None


def _glb_name(src_abs, seen):
    """Stable, unique .glb basename for a source mesh path."""
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(src_abs))[0])
    name, n = base + ".glb", 1
    used = set(seen.values())
    while name in used and seen.get(src_abs) != name:
        name, n = f"{base}_{n}.glb", n + 1
    return name


def _to_glb(src_abs, out_path, color):
    import numpy as np
    import trimesh
    geom = trimesh.load(src_abs, force="scene")
    if color is not None and src_abs.lower().endswith(".stl"):
        rgba = (np.array(color) * 255).astype(np.uint8)
        for g in geom.geometry.values():
            g.visual = trimesh.visual.ColorVisuals(g, face_colors=rgba)
    geom.export(out_path)


def _convert(tree, resolver, out_dir, out_name):
    """Strip collisions, convert visual meshes to .glb, rewrite refs, write URDF."""
    os.makedirs(out_dir, exist_ok=True)
    root = tree.getroot()
    seen = {}  # src_abs -> glb basename
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
        for vis in link.findall("visual"):
            color = _vis_color(vis)
            for mesh in vis.iter("mesh"):
                fn = mesh.get("filename")
                if not fn:
                    continue
                src = resolver(fn)
                if src.startswith("file://"):
                    src = src[len("file://"):]
                src = os.path.abspath(src)
                if src not in seen:
                    glb = _glb_name(src, seen)
                    _to_glb(src, os.path.join(out_dir, glb), color)
                    seen[src] = glb
                mesh.set("filename", seen[src])
    tree.write(os.path.join(out_dir, out_name), encoding="utf-8", xml_declaration=True)
    return len(seen)


def _ur15_tree():
    """The synthesized UR15 URDF (no local file) as an ElementTree; mesh refs are file://."""
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    u = load_robot_description("ur15_description")
    fd, tmp = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    try:
        u.write_xml_file(tmp)
        return ET.parse(tmp)
    finally:
        os.remove(tmp)


def export():
    # UR15 bundle: arm + Hand-E (Hand-E meshes resolve under the project root).
    n = _convert(_ur15_tree(), lambda f: f,
                 os.path.join(MODELS_DIR, "ur15"), "ur15.urdf")
    n += _convert(ET.parse(os.path.join(_ROOT, "urdf", "hande.urdf")),
                  rc.make_mesh_resolver(rc.UR_MESH_DIR_PREFIX),
                  os.path.join(MODELS_DIR, "ur15"), "hande.urdf")
    print(f"ur15: {n} meshes -> web/models/ur15/")

    # GoFa bundle: meshes live under abb_desc/.
    n = _convert(ET.parse(os.path.join(_ROOT, "urdf", "crb15000_5_95.urdf")),
                 rc.make_mesh_resolver(os.path.join(_ROOT, "abb_desc")),
                 os.path.join(MODELS_DIR, "gofa"), "gofa.urdf")
    print(f"gofa: {n} meshes -> web/models/gofa/")


def check():
    """Every committed bundle URDF parses and every referenced .glb exists."""
    bundles = {
        "ur15": ["ur15.urdf", "hande.urdf"],
        "gofa": ["gofa.urdf"],
    }
    total = 0
    for arm, urdfs in bundles.items():
        d = os.path.join(MODELS_DIR, arm)
        for u in urdfs:
            path = os.path.join(d, u)
            assert os.path.isfile(path), f"missing {arm}/{u}"
            root = ET.parse(path).getroot()
            refs = [m.get("filename") for m in root.iter("mesh") if m.get("filename")]
            assert refs, f"{arm}/{u} references no meshes"
            for r in refs:
                assert os.path.isfile(os.path.join(d, r)), f"{arm}/{u} -> missing {r}"
                total += 1
            assert not list(root.iter("collision")), f"{arm}/{u} still has <collision>"
    print(f"OK web/models check — {total} mesh references, all present")


def main():
    ap = argparse.ArgumentParser(description="Bake web-console 3D model bundles.")
    ap.add_argument("--check", action="store_true", help="verify the committed bundles, no writes")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        export()
        check()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `--check` to confirm it fails (no bundles yet)**

Run: `./robot_control/bin/python scripts/export_web_models.py --check`
Expected: FAIL — `AssertionError: missing ur15/ur15.urdf`.

- [ ] **Step 3: Generate the bundles**

Run: `./robot_control/bin/python scripts/export_web_models.py`
Expected: prints `ur15: N meshes -> web/models/ur15/`, `gofa: M meshes -> web/models/gofa/`, then `OK web/models check — …`. Creates `web/models/ur15/{ur15.urdf,hande.urdf,*.glb}` and `web/models/gofa/{gofa.urdf,*.glb}`.

- [ ] **Step 4: Eyeball the output**

Run: `ls -1 web/models/ur15 web/models/gofa && grep -c "filename" web/models/gofa/gofa.urdf && grep -c "collision" web/models/gofa/gofa.urdf`
Expected: `.glb` files + `.urdf` present in each dir; `gofa.urdf` has 7 `filename=` refs and `0` `collision` occurrences.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_web_models.py web/models
git commit -m "$(cat <<'EOF'
feat(web): export script + baked 3D model bundles for the console

scripts/export_web_models.py converts each arm's visual meshes to glTF
(meters scale, collisions stripped, STL colors baked from the URDF) and
writes self-contained web/models/<arm>/ bundles, committed so the server
serves them without the robot_descriptions cache.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Vendor the three.js + urdf-loader libraries

**Files:**
- Create: `web/vendor/three.module.js`, `web/vendor/jsm/controls/OrbitControls.js`, `web/vendor/jsm/loaders/GLTFLoader.js` (+ any transitive `three/addons/` deps), `web/vendor/urdf-loader/URDFLoader.js`, `web/vendor/urdf-loader/URDFClasses.js` (+ any transitive relative deps).

**Interfaces:**
- Produces: vendored ESM resolvable by the import map in Task 4 — bare specifier `three` → `/vendor/three.module.js`, `three/addons/` → `/vendor/jsm/`, `urdf-loader` → `/vendor/urdf-loader/URDFLoader.js`.

- [ ] **Step 1: Download the entry-point files and auto-resolve their dependency closure**

Run (needs network; pins three 0.160.1 + urdf-loader 0.12.3):

```bash
cd /Users/samarthmahapatra/Developer/abb_foga
V=0.160.1; UV=0.12.3
mkdir -p web/vendor/jsm/controls web/vendor/jsm/loaders web/vendor/urdf-loader
curl -fsSL "https://unpkg.com/three@$V/build/three.module.js" -o web/vendor/three.module.js
curl -fsSL "https://unpkg.com/three@$V/examples/jsm/controls/OrbitControls.js" -o web/vendor/jsm/controls/OrbitControls.js
curl -fsSL "https://unpkg.com/three@$V/examples/jsm/loaders/GLTFLoader.js" -o web/vendor/jsm/loaders/GLTFLoader.js
# pull any further three/addons/* modules the addons import, transitively
while :; do
  miss=$(grep -rhoE "three/addons/[A-Za-z0-9_./-]+\.js" web/vendor/jsm | sort -u | while read m; do
    f="web/vendor/jsm/${m#three/addons/}"; [ -f "$f" ] || echo "$m"; done)
  [ -z "$miss" ] && break
  for m in $miss; do f="web/vendor/jsm/${m#three/addons/}"; mkdir -p "$(dirname "$f")"
    curl -fsSL "https://unpkg.com/three@$V/examples/jsm/${m#three/addons/}" -o "$f"; done
done
curl -fsSL "https://unpkg.com/urdf-loader@$UV/src/URDFLoader.js" -o web/vendor/urdf-loader/URDFLoader.js
curl -fsSL "https://unpkg.com/urdf-loader@$UV/src/URDFClasses.js" -o web/vendor/urdf-loader/URDFClasses.js
# pull any further relative deps urdf-loader imports, transitively
while :; do
  miss=$(grep -rhoE "from ['\"]\\./[A-Za-z0-9_./-]+\.js" web/vendor/urdf-loader | sed -E "s/.*\.\///; s/['\"].*//" | sort -u | while read m; do
    [ -f "web/vendor/urdf-loader/$m" ] || echo "$m"; done)
  [ -z "$miss" ] && break
  for m in $miss; do curl -fsSL "https://unpkg.com/urdf-loader@$UV/src/$m" -o "web/vendor/urdf-loader/$m"; done
done
```

Expected: downloads succeed; the loops terminate (no missing modules remain).

- [ ] **Step 2: Verify the closure is complete and self-consistent**

Run:

```bash
# every bare 'three' / 'three/addons/...' / relative import in the vendored tree resolves to a present file
grep -rhoE "from ['\"](three(/addons/[A-Za-z0-9_./-]+\.js)?|\\.[A-Za-z0-9_./-]+\.js|urdf-loader)['\"]" web/vendor | sort -u
echo "--- files present ---"; find web/vendor -name '*.js' | sort
echo "--- three.module.js exports? ---"; grep -c "export" web/vendor/three.module.js
```

Expected: every `three/addons/...` specifier listed has a matching file under `web/vendor/jsm/`; bare `three` and `urdf-loader` appear (resolved by the import map, not files-relative); `three.module.js` reports a non-zero export count.

- [ ] **Step 3: Commit**

```bash
git add web/vendor
git commit -m "$(cat <<'EOF'
chore(web): vendor three.js 0.160.1 + urdf-loader 0.12.3 (ESM)

Vendored under web/vendor/ so the console's 3D viewer runs offline on the
LAN with no CDN; wired via a native import map (no bundler).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Serve `web/vendor` and `web/models` statically

**Files:**
- Modify: `scripts/api_server.py` (add `StaticFiles` mounts beside the dashboard route)
- Modify: `scripts/api_smoketest.py:413-454` (`test_e2e_subprocess`)

**Interfaces:**
- Consumes: `web/models/<arm>/<arm>.urdf` (Task 1), `web/vendor/three.module.js` (Task 2).
- Produces: `GET /vendor/<path>` and `GET /models/<path>` return static files (unauthenticated, like `/`). Routes live at the **root** app, shared by both arms in the multi-app.

- [ ] **Step 1: Add the import for `StaticFiles`**

In `scripts/api_server.py`, modify line 18:

```python
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
```

- [ ] **Step 2: Mount the static dirs next to the dashboard route**

In `scripts/api_server.py`, replace the dashboard block (currently lines 68-71):

```python
        # Serve the static web console (talks to the robot only via the API endpoints).
        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(_DASHBOARD)
```

with:

```python
        # Serve the static web console (talks to the robot only via the API endpoints).
        # Vendored 3D libs + baked model bundles are static assets, served unauthenticated
        # like the dashboard shell itself — the bearer token still gates all state/control.
        _WEB = os.path.join(_ROOT, "web")
        for _route, _sub in (("/vendor", "vendor"), ("/models", "models")):
            _d = os.path.join(_WEB, _sub)
            if os.path.isdir(_d):
                app.mount(_route, StaticFiles(directory=_d), name=_sub)

        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(_DASHBOARD)
```

- [ ] **Step 3: Add the failing assertions to the e2e test**

In `scripts/api_smoketest.py`, in `test_e2e_subprocess`, immediately after line 454
(`assert max(abs(a - b) ... ) < 1e-6`) and before the `finally:` on line 455, add:

```python
            # static web assets: dashboard shell + vendored 3D libs + baked model bundle
            assert cl.get("/").status_code == 200
            assert cl.get("/vendor/three.module.js").status_code == 200
            r_urdf = cl.get("/models/ur15/ur15.urdf")
            assert r_urdf.status_code == 200 and "<robot" in r_urdf.text
```

- [ ] **Step 4: Run the e2e test to verify it passes**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: ends with `PASS test_e2e_subprocess` and all other `PASS` lines (the subprocess `sim.py api ur15` now serves `/vendor/*` and `/models/*`).

- [ ] **Step 5: Commit**

```bash
git add scripts/api_server.py scripts/api_smoketest.py
git commit -m "$(cat <<'EOF'
feat(api): serve web/vendor and web/models static assets

Mounts the vendored 3D libs and baked model bundles at /vendor and /models
on the root app (shared by both arms in multi mode); e2e smoketest asserts
the dashboard shell, a vendored lib, and a model URDF are all served.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Dashboard — viewer panel, import map, ES-module viewer, and hooks

**Files:**
- Modify: `web/dashboard.html` (CSS ~line 145, import map in `<head>`, panel markup at left-column top ~line 306, hook calls in `connect`/`selectArm`/`onArmState`/`disconnect`, module `<script>` before `</body>`)

**Interfaces:**
- Consumes: `RobotState` from `onArmState(name, s)` (`s.q`, `s.gripper_frac`); the active arm name `active`; the API origin `base`. Static `/models/<arm>/…` (Task 3), import map → vendored libs (Task 2).
- Produces: a global `window.viewer3d` with `{ load(arm, origin), update(state), dispose(), ready }`; all hooks optional-chained so a viewer failure is inert.

- [ ] **Step 1: Add the import map in `<head>`**

In `web/dashboard.html`, immediately after the Google-Fonts `<link …rel="stylesheet">` (line 9), add:

```html
<script type="importmap">
{ "imports": {
  "three": "/vendor/three.module.js",
  "three/addons/": "/vendor/jsm/",
  "urdf-loader": "/vendor/urdf-loader/URDFLoader.js"
}}
</script>
```

- [ ] **Step 2: Add the viewer CSS**

In `web/dashboard.html`, just before the `/* ---------- telemetry readouts ---------- */` comment (line 145), add:

```css
  /* ---------- 3D viewer ---------- */
  .viewer3d{position:relative;width:100%;height:340px;overflow:hidden;
    background:radial-gradient(120% 120% at 50% 0%, #0d1217, #070809)}
  .viewer3d canvas{display:block}
  .viewer3d-note{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
    color:var(--dim);pointer-events:none}
```

- [ ] **Step 3: Add the viewer panel at the top of the left column**

In `web/dashboard.html`, immediately after `<div class="col">` (line 306, the LEFT column) and before the Telemetry `<section>` (line 307), add:

```html
    <section class="panel" style="animation-delay:.02s">
      <div class="phead"><span class="idx">00</span> 3D View <span class="tag">live</span></div>
      <div class="pbody" style="padding:0">
        <div id="viewer3d" class="viewer3d">
          <div id="viewer3dNote" class="viewer3d-note">connect to view</div>
        </div>
      </div>
    </section>
```

- [ ] **Step 4: Add the four viewer hook calls into the classic script**

In `web/dashboard.html`:

(a) In `connect()`, after `openStreams();` (line 604), add:
```javascript
    if(window.viewer3d) window.viewer3d.load(active, base);
```

(b) In `selectArm(name)`, after `if(lastStates[active]) render(lastStates[active]);` (line 655), add:
```javascript
    if(window.viewer3d) window.viewer3d.load(active, base);
```

(c) In `onArmState(name, s)`, change the active-arm branch (line 691) from:
```javascript
  if(name===active) render(s);             // only the selected arm drives the panels
```
to:
```javascript
  if(name===active){ render(s);            // only the selected arm drives the panels
    if(window.viewer3d) window.viewer3d.update(s); }
```

(d) In `disconnect()`, after `setLed("off","OFFLINE"); updateLeaseUI(); log("sys","disconnected");` (line 629), add:
```javascript
  if(window.viewer3d) window.viewer3d.dispose();
```

- [ ] **Step 5: Add the ES-module viewer before `</body>`**

In `web/dashboard.html`, immediately before the closing `</body>` tag, add:

```html
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import URDFLoader from 'urdf-loader';

const JOINTS = {
  ur15: ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
  gofa: ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"],
};
const FINGER_JOINT = "robotiq_hande_left_finger_joint";
const FINGER_OPEN = 0.025;            // m; gripper_frac 0=open..1=closed

const mount = document.getElementById("viewer3d");
const note  = document.getElementById("viewer3dNote");
let renderer, scene, camera, controls;
let robot = null, hande = null, tool0 = null, curArm = null, loadTok = 0;

const showNote = m => { if(note){ note.textContent = m || ""; note.style.display = m ? "" : "none"; } };

function init(){
  try { renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true }); }
  catch(e){ showNote("3D unavailable (no WebGL)"); return false; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  mount.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.up.set(0, 0, 1);                       // URDF is Z-up
  camera.position.set(1.2, -1.2, 1.0);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.4);
  scene.add(new THREE.HemisphereLight(0xddeeff, 0x222428, 1.1));
  const dir = new THREE.DirectionalLight(0xffffff, 1.4);
  dir.position.set(2, -2, 3); scene.add(dir);
  const grid = new THREE.GridHelper(2, 20, 0x2a3640, 0x1d262e);
  grid.rotation.x = Math.PI / 2;                // grid into the XY plane (Z up)
  scene.add(grid);
  const resize = () => {
    const w = mount.clientWidth, h = mount.clientHeight || 340;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  };
  resize();
  new ResizeObserver(resize).observe(mount);
  (function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
  return true;
}

function makeLoader(){
  const mgr = new THREE.LoadingManager();
  const gltf = new GLTFLoader(mgr);
  const loader = new URDFLoader(mgr);
  loader.loadMeshCb = (path, manager, done) =>
    gltf.load(path, g => done(g.scene), undefined, err => { console.warn("mesh load failed", path, err); done(null); });
  return loader;
}

function clearRobot(){ if(robot) scene.remove(robot); robot = null; hande = null; tool0 = null; }

function frameCamera(){
  const box = new THREE.Box3().setFromObject(robot);
  if(box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const r = box.getSize(new THREE.Vector3()).length() * 0.6 || 1;
  controls.target.copy(c);
  camera.position.set(c.x + r, c.y - r, c.z + r * 0.8);
  camera.near = r / 100; camera.far = r * 20; camera.updateProjectionMatrix();
}

function load(arm, origin){
  if(!renderer) return;                          // init failed → inert
  if(arm === curArm && robot) return;
  curArm = arm;
  const tok = ++loadTok;                          // guard against overlapping switches
  showNote("loading model…");
  clearRobot();
  const baseUrl = (origin || location.origin).replace(/\/$/, "");
  makeLoader().load(`${baseUrl}/models/${arm}/${arm}.urdf`, obj => {
    if(tok !== loadTok) return;                   // superseded
    robot = obj; scene.add(robot); showNote(""); frameCamera();
    if(arm === "ur15"){
      tool0 = robot.links && robot.links["tool0"];
      makeLoader().load(`${baseUrl}/models/ur15/hande.urdf`, h => {
        if(tok !== loadTok) return;
        hande = h; (tool0 || scene).add(hande);
      }, undefined, () => {});                     // gripper is best-effort
    }
  }, undefined, err => {
    if(tok !== loadTok) return;
    console.warn("URDF load failed", err); showNote("3D model unavailable");
  });
}

function update(s){
  if(!robot || !s || !s.q) return;
  const names = JOINTS[curArm] || [];
  for(let i = 0; i < names.length && i < s.q.length; i++)
    if(typeof s.q[i] === "number") robot.setJointValue(names[i], s.q[i]);
  if(hande && typeof s.gripper_frac === "number")
    hande.setJointValue(FINGER_JOINT, (1 - s.gripper_frac) * FINGER_OPEN);
}

function dispose(){ clearRobot(); curArm = null; ++loadTok; showNote("connect to view"); }

window.viewer3d = { load, update, dispose, ready: init() };
</script>
```

- [ ] **Step 6: Syntax-check the module**

Run (extracts the module body to a temp file and validates ES-module syntax — `node --check` is syntax-only, so the bare imports don't need to resolve):

```bash
./robot_control/bin/python - <<'PY'
import re, pathlib
html = pathlib.Path("web/dashboard.html").read_text()
m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
assert m, "module script not found"
pathlib.Path("/tmp/viewer3d.mjs").write_text(m.group(1))
print("extracted", len(m.group(1)), "chars")
PY
node --check /tmp/viewer3d.mjs && echo "JS OK"
```

Expected: `extracted …` then `JS OK`.

- [ ] **Step 7: Confirm the static structure is wired**

Run:
```bash
grep -c 'type="importmap"' web/dashboard.html
grep -c 'window.viewer3d' web/dashboard.html
grep -c 'id="viewer3d"' web/dashboard.html
```
Expected: `1`, `5` (one assignment + four hook calls), `1`.

- [ ] **Step 8: Sanity-run the existing suites (no regressions)**

Run: `./robot_control/bin/python scripts/api_smoketest.py && ./robot_control/bin/python scripts/control_smoketest.py`
Expected: both end with all `PASS` / `ALL … PASSED`.

- [ ] **Step 9: Commit**

```bash
git add web/dashboard.html
git commit -m "$(cat <<'EOF'
feat(web): client-side 3D arm viewer in the console (three.js)

Top-left "3D View" card renders the live arm with urdf-loader off the
baked model bundle, driven entirely by the existing telemetry WS (joints
+ Hand-E gripper). View-only OrbitControls — no robot-driving controls.
Wired via a native import map; all hooks optional-chained so a viewer
failure (no WebGL / missing model) never breaks the rest of the console.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md` (web console section + project-layout tree), `API.md`, `README.md`, `usage.html`

**Interfaces:** none (docs only).

- [ ] **Step 1: CLAUDE.md — project layout tree**

In `CLAUDE.md`, under the `scripts/` tree, after the `api_smoketest.py` line, add:
```
│   ├── export_web_models.py    #   bake web/models/<arm>/ glTF bundles for the console 3D viewer
```
Under the assets section, after the `urdf/` line, add:
```
├── web/                        # static remote console: dashboard.html + vendor/ (three.js, urdf-loader) + models/ (baked glTF bundles)
```
(If a `web/` line already exists, extend its comment to mention `vendor/` + `models/` rather than adding a second line.)

- [ ] **Step 2: CLAUDE.md — web console section**

In `CLAUDE.md`, in the **Web console** paragraph, append a sentence describing the viewer:
```
A **3D View** card at the top of the left column renders the live arm in three.js (vendored urdf-loader + a baked glTF bundle under `web/models/<arm>/`, served at `/models`), driven entirely off the open `WS /telemetry` stream (joints + the Hand-E gripper on the UR15). It is **display-only** — OrbitControls move the camera, nothing drives the robot — and degrades to a quiet "3D model unavailable" note if WebGL or the bundle is missing. Bundles are regenerated with `scripts/export_web_models.py` (vendored libs live in `web/vendor/`, wired via a native import map — no build step, no CDN).
```

- [ ] **Step 3: API.md — note the static asset routes**

In `API.md`, in the web-console / server section, add:
```
The server also serves static assets for the console at **`/vendor/…`** (vendored
three.js + urdf-loader) and **`/models/<arm>/…`** (baked glTF arm bundles),
unauthenticated like the dashboard shell. The 3D viewer reads live joint state
from the existing `WS /telemetry` — it adds no new API endpoint.
```

- [ ] **Step 4: README.md — vendored sources note**

In `README.md`, in the "Vendored third-party sources" section, add:
```
- `web/vendor/` — three.js 0.160.1 (`build/three.module.js` + `examples/jsm` addons) and `urdf-loader` 0.12.3 (ESM), for the console's client-side 3D viewer. Regenerate by re-downloading the pinned versions (see `docs/superpowers/plans/2026-06-18-web-console-3d-viewer.md`).
- `web/models/` — per-arm glTF bundles baked from the URDFs by `scripts/export_web_models.py`.
```

- [ ] **Step 5: usage.html — mention the 3D view**

In `usage.html`, where the dashboard/console panels are described, add a line noting the **3D View** card renders the live arm client-side off telemetry (display-only, no robot controls). Match the surrounding markup style.

- [ ] **Step 6: Verify docs reference real paths**

Run:
```bash
grep -rl "export_web_models.py" CLAUDE.md README.md && \
grep -l "3D View" CLAUDE.md usage.html && \
grep -l "/models" API.md && echo "docs OK"
```
Expected: prints the matching files then `docs OK`.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md API.md README.md usage.html
git commit -m "$(cat <<'EOF'
docs: document the console 3D arm viewer

CLAUDE.md (console section + layout tree), API.md (static /vendor /models
routes), README (vendored web libs + baked models), usage.html.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §1 Asset pipeline (`export_web_models.py`, glTF, strip collisions, bake STL color, `--check`) → Task 1. ✓
- §2 Vendored libs + import map → Task 2 (vendor) + Task 4 Steps 1. ✓
- §3 Viewer component (panel top-left, Z-up scene, OrbitControls, lazy load, Hand-E slaved to tool0, frame-to-fit) → Task 4 Steps 2-5. ✓
- §4 Data flow (telemetry `q` + `gripper_frac`, rAF) → Task 4 Step 5 (`update`) + Step 4 hook (c). ✓
- §5 Serving (`StaticFiles` for `web/`, root app, unauth, dashboard preserved) → Task 3. ✓
- §6 Error handling (missing model / no WebGL → placeholder, switch teardown, dispose on disconnect) → Task 4 (`showNote`, guards, `dispose`). ✓
- Testing (`--check`, api_smoketest static asserts, `node --check`, docs) → Tasks 1/3/4/5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output.

**Type/name consistency:** `window.viewer3d.{load,update,dispose,ready}` defined in Task 4 Step 5 and called identically in Step 4 (a-d). Joint-name arrays, `FINGER_JOINT`, `FINGER_OPEN` match the Global Constraints. `web/models/<arm>/<arm>.urdf` + `hande.urdf` naming consistent across Tasks 1, 3, 4. Static routes `/vendor`, `/models` consistent across Tasks 2, 3, 4, 5.
```
