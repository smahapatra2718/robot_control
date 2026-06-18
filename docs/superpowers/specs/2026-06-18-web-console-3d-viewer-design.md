# Web console 3D arm viewer — design

**Date:** 2026-06-18
**Status:** approved, pre-implementation

## Goal

Add a lightweight, **client-side** 3D render of the live arm to the web console
(`web/dashboard.html`), so an operator can see the robot's current configuration
the way they would in viser — without running a viser server and without
exposing any controls that drive the robot. The render mirrors the live joint
state already arriving over the telemetry WebSocket; it is **display-only**.

"Lightweight" here means the *approach*: self-contained three.js in the existing
vanilla-HTML console (no viser process, no build step, no robot-manipulation
gizmo). It does **not** mean abstract geometry — the user chose faithful meshes.

## Decisions (locked during brainstorming)

- **Fidelity:** real robot meshes (viser-like), not primitive shapes.
- **Placement:** a "3D View" card at the **top of the left column** of the
  existing two-column card grid.
- **Mesh format:** convert visual meshes to **glTF `.glb`** so the browser needs
  a single loader (GLTFLoader). Link colors come from the URDF `<material>`.
- **Gripper:** include the **Hand-E** on the UR15 render (slaved to `tool0`,
  finger driven by `gripper_frac`) for viser-parity.
- **Libraries:** **vendored** three.js + OrbitControls + GLTFLoader +
  urdf-loader, wired via a native **import map**. No CDN (console runs on a
  LAN / direct cable and must work offline), no bundler.
- **Controls:** OrbitControls = camera rotate/zoom/pan **only**. No gizmo, no
  endpoint is called from the viewer to move the arm.

## Architecture

### 1. Asset pipeline — `scripts/export_web_models.py` (new)

The console cannot synthesize a URDF (UR15 has no local URDF — it is built at
runtime from `robot_descriptions`) or read the home-directory mesh cache at
request time. So we bake a **self-contained model bundle into the repo**,
consistent with the project's existing pattern of vendoring third-party asset
trees (`abb_desc/`, `robotiq_hande_description/`).

The script is run once per machine and its output is committed. For each arm it:

1. Loads the source URDF:
   - **GoFa** — `urdf/crb15000_5_95.urdf` (local; meshes in
     `abb_desc/abb_crb15000_support/meshes/crb15000_5_95/visual/*.stl`).
   - **UR15** — `robot_descriptions.loaders.yourdfpy.load_robot_description("ur15_description")`
     (meshes `~/.cache/robot_descriptions/ur_description/meshes/ur15/visual/*.dae`).
   - **Hand-E** — `urdf/hande.urdf` (meshes in `robotiq_hande_description/`).
2. Resolves every **visual** mesh and converts it to glTF `.glb` via trimesh,
   **preserving meters scale** (no rescale — URDF link origins are in meters and
   trimesh loads source units, so geometry lines up with the kinematics).
3. Writes a self-contained `<arm>.urdf` (and `hande.urdf` for UR15) with mesh
   references rewritten to **relative `.glb` paths** — no `package://`, no
   `file://` (avoids the yourdfpy `file://` gotcha and any package resolution in
   the browser).

Output layout:

```
web/models/
├── ur15/   { ur15.urdf, hande.urdf, *.glb }
└── gofa/   { gofa.urdf, *.glb }
```

`--check` mode (no writes) re-parses each committed `web/models/<arm>/*.urdf`
and asserts every referenced `.glb` exists — the Python-side test hook.

Collision meshes are **not** exported (visual only — smaller payload, the viewer
is cosmetic).

### 2. Libraries — vendored, no build step

Vendor under `web/vendor/`:

- `three.module.js` (three.js core, ESM)
- `OrbitControls.js`, `GLTFLoader.js` (three.js `examples/jsm` addons, ESM)
- `URDFLoader.js` (gkjohnson `urdf-loader`, ESM)

`dashboard.html` declares a native **import map** mapping `three`,
`three/addons/…`, and `urdf-loader` to those local files, and the viewer is a
`<script type="module">`. Import maps are natively supported by the modern
Chrome/Edge the console runs in; no bundler is introduced.

A custom `loadMeshCb` on URDFLoader routes all mesh loads through GLTFLoader
(every mesh is `.glb`).

### 3. Viewer component (in `web/dashboard.html`)

A new **"3D View"** `.panel` card inserted as the first panel of the left column.

- Fixed-height canvas (≈ 320–380 px) with a `ResizeObserver` to stay crisp.
- Z-up scene (URDF convention), a ground grid, hemisphere + directional light,
  neutral background matching the console theme.
- **OrbitControls** for camera rotate / zoom / pan only.
- On **connect / arm-select**, lazy-loads `/models/<arm>/<arm>.urdf` via
  URDFLoader. For **UR15** it additionally loads `hande.urdf` and re-parents it
  to the live `tool0` frame each tick (mirrors `teleop_ur15.py`'s slaved gripper).
- A one-time camera frame-to-fit after the model loads.

### 4. Data flow — reuses the open telemetry WebSocket

No new endpoint for live state. The viewer reads the **same `RobotState`** the
console already streams over `WS /telemetry`:

- Each animation frame, set the robot's six joint values from `state.q`
  (URDFLoader's `setJointValue` performs the FK).
- For UR15, map `gripper_frac` (0 open … 1 closed) to the Hand-E actuated finger
  joint, and update the slaved gripper transform from the URDF's current `tool0`.
- `requestAnimationFrame` render loop; joints snap to the most recent telemetry
  (no client-side interpolation needed — telemetry is already smooth).

### 5. Serving

`scripts/api_server.py` currently serves only `dashboard.html` (a `FileResponse`
at `/`). Add a **`StaticFiles` mount for the `web/` directory** so `/vendor/…`
and `/models/…` resolve. Mounted at the **root** app (shared by both arms in the
multi-app), served **unauthenticated** like the dashboard shell itself — the
geometry is not sensitive, and the bearer token still gates every state/control
endpoint. The existing `/` dashboard route is preserved.

### 6. Error handling

The viewer is strictly non-blocking and never breaks the rest of the console:

- Missing/failed model fetch or parse → the card shows a quiet
  "3D model unavailable" placeholder; all other panels work unchanged.
- No WebGL context → same placeholder.
- Arm switch tears down the current scene/model and loads the newly selected
  arm; disconnect disposes geometry/materials and stops the render loop.

## Testing

- **Python**
  - `export_web_models.py --check`: each committed `web/models/<arm>/*.urdf`
    parses and every referenced `.glb` exists on disk.
  - Extend `scripts/api_smoketest.py`: against the live `sim.py api` subprocess,
    assert `GET /models/<arm>/<arm>.urdf` → 200 and a known `GET /vendor/<file>`
    → 200 (static serving wired correctly).
- **JS**: structural checks where feasible (`node --check` on extracted module
  logic). The visual confirmation (meshes render, joints track, gripper
  animates) is the user's, in-browser on the Oracle PC — it cannot be verified
  headlessly.
- **Docs**: update CLAUDE.md (web console section, the new script + `web/models`
  / `web/vendor` in the layout tree, vendored-sources note), API.md, usage.html,
  README.

## Out of scope

- Any interactive manipulation (gizmo, drag-to-IK) in the 3D view — display only.
- Collision-mesh rendering.
- Embedding viser or running a viser/websocket scene server.
- Trajectory/waypoint frame overlays in 3D (possible follow-on).
```
