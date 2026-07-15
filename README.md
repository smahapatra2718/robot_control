# robot_control

Browser-based teleop for two robot arms sharing the same [viser](https://github.com/nerfstudio-project/viser) + [pyroki](https://github.com/chungmin99/pyroki) stack:

- **`teleop_ur15.py`** — Universal Robots UR15 over RTDE (`ur_rtde`), streaming `servoJ`, with a Robotiq Hand-E gripper on the wrist (mesh + Open/Close via the Grippers URCap socket; see `hande_gripper.py` / `verify_hande.py`).
- **`teleop_gofa_egm.py`** — ABB GoFa CRB 15000 over Externally Guided Motion (EGM): joint targets stream over UDP to a RAPID supervisor (`PyEgm.mod`), with RWS (`abb_rws.py`) for mastership and the start/stop flag. Slider-unified like the UR15, with a TCP-speed cap.
- **`play_trajectory.py <robot>/<name> [more...]`** — headless replay of a saved trajectory (the `<robot>/` prefix — `ur15/` or `gofa/` — picks the folder and the arm), no viser. Pass several names to chain them into one continuous motion (gripper calibrated once at the start, not between). `--dry-run` to preview, `--no-confirm` to skip the prompt.
- **`teleop.py [name] [--robot ur|gofa]`** — headless CLI trajectory *recorder*: hand-guide the arm in free-drive, capture waypoints with single keypresses, save in the same format `play_trajectory.py` replays.

Both teleop scripts share the same UI (viser scene + 6-DoF gizmo + waypoints), the same seeded IK (`pyroki_snippets/_solve_ik_seeded.py`), and the same trapezoidal play loop. All four entry points pull shared config + helpers from **`robot_common.py`**. See [`CLAUDE.md`](CLAUDE.md) for the full architecture, controller bring-up notes, tunables, and hard-won gotchas.

```bash
# real hardware:
uv run scripts/real.py ur15      # or: gofa | play <robot>/<name> | teleop
# offline simulation (no robot, no network):
uv run scripts/sim.py  ur15      # same targets — runs the real scripts vs a fake arm
```

Remote control API (offline): `ROBOT_API_TOKEN=secret uv run scripts/sim.py api ur15` → HTTP+WebSocket on `:8000` (see `CLAUDE.md` → Remote API).

Then open the printed `http://localhost:8080`. The four scripts (`teleop_ur15.py`, `teleop_gofa_egm.py`, `play_trajectory.py`, `teleop.py`) still run directly too.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — `pyproject.toml` declares
them, `uv.lock` pins them. Install uv, then:

```bash
# ur_rtde publishes no wheels, so it always compiles from source. On macOS that
# needs boost@1.85 (1.87+ made Boost.System header-only and breaks the build):
brew install boost@1.85
export BOOST_ROOT=/opt/homebrew/opt/boost@1.85
export CMAKE_PREFIX_PATH=/opt/homebrew/opt/boost@1.85

uv sync
```

That's the whole install. `uv sync` builds `.venv/` from `uv.lock`, which pins everything
exactly — including uv's own CPython 3.13 (no system or conda Python is involved), the
vendored `pyroki_src/` as an editable install, and `jaxls` at an exact git commit.

Run anything with `uv run`; it re-syncs first if the lock changed, and nothing needs activating:

```bash
uv run scripts/sim.py ur15
uv run scripts/sim_smoketest.py
```

`.venv/` (~655M) is **not** committed; `uv.lock` is, so every machine resolves identically.
To change a dependency, edit `pyproject.toml` and run `uv sync` (or `uv add <pkg>`) — this
regenerates `uv.lock`, which should be committed alongside.

See `CLAUDE.md` → "Dependencies" and "Other gotchas" for the why behind each step.

## Git LFS

Robot meshes (`.stl`, `.dae`) and images are stored via [Git LFS](https://git-lfs.com). After cloning:

```bash
git lfs install
git lfs pull
```

## Vendored third-party sources

These directories are vendored copies (their upstream `.git` history was stripped). Pinned to:

| Directory | Upstream | Commit | License |
|---|---|---|---|
| `pyroki_src/` | https://github.com/chungmin99/pyroki | `388e43e` | see dir |
| `abb_desc/` | https://github.com/ros-industrial/abb | `45f4769` | see dir |
| `robotiq_hande_description/` | https://github.com/macmacal/robotiq_hande_description | `5ae8b97` | Apache-2.0 |
| `web/vendor/three.module.js` + `jsm/` | https://github.com/mrdoob/three.js | `0.160.1` | MIT |
| `web/vendor/urdf-loader/` | https://github.com/gkjohnson/urdf-loaders | `0.12.3` | Apache-2.0 |

`pyroki_snippets/` is a copy of `pyroki_src/examples/pyroki_snippets/` plus the custom `_solve_ik_seeded.py`.

`web/vendor/` holds the console's 3D-viewer libraries (vanilla ESM, no build step) — re-download the
pinned versions to regenerate. `web/models/` holds per-arm glTF bundles baked from the URDFs by
`scripts/export_web_models.py` (committed so the server needs no robot_descriptions cache to serve them).
