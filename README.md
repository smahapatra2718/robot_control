# robot_control

Browser-based teleop for two robot arms sharing the same [viser](https://github.com/nerfstudio-project/viser) + [pyroki](https://github.com/chungmin99/pyroki) stack:

- **`teleop_ur15.py`** — Universal Robots UR15 over RTDE (`ur_rtde`), streaming `servoJ`, with a Robotiq Hand-E gripper on the wrist (mesh + Open/Close via the Grippers URCap socket; see `hande_gripper.py` / `verify_hande.py`).
- **`teleop_gofa_egm.py`** — ABB GoFa CRB 15000 over Externally Guided Motion (EGM): joint targets stream over UDP to a RAPID supervisor (`PyEgm.mod`), with RWS (`abb_rws.py`) for mastership and the start/stop flag. Slider-unified like the UR15, with a TCP-speed cap.
- **`play_trajectory.py <name>`** — headless replay of a saved trajectory (UR15 or GoFa), no viser. `--dry-run` to preview, `--no-confirm` to skip the prompt.

Both share the same UI (viser scene + 6-DoF gizmo + waypoints), the same seeded IK (`pyroki_snippets/_solve_ik_seeded.py`), and the same trapezoidal play loop. See [`CLAUDE.md`](CLAUDE.md) for the full architecture, controller bring-up notes, tunables, and hard-won gotchas.

```bash
./robot_control/bin/python teleop_ur15.py   # or teleop_gofa_egm.py
```

Then open the printed `http://localhost:8080`.

## Setup (rebuild the venv)

The `robot_control/` virtualenv is **not** committed (655M, machine-specific). Recreate it:

```bash
python3.13 -m venv robot_control
./robot_control/bin/pip install numpy viser yourdfpy jaxlie jax jaxlib \
    robot_descriptions xacrodoc requests urllib3

# pyroki — install from the vendored source (no PyPI package exists)
./robot_control/bin/pip install -e ./pyroki_src

# ur_rtde — on macOS, build against boost@1.85 (1.87+ breaks the build):
brew install boost@1.85
BOOST_ROOT=/opt/homebrew/opt/boost@1.85 \
CMAKE_PREFIX_PATH=/opt/homebrew/opt/boost@1.85 \
  ./robot_control/bin/pip install ur_rtde==1.6.3
```

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

`pyroki_snippets/` is a copy of `pyroki_src/examples/pyroki_snippets/` plus the custom `_solve_ik_seeded.py`.
