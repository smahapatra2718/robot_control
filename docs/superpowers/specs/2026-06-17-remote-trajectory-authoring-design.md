# Remote trajectory authoring — design

**Date:** 2026-06-17
**Status:** approved (pre-implementation)

## Overview

Add full trajectory **creation** (teach + save) through the remote API and web
console, and replace the console's free-text Play field with a **type-to-filter
dropdown** that lists the trajectories already saved for the selected arm.

Capturing a waypoint is a *client-side* act: read the live `RobotState`
(`q`, `pose.pos`, `pose.wxyz`, `gripper_frac`) and append it to an in-browser
list. So the only new server work is a small REST surface over the existing
`save_trajectory` / `load_trajectory` helpers plus a new `list_trajectories`.

Combined with the free-drive shipped previously, the remote teach loop becomes:
**free-drive → hand-move → Capture → (set gripper, capture) → … → name → Save →
pick in Play**. Jogging without free-drive (Move Joints/Pose) is an equally valid
way to position between captures.

## Goals

- Author a trajectory entirely from the dashboard: capture live waypoints into a
  reviewable list, delete individual waypoints, clear, load an existing
  trajectory back into the editor to append/re-save, name it, and save it to
  `trajectories/<robot>/<name>.json`.
- A Play picker that auto-lists the arm's saved trajectories and filters as you
  type, matching the console theme.
- A complete, safe REST surface (`GET`/`POST`/`DELETE`) usable from curl/CLI too.

## Non-goals (out of scope for this change)

- Reordering waypoints; per-waypoint pose/gripper editing beyond delete.
- A dashboard button to delete a whole saved trajectory file (the DELETE
  endpoint exists, but is not surfaced in the UI this round).
- Overwrite confirmation dialogs (save is upsert / last-write-wins).
- Server-side capture endpoint (capture stays client-side off telemetry).

## API endpoints

Added in `lib/robot_api.py` `build_app`, so they are arm-scoped and auto-namespace
under `/ur15`, `/gofa` in the multi-arm server with no extra work.

| Method | Path | Lease? | Success | Purpose |
|---|---|---|---|---|
| `GET` | `/trajectories` | — (auth only) | `200` | List saved names for this arm |
| `GET` | `/trajectories/{name}` | — (auth only) | `200` | Return one trajectory's JSON |
| `POST` | `/trajectories` | ✔ | `200` | Save `{name, waypoints}` |
| `DELETE` | `/trajectories/{name}` | ✔ | `200` | Delete the file |

- `GET /trajectories` → `{"trajectories": ["_sample_ur15", ...]}` — sorted
  basenames (no `.json`) in `trajectories/<robot>/`. Missing folder → `[]`.
- `GET /trajectories/{name}` → the stored `{robot, created, waypoints}` object.
  Unknown name → `404`; invalid name → `422`.
- `POST /trajectories` body `{"name": str, "waypoints": [...]}` → validates name +
  waypoints, writes via `rc.save_trajectory(name, controller.robot_name, waypoints)`,
  overwriting if present. Returns `{"saved": true, "name": name}`. Lease-gated
  (`423` without a valid `X-Lease`).
- `DELETE /trajectories/{name}` → lease-gated; removes the file. Unknown → `404`;
  invalid name → `422`. Returns `{"deleted": true, "name": name}`.

`/play` is unchanged; the dropdown feeds it a `name`.

## `lib/robot_common.py` additions

- `list_trajectories(robot, traj_dir=TRAJ_DIR) -> list[str]` — sorted basenames of
  `*.json` in `traj_dir/<robot>/`; `[]` if the folder is absent. Stdlib only.
- `validate_traj_name(name) -> None` — raises `ValueError` unless `name` matches
  `^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$`. This rejects `/`, `\`, `..`, a leading
  dot, the empty string, and names over 64 chars — i.e. path traversal and hidden
  files. Called inside `save_trajectory` (defense-in-depth for every caller,
  including the teleop save buttons) and by the GET/DELETE endpoints. The API maps
  `ValueError → 422`.

## Validation (server-side, in the API)

Don't trust the client. `POST /trajectories` rejects (`422`) unless:
- `name` passes `validate_traj_name`.
- `waypoints` is a non-empty list, and each item is an object with:
  - `pos`: list of 3 finite numbers,
  - `wxyz`: list of 4 finite numbers,
  - `q`: `null` or a list of `NUM_JOINTS` (6) finite numbers,
  - `grip`: `null` or a finite number.

Reuses the existing `_check_vec` finiteness helper where applicable.

## Dashboard (`web/dashboard.html`)

New **Record** panel and a themed Play dropdown. Final right-column order:
`04 Teleop · 05 Gripper · 06 Record · 07 Play` (drive → record → replay).

### Record panel
- `Capture waypoint` button — snapshots `lastStates[active]` into a JS waypoint
  array as `{q, pos, wxyz, grip}` (grip is `gripper_frac`, `null` on GoFa). If no
  telemetry yet, logs a warning and does nothing.
- Scrollable waypoint list: each row shows `#`, a short `pos`, `grip%` (or `—`),
  and an `×` to delete that waypoint.
- `Clear` empties the list.
- Name field + `Load` (GET `/trajectories/{name}` → replace the list with the
  loaded waypoints) + `Save` (POST `/trajectories`).
- Lease: Capture / Load / Clear need **no** lease (read + client-side). **Save is
  lease-gated** — disabled with a hint when the lease isn't held.
- After a successful Save, refetch the Play dropdown list so the new name appears.

### Play dropdown (themed custom combobox)
- Styled text input + a filtered list populated from `GET /trajectories`.
- Typing filters the list (case-insensitive substring); click or ↑/↓+Enter
  selects; the chosen name feeds the existing `Send /play`.
- List refetched on connect, on arm-switch, and after a save.
- Replaces the current free-text `#trajName` input.

## Data shapes

Waypoint (unchanged from the saved format):
```json
{ "q": [6 floats] | null, "pos": [x,y,z], "wxyz": [w,x,y,z], "grip": float | null }
```
Saved file (unchanged): `{ "robot": str, "created": iso8601, "waypoints": [...] }`.

## Testing

`scripts/api_smoketest.py` — `test_trajectories` (in-process TestClient over the
sim, both reachable arms as needed):
- `GET /trajectories` is auth-only and includes `_sample_ur15`.
- `GET /trajectories/_sample_ur15` returns waypoints.
- `POST /trajectories` → `423` without a lease; with the lease → `200`.
- After save, the new name appears in `GET /trajectories` and `GET
  /trajectories/{name}` loads it back with the same waypoints.
- `POST` with a bad name (`"../x"`) → `422`; with malformed waypoints → `422`.
- `DELETE /trajectories/{name}` → `423` without a lease, `200` with; the name then
  disappears and a `GET` of it → `404`.
- Uses a throwaway name (e.g. `_api_test_traj`) created and deleted within the
  test; never writes or removes the committed `_sample_*` fixtures.

`scripts/api_smoketest.py` — a quick `robot_common` unit check (in `test_trajectories`
or its own `test_traj_helpers`): `list_trajectories("ur15")` contains
`_sample_ur15`; `validate_traj_name` accepts `"pick_place-1"` and raises
`ValueError` on `"../evil"`, `"a/b"`, and `""`.

All existing smoke tests must still pass.

## Docs

- `API.md` — new endpoint sections + table rows; note save/delete are lease-gated,
  list/get are auth-only.
- `CLAUDE.md` — Remote API endpoint list + the console Record panel and Play
  dropdown in the Web console paragraph.
- `usage.html` — endpoint cards for the four new routes; update the bundled-console
  description to mention recording + the Play dropdown.

## Security notes

- Filename validation (`validate_traj_name`) is the path-traversal guard and runs
  before any filesystem path is constructed, in `save_trajectory` and the GET/
  DELETE endpoints.
- Save/delete require the lease, so trajectory files can't be written/removed by a
  client that doesn't hold write control.
- The endpoints only ever touch `trajectories/<robot>/`; `name` can't escape it.
