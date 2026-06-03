# GoFa software lead-through (free-drive) — design

**Date:** 2026-06-03
**Goal:** Add a software "Free-drive (lead-through)" toggle to the GoFa teleop so the
arm goes compliant for hand-guiding from the browser — no physical lead-through
button — matching the UR15's free-drive checkbox. Capture then records the
hand-moved joints (already works via RWS polling).

## Background / why this is now possible

Earlier we deferred this because we couldn't find a programmatic lead-through API over
EGM/RWS. The RAPID reference resolves it: **`SetLeadThrough`** activates/deactivates
hand-guiding for a task's TCP robot (3HAC050917-001 / RW7 3HAC065038).

- `SetLeadThrough \On;` — engage compliance (default also orders `StopMove`, so it
  activates even while the supervisor task loops).
- `SetLeadThrough \Off;` — release (default also does `ClearPath` + `StartMove`).
- Valid when executed *inside the motion task* (our `T_ROB1`).
- Auto-cleared by the controller on motors-off / Reset RAPID — built-in safety.

This fits the existing supervisor pattern exactly: a second RWS-driven flag alongside
`egm_go`. Enabling it is a `PyEgm.mod` change pushed by re-running `install_gofa_egm.py`.

## Architecture

A new `PERS bool lead_go` flag mirrors `egm_go`. The supervisor's `WHILE TRUE` loop
waits on *either* flag and branches:

```rapid
PROC main()
  AccSet 50, 50;
  WHILE TRUE DO
    WaitUntil egm_go = TRUE OR lead_go = TRUE;
    IF lead_go = TRUE THEN
      SetLeadThrough \On;        ! compliant (default StopMove)
      WaitUntil lead_go = FALSE;
      SetLeadThrough \Off;       ! resume (default ClearPath + StartMove)
    ELSE
      ! ... existing EGM branch (EGMReset/GetId/SetupUC/ActJoint/RunJoint/Reset) ...
      egm_go := FALSE;
    ENDIF
  ENDWHILE
ENDPROC
```

The two branches are mutually exclusive in RAPID; the teleop UI also prevents both
flags being set at once.

## teleop_gofa_egm.py

- `RAPID_LEAD_FLAG_VAR = "lead_go"`.
- **Startup safety:** set `lead_go = FALSE` (next to the existing `egm_go = FALSE`).
- **`freedrive` Event** + **`gui_freedrive` checkbox** ("Free-drive (lead-through)"):
  - **On enable:** refuse if playing/live; set `egm_go = FALSE` then `lead_go = TRUE`
    over RWS; disable Plan / Play / Live / Execute; status: "Free-drive ON — hand-guide,
    then Capture".
  - **On disable:** set `lead_go = FALSE`; re-enable the buttons; status: "Free-drive off".
- **Stop button** also clears `freedrive`, unchecks the box, and sets `lead_go = FALSE`.
- **Mutual exclusion:** `gui_freedrive` disabled while playing or live; Plan/Play/Live/
  Execute disabled while free-driving (same shape as the UR15).
- **Clean exit:** best-effort `atexit` (or the existing shutdown path) sets
  `lead_go = FALSE` so Ctrl-C while compliant doesn't leave the arm loose (the controller
  also auto-clears on motors-off as a backstop).
- **Capture is unchanged** — RWS joint polling already tracks the hand-moved arm.

## install_gofa_egm.py

The `PYEGM_MOD` string (the supervisor source) gains the `lead_go` flag + branch above.
No other installer logic changes. Applying it = **re-run the installer** (Ctrl-C any
running teleop first to free mastership); it already unloads colliding modules and
reloads `PyEgm`.

## Risks / open questions (verify on hardware)

1. **GoFa support on RW7.** The RW6 manual restricts `SetLeadThrough` to YuMi
   (IRB 14000); RW7/OmniCore reportedly extends it to cobots, unconfirmed from docs.
   **Mitigation:** after the installer loads the module, check
   `/rw/rapid/tasks/T_ROB1/program/builderror` — a wrong/unsupported instruction shows
   there (same endpoint we used for the original bring-up). If `SetLeadThrough` is
   rejected for `ROB_1`, fall back to the hardware button (no regression) and explore
   the undocumented RWS lead-through route.
2. **Auto vs Manual.** EGM play needs Auto; lead-through may require Manual + enabling
   device per the safety config. The physical button already hand-guides in the current
   Auto setup, which suggests Auto is permitted — but confirm. If it needs Manual, the
   toggle still works there (teach in Manual, switch to Auto to replay) — document it.
3. **Leaving the arm compliant.** Handled by client-side `lead_go = FALSE` on
   disable/Stop/exit + the controller's motors-off auto-clear.

## Out of scope

- Linear/Cartesian-constrained lead-through (`SetLeadThrough \On` is all-axis/joint —
  fine for teaching waypoints).
- UR15 (already has software free-drive via `teachMode`).
- Gripper actions on the GoFa (no gripper).
