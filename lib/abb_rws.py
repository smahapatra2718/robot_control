"""
Minimal Robot Web Services (RWS) client for ABB OmniCore controllers
(GoFa CRB 15000 and other RobotWare 7+ controllers).

Just the endpoints we need for the teleop scaffold:
  - get_joints()        : read current joint positions (rad)
  - request_mastership(): grab/release RAPID mastership
  - set_rapid_bool() / get_rapid_data() : write/read RAPID variables
  - start_program() / stop_program() / get_execution_state()
  - get_controller_state() / set_motors_on()

Not a complete RWS wrapper. Uses requests + HTTP Basic auth (default credentials
"Default User"/"robotics" — change via constructor).

The execute path drives a small RAPID supervisor (PyEgm.mod, loaded by
install_gofa_egm.py) by flipping a bool flag; see teleop_gofa_egm.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import urllib3
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth


@dataclass
class RWSClient:
    host: str                                # e.g. "192.168.125.1"
    user: str = "Default User"
    password: str = "robotics"
    timeout: float = 5.0
    https: bool = True                       # OmniCore is HTTPS-only by default
    auth_scheme: str = "basic"               # "basic" (OmniCore) or "digest" (IRC5)
    _session: requests.Session = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # OmniCore ships with a self-signed cert. Suppress the warning spam.
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._session = requests.Session()
        if self.auth_scheme == "digest":
            self._session.auth = HTTPDigestAuth(self.user, self.password)
        else:
            self._session.auth = HTTPBasicAuth(self.user, self.password)
        self._session.verify = False
        self._session.headers.update({"Accept": "application/hal+json;v=2.0"})
        # Register an atexit hook so a Ctrl+C or crash still releases mastership.
        import atexit
        atexit.register(self._best_effort_release)

    def _best_effort_release(self) -> None:
        try:
            self.release_mastership()
        except Exception:
            pass

    @property
    def base(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}"

    # ---- raw helpers ----
    def _get(self, path: str, **kw: Any) -> dict[str, Any]:
        r = self._session.get(self.base + path, timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict[str, str] | None = None, **kw: Any) -> requests.Response:
        # OmniCore RWS 2.0 requires this specific content-type on POST/PUT.
        headers = kw.pop("headers", {})
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded;v=2.0")
        r = self._session.post(
            self.base + path, data=data or "", headers=headers, timeout=self.timeout, **kw
        )
        r.raise_for_status()
        return r

    # ---- state read ----
    def get_joints(self, mechunit: str = "ROB_1") -> list[float]:
        """Return current actuated joint angles in radians (6 elements for GoFa)."""
        data = self._get(f"/rw/motionsystem/mechunits/{mechunit}/jointtarget")
        # RWS 2.0 returns: {"state": [{"rax_1": "0.0", ..., "rax_6": "0.0"}, ...]}
        # RWS 1.x returns slightly different. We dig through common shapes.
        state = _find_jointtarget(data)
        return [math.radians(float(state[f"rax_{i + 1}"])) for i in range(6)]

    def get_controller_state(self) -> str:
        data = self._get("/rw/panel/ctrl-state")
        return _find_first_value(data, key="ctrlstate")

    def get_operation_mode(self) -> str:
        data = self._get("/rw/panel/opmode")
        return _find_first_value(data, key="opmode")

    def get_execution_state(self) -> str:
        data = self._get("/rw/rapid/execution")
        return _find_first_value(data, key="ctrlexecstate")

    # ---- control (mastership + motors + program) ----
    # OmniCore RWS 2.0 puts the action in the URL path for mastership endpoints,
    # but in the body for most others. Don't ask why.
    def request_mastership(self) -> None:
        self._post("/rw/mastership/edit/request")

    def release_mastership(self) -> None:
        self._post("/rw/mastership/edit/release")

    def set_motors_on(self) -> None:
        self._post("/rw/panel/ctrl-state", data={"ctrl-state": "motoron"})

    def reset_pp(self, task: str = "T_ROB1") -> None:
        self._post("/rw/rapid/execution/resetpp", data={"task": task})

    def unload_module(self, module_name: str, task: str = "T_ROB1") -> None:
        """Unload a RAPID module from a task. Silent if module isn't loaded."""
        try:
            self._post(f"/rw/rapid/tasks/{task}/unloadmod", data={"module": module_name})
        except requests.HTTPError:
            pass

    def start_program(self) -> None:
        self._post(
            "/rw/rapid/execution/start",
            data={
                "regain": "continue",
                "execmode": "continue",
                "cycle": "once",
                "condition": "none",
                "stopatbp": "disabled",
                "alltaskbytsp": "false",
            },
        )

    def stop_program(self) -> None:
        self._post("/rw/rapid/execution/stop", data={"stopmode": "stop"})

    # ---- RAPID data (OmniCore RWS 2.0 URL form requires module in path) ----
    def _symbol_data_url(self, var: str, task: str, module: str) -> str:
        return f"/rw/rapid/symbol/RAPID/{task}/{module}/{var}/data"

    def set_rapid_bool(
        self, var: str, value: bool, task: str = "T_ROB1", module: str = "PyExec"
    ) -> None:
        self._post(
            self._symbol_data_url(var, task, module),
            data={"value": "TRUE" if value else "FALSE"},
        )

    def get_rapid_data(self, var: str, task: str = "T_ROB1", module: str = "PyExec") -> str:
        data = self._get(self._symbol_data_url(var, task, module))
        return _find_first_value(data, key="value")


# ---- response-shape helpers ----
def _find_jointtarget(data: dict[str, Any]) -> dict[str, str]:
    """Pull the dict containing rax_1..rax_6 out of an RWS response, tolerant to shape."""
    if not isinstance(data, dict):
        raise ValueError(f"unexpected RWS response: {data!r}")
    if "state" in data and isinstance(data["state"], list) and data["state"]:
        s = data["state"][0]
        if "rax_1" in s:
            return s
    if "rax_1" in data:
        return data  # type: ignore[return-value]
    embedded = data.get("_embedded")
    if isinstance(embedded, dict):
        for v in embedded.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "rax_1" in v[0]:
                return v[0]
    raise ValueError(f"could not locate rax_1..rax_6 in RWS response: {data!r}")


def _find_first_value(data: dict[str, Any], key: str) -> str:
    if key in data:
        return str(data[key])
    if "state" in data and data["state"]:
        s = data["state"][0]
        if key in s:
            return str(s[key])
    embedded = data.get("_embedded")
    if isinstance(embedded, dict):
        for v in embedded.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and key in v[0]:
                return str(v[0][key])
    raise ValueError(f"could not find {key!r} in RWS response: {data!r}")
