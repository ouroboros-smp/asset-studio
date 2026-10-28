"""Stage entry point the orchestrator runs for the Pixal3D stage (through the stage runner, one process per job).

Hands the job to the resident session (serve.py) when one is ready and matches the request; otherwise runs the
job in this process exactly as before (`generate.run_one_shot`). Either way this process's stdout is the stage
log, its exit code is the stage result and SIGTERM (runner cancel) cancels the job, so the orchestrator, OOM
fallback and re-attach logic do not know the difference.

A missing health response is the restart gap (serve exits, entrypoint.sh sleeps, then reloads), not proof that
no session exists. One-shotting in that gap loads a second copy of the models next to the session that is about
to come back. Connection refused while polling a job means the process is gone; the long grace is only for a
live process whose C++ step holds the GIL and cannot answer HTTP.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("STUDIO_PIXAL3D_RESIDENT_PORT", "8712"))
BASE = f"http://127.0.0.1:{PORT}"
LOAD_WAIT_S = int(os.environ.get("STUDIO_PIXAL3D_RESIDENT_WAIT_S", "300"))
# A refused connection is the process gone (OOM killer, segfault, restart). A timeout is a live process that
# has not released the GIL. Those must not share a grace period.
DEAD_GRACE_S = float(os.environ.get("STUDIO_PIXAL3D_RESIDENT_DEAD_S", "5"))
GIL_GRACE_S = float(os.environ.get("STUDIO_PIXAL3D_RESIDENT_POLL_GRACE_S", "900"))
PID_FILE = os.environ.get("STUDIO_PIXAL3D_RESIDENT_PID", "/tmp/pixal3d-resident.pid")
_TRANSIENT = ("loading", "busy", "restarting")
_DEAD_ERRNOS = {errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE, errno.ENOTCONN}


def _api(path: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def _health():
    try:
        return _api("/health")[1]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _own_log_path() -> str | None:
    """The runner redirects our stdout to the stage log; the session appends to that same file."""
    try:
        p = os.readlink("/proc/self/fd/1")
        return p if p.startswith("/") and os.path.isfile(p) else None
    except OSError:
        return None


def _refusal_error(detail: str):
    try:
        return json.loads(detail).get("error")
    except json.JSONDecodeError:
        return None


def acquire_resident(health, post, *, wait_s: float, sleep=time.sleep, now=time.time, poll_s: float = 2.0):
    """Wait until the resident session accepts the job.

    Returns ("ok", response, health), ("refused", detail, health), or ("down", None, last_health).
    Health that is down is waited out for the whole budget: the entrypoint's restart sleep is several seconds
    during which nothing is listening, and one-shotting then races the reload.
    """
    deadline = now() + wait_s
    last = None
    while True:
        h = health()
        if h is not None:
            last = h
            state = h.get("state")
            if state == "ready":
                try:
                    return "ok", post(), h
                except urllib.error.HTTPError as e:
                    try:
                        detail = e.read().decode("utf-8", "replace")[:300]
                    except Exception:  # noqa: BLE001
                        detail = str(e)[:300]
                    if _refusal_error(detail) not in _TRANSIENT or now() >= deadline:
                        return "refused", f"{e.code}: {detail}", h
                except (urllib.error.URLError, OSError):
                    if now() >= deadline:
                        return "down", None, last
            elif state not in _TRANSIENT:
                return "refused", str(state), h
        if now() >= deadline:
            return "down", None, last
        sleep(min(poll_s, deadline - now()))


def classify_poll_error(exc: BaseException) -> str:
    """'dead' if nothing is accepting connections; 'slow' if the process is alive but not answering."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError) else exc
    if isinstance(reason, TimeoutError):
        return "slow"
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in _DEAD_ERRNOS:
        return "dead"
    return "slow"


def poll_job(fetch, *, gil_grace_s: float, dead_grace_s: float, sleep=time.sleep, now=time.time, poll_s: float = 1.0):
    """Poll until the resident job finishes.

    Returns ("done"|"failed"|"lost"|"dead", record|None). Connection refused fails after dead_grace_s.
    Timeouts (GIL held by xatlas/remeshing) keep gil_grace_s.
    """
    last_ok = now()
    dead_since = None
    while True:
        sleep(poll_s)
        kind = None
        try:
            s = fetch()
        except urllib.error.HTTPError as e:
            if e.code == 404:  # the process restarted and no longer has this job
                return "lost", None
            kind = "slow"
            s = None
        except (urllib.error.URLError, OSError, ValueError) as e:
            kind = classify_poll_error(e)
            s = None
        if kind:
            if kind == "dead":
                if dead_since is None:
                    dead_since = now()
                if now() - dead_since >= dead_grace_s:
                    return "dead", None
            else:
                dead_since = None
                if now() - last_ok >= gil_grace_s:
                    return "lost", None
            continue
        last_ok = now()
        dead_since = None
        if not isinstance(s, dict):
            continue
        if s.get("state") == "done":
            return "done", s
        if s.get("state") == "failed":
            return "failed", s
        if s.get("error") == "unknown job":
            return "lost", s


def acknowledge_cancel(job_id: str, pid: int | None, api, kill) -> bool:
    """Ask the session to exit. If it does not answer, `kill` is the out-of-band stop.

    The runner's SIGTERM/SIGKILL hits this submit process, not the resident session (a sibling started by
    entrypoint.sh). A cancel that times out because a native step holds the GIL would otherwise leave that
    session running and holding the GPU.
    """
    try:
        api(f"/jobs/{job_id}/cancel", {})
        return True
    except Exception:  # noqa: BLE001
        if pid:
            kill(pid)
        return False


def read_resident_pid(health: dict | None = None) -> int | None:
    pid = (health or {}).get("pid") if isinstance(health, dict) else None
    if not isinstance(pid, int) or pid <= 1:
        try:
            pid = int(open(PID_FILE, encoding="utf-8").read().strip())
        except (OSError, ValueError):
            return None
    return pid if isinstance(pid, int) and pid > 1 else None


def process_is_resident(pid: int) -> bool:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return False
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return "pixal3d_worker.serve" in cmd


def kill_resident_pid(pid: int | None) -> bool:
    if not pid or pid <= 1 or not process_is_resident(pid):
        print(f"[resident] not killing pid {pid}: it is not the pixal3d session", flush=True)
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return False
    print(f"[resident] killed unresponsive resident session pid {pid}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    a = ap.parse_args()
    req = json.loads(open(a.request, encoding="utf-8").read())
    log_path = _own_log_path()

    def one_shot(reason: str) -> int:
        print(f"[resident] {reason}; running the stage in this process", flush=True)
        from pixal3d_worker.generate import run_one_shot
        run_one_shot(req)
        return 0

    if os.environ.get("STUDIO_PIXAL3D_RESIDENT", "1") == "0" or req.get("mode", "single") != "single" or not log_path:
        return one_shot("resident session disabled or not applicable")

    def _post():
        _status, body = _api("/jobs", {"request_path": a.request, "log_path": log_path})
        return body

    kind, payload, health = acquire_resident(_health, _post, wait_s=LOAD_WAIT_S)
    if kind != "ok":
        if kind == "refused":
            return one_shot(f"resident session refused the job ({payload})")
        state = (health or {}).get("state") if isinstance(health, dict) else None
        if not state:
            return one_shot(f"no resident session after {LOAD_WAIT_S}s")
        return one_shot(f"resident session is {state} after {LOAD_WAIT_S}s")
    job_id = payload["job_id"]
    resident_pid = read_resident_pid(health)

    def on_term(signum, frame):
        try:
            acknowledge_cancel(job_id, resident_pid, lambda path, body=None: _api(path, body, timeout=5), kill_resident_pid)
        finally:
            sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def _fetch():
        _status, rec = _api(f"/jobs/{job_id}", timeout=20)
        return rec

    outcome, rec = poll_job(_fetch, gil_grace_s=GIL_GRACE_S, dead_grace_s=DEAD_GRACE_S)
    if outcome == "done":
        return 0
    if outcome == "failed":
        print(f"[resident] job failed: {(rec or {}).get('error')}", flush=True)
        return 1
    if outcome == "dead":
        print("[resident] resident session is gone", flush=True)
        return 1
    print("[resident] lost the resident session mid-job", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
