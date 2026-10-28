"""Resident-session handoff policy (no GPU, no models). The entrypoint restart gap and a dead session must not
share the one-shot fallback or the GIL poll grace."""
import os
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))

from pixal3d_worker import submit  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def test_restart_gap_waits_instead_of_one_shot():
    clock = Clock()
    script = [None, None, {"state": "restarting"}, None, {"state": "loading"}, {"state": "ready", "pid": 42}]
    seen = []

    def health():
        h = script.pop(0) if script else {"state": "ready", "pid": 42}
        seen.append(h)
        return h

    posts = []

    def post():
        posts.append(seen[-1])
        return {"job_id": "j1"}

    kind, resp, health_doc = submit.acquire_resident(
        health, post, wait_s=300, sleep=clock.sleep, now=clock.now, poll_s=2)
    assert kind == "ok" and resp["job_id"] == "j1"
    assert posts == [{"state": "ready", "pid": 42}]
    assert clock.t < 300
    assert None in seen  # the down window was waited out, not treated as "no session"


def test_absent_session_gives_up_only_after_the_budget():
    clock = Clock()
    calls = {"n": 0}

    def health():
        calls["n"] += 1
        return None

    def post():
        raise AssertionError("must not hand off while nothing is listening")

    kind, resp, health_doc = submit.acquire_resident(
        health, post, wait_s=6, sleep=clock.sleep, now=clock.now, poll_s=2)
    assert kind == "down" and health_doc is None and resp is None
    assert calls["n"] >= 3
    assert clock.t >= 6


def test_refused_post_during_restart_does_not_give_up_immediately():
    clock = Clock()
    step = {"n": 0}

    def health():
        step["n"] += 1
        if step["n"] == 1:
            return {"state": "ready", "pid": 7}
        if step["n"] < 4:
            return None
        return {"state": "ready", "pid": 8}

    posts = {"n": 0}

    def post():
        posts["n"] += 1
        if posts["n"] == 1:
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        return {"job_id": "j2"}

    kind, resp, _h = submit.acquire_resident(health, post, wait_s=30, sleep=clock.sleep, now=clock.now, poll_s=2)
    assert kind == "ok" and resp["job_id"] == "j2" and posts["n"] == 2
    assert clock.t < 30


def test_dead_session_fails_without_the_gil_grace():
    clock = Clock()

    def fetch():
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    outcome, rec = submit.poll_job(
        fetch, gil_grace_s=900, dead_grace_s=5, sleep=clock.sleep, now=clock.now)
    assert outcome == "dead" and rec is None
    assert clock.t < 30


def test_gil_timeout_keeps_the_long_grace():
    clock = Clock()

    def fetch():
        if clock.t < 30:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return {"state": "done"}

    outcome, rec = submit.poll_job(
        fetch, gil_grace_s=900, dead_grace_s=5, sleep=clock.sleep, now=clock.now)
    assert outcome == "done" and rec["state"] == "done"
    assert clock.t >= 30


def test_cancel_kills_session_only_when_unacknowledged():
    killed = []

    def timeout(path, body=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    assert submit.acknowledge_cancel("job", 4321, timeout, killed.append) is False
    assert killed == [4321]

    killed.clear()

    def ok(path, body=None):
        assert path == "/jobs/job/cancel"
        return 200, {"ok": True}

    assert submit.acknowledge_cancel("job", 4321, ok, killed.append) is True
    assert killed == []


def test_kill_resident_refuses_unrelated_pid():
    assert submit.process_is_resident(os.getpid()) is False
    assert submit.kill_resident_pid(os.getpid()) is False
