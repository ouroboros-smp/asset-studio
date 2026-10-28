"""Orchestrator worker: two lanes (image jobs, asset jobs) so a quick image session never waits behind a 3D job.
Each lane runs one job at a time; GPU stages start without VRAM gating and retry after the other lane finishes on OOM.
Durable state in SQLite; on restart, jobs are requeued and stages still running in the runner containers are re-attached."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback

from . import config
from .db import JobStore
from .pipeline import JobCancelled, JobFailed, JobRun
from .runner_client import Runner

LANE_KINDS = {"image": "image", "asset": "asset"}


def _recover(store: JobStore, worker_id: str):
    requeued, cancelled = store.requeue_orphans(worker_id)
    if not requeued and not cancelled:
        return
    if requeued:
        print(f"[worker] requeued {len(requeued)} interrupted job(s): {requeued}", flush=True)
    if cancelled:
        print(f"[worker] finished {len(cancelled)} in-flight cancel(s) at restart: {cancelled}", flush=True)
    # Stage subprocesses started by the previous worker may still be running. Keep the ones a requeued job will
    # re-attach to (their run.json names the run id); cancel anything else that is busy, including a stage whose
    # job was only marked cancelled (that list can be non-empty while `requeued` is empty).
    keep = set()
    for jid in requeued:
        for rf in (config.JOBS_DIR / jid / "stages").glob("*/run.json"):
            try:
                keep.add(json.loads(rf.read_text(encoding="utf-8"))["run_id"])
            except Exception:  # noqa: BLE001
                pass
    for name in ("image", "pixal3d", "blender"):
        rn = Runner(name)
        try:
            h = rn.health()
        except Exception:  # noqa: BLE001
            continue
        if not h.get("busy"):
            continue
        rid = h.get("run_id")
        if rid and rid in keep:
            print(f"[worker] {name} runner still working on run {rid}; the requeued job will re-attach", flush=True)
            continue
        r = rn.cancel_current()
        if r.get("cancelled"):
            print(f"[worker] cancelled orphaned {name} run {r['cancelled']}", flush=True)


def _lane(lane: str, worker_id: str):
    store = JobStore()  # one connection per lane thread
    kind = LANE_KINDS[lane]
    while True:
        try:
            job = store.claim_next(worker_id, kind=kind)
        except Exception as e:  # noqa: BLE001
            print(f"[worker:{lane}] claim failed: {e}", flush=True)
            time.sleep(2)
            continue
        if not job:
            time.sleep(1.0)
            continue
        if job["cancel_requested"]:
            store.update(job["id"], status="cancelled", stage="cancelled", finished_at=time.time())
            continue
        print(f"[worker:{lane}] running job {job['id']} (attempt {job['attempt']})", flush=True)
        try:
            run = JobRun(store, job)
            run.run()
        except JobCancelled:
            store.update(job["id"], status="cancelled", stage="cancelled", finished_at=time.time())
            store.event(job["id"], "warn", "job cancelled")
            print(f"[worker:{lane}] job {job['id']} cancelled", flush=True)
        except JobFailed as e:
            err = {"stage": e.stage, "message": str(e), "details": e.details}
            store.update(job["id"], status="failed", stage=e.stage, finished_at=time.time(), error=err)
            store.event(job["id"], "error", f"stage {e.stage} failed: {e}")
            print(f"[worker:{lane}] job {job['id']} failed in {e.stage}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            err = {"stage": "worker", "message": f"{type(e).__name__}: {e}", "details": {"traceback": tb[-4000:]}}
            store.update(job["id"], status="failed", finished_at=time.time(), error=err)
            store.event(job["id"], "error", f"worker exception: {type(e).__name__}: {e}")
            print(f"[worker:{lane}] job {job['id']} crashed: {tb}", flush=True)


def main():
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    _recover(JobStore(), worker_id)
    print(f"[worker] {worker_id} ready; lanes=image,asset jobs={config.JOBS_DIR} db={config.DB_PATH}", flush=True)
    threads = [threading.Thread(target=_lane, args=(lane, worker_id), name=f"lane-{lane}", daemon=True) for lane in LANE_KINDS]
    for t in threads:
        t.start()
    while True:
        time.sleep(30)
        for t in threads:
            if not t.is_alive():
                raise SystemExit(f"lane thread {t.name} died")


if __name__ == "__main__":
    main()
