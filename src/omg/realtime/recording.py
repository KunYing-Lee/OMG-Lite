"""Per-run recorder for the realtime chain (module-level, no ROS dependency).

Records the four module outputs plus the transforms' parameters and the event
timeline, so every other quantity (timeline, tape, obs, target) is derivable:

    M1 main model   : plan block 100x36, one per accepted plan     -> <base>.plans.npz
    M2 corrector    : 25x29 joint delta, one per decision frame    -> <base>.decisions.npz
    M3 tracker      : action 29 per frame                          -> <base>.frames.csv
    M4 robot        : q/dq/tau/quat/gyro per frame                 -> <base>.frames.csv
    events          : request/accept frames, wall times, latency   -> <base>.plans.npz
    transforms      : switch_seconds, action_scale, default_angles,
                      onnx_to_real, residual manifest, tracker meta -> <base>.meta.json

Frame timings are stored next to the frame row (read/tick/encode/consume/step).
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

RECORD_VERSION = 2
FRAME_COLUMNS = (
    ["frame", "t"]
    + ["action%d" % i for i in range(29)]
    + ["q%d" % i for i in range(29)]
    + ["dq%d" % i for i in range(29)]
    + ["tau%d" % i for i in range(29)]
    + ["quat%d" % i for i in range(4)]
    + ["gyro%d" % i for i in range(3)]
    + ["residual"]
    + ["read_ms", "tick_ms", "encode_ms", "consume_ms", "step_ms"]
)


class Recorder:
    """Writes one directory-less file family next to `path`:

    <path>.frames.csv, <path>.plans.npz, <path>.decisions.npz, <path>.meta.json
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.base = Path(path)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.frames_path = Path(str(self.base) + ".frames.csv")
        self.plans_path = Path(str(self.base) + ".plans.npz")
        self.decisions_path = Path(str(self.base) + ".decisions.npz")
        self.debug_path = Path(str(self.base) + ".debug.npz")
        self.meta_path = Path(str(self.base) + ".meta.json")

        self.meta = dict(meta or {})
        self.meta.setdefault("record_version", RECORD_VERSION)
        self.meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.meta.setdefault("frame_columns", FRAME_COLUMNS)
        self._write_meta()

        self._frame_file = open(self.frames_path, "w", encoding="utf-8", newline="")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow(FRAME_COLUMNS)
        self._frames = 0

        self._plans: list[dict] = []
        self._decisions: list[dict] = []
        # Per-frame diagnostic payloads not needed by the compact CSV: raw
        # vs filtered dq, raw vs mapped tracker action, target and limits,
        # per-frame raw/corrected reference, gravity.
        self._debug = {
            key: [] for key in (
                "frame", "t", "action_raw", "action", "dq_raw", "dq",
                "target", "final_target", "action_clip", "target_limit",
                "ref_plan", "ref_corrected", "gravity",
            )
        }
        self._debug_frames = 0
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._written_plans = 0
        self._written_decisions = 0
        self._writer_error = None

        # The control loop only appends + sets an event.  Compression and the
        # O(n) rewrite of plans/decisions happen on a writer thread, so a plan
        # accept cannot consume the 20 ms tracker budget.
        self._write_plans([])
        self._write_decisions([])
        self._writer = threading.Thread(
            target=self._writer_loop, name="omg-recorder-writer", daemon=True
        )
        self._writer.start()

    # -- metadata ---------------------------------------------------------
    def update_meta(self, **fields) -> None:
        self.meta.update(fields)
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self.meta, indent=1, default=str), encoding="utf-8")

    # -- M1 events --------------------------------------------------------
    def plan(self, *, plan_id, request_frame, accept_frame, switch_frame, deadline_frame,
             block, request_time=None, accept_time=None) -> None:
        block = np.asarray(block, dtype=np.float32)
        latency = None
        if request_time is not None and accept_time is not None:
            latency = float((accept_time - request_time) * 1000.0)
        with self._lock:
            self._plans.append(dict(
                plan_id=int(plan_id), request_frame=int(request_frame),
                accept_frame=int(accept_frame), switch_frame=int(switch_frame),
                deadline_frame=int(deadline_frame), request_time=request_time,
                accept_time=accept_time, latency_ms=latency, block=block,
            ))
        self._dirty.set()

    def _write_plans(self, plans: list[dict]) -> None:
        if not plans:
            arrays = dict(
                plan_id=np.zeros(0, np.int64), request_frame=np.zeros(0, np.int64),
                accept_frame=np.zeros(0, np.int64), switch_frame=np.zeros(0, np.int64),
                deadline_frame=np.zeros(0, np.int64), request_time=np.zeros(0, np.float64),
                accept_time=np.zeros(0, np.float64), latency_ms=np.zeros(0, np.float64),
                blocks=np.zeros((0, 0, 36), np.float32),
            )
        else:
            arrays = dict(
                plan_id=np.array([p["plan_id"] for p in plans], np.int64),
                request_frame=np.array([p["request_frame"] for p in plans], np.int64),
                accept_frame=np.array([p["accept_frame"] for p in plans], np.int64),
                switch_frame=np.array([p["switch_frame"] for p in plans], np.int64),
                deadline_frame=np.array([p["deadline_frame"] for p in plans], np.int64),
                request_time=np.array([p["request_time"] if p["request_time"] is not None else np.nan
                                       for p in plans], np.float64),
                accept_time=np.array([p["accept_time"] if p["accept_time"] is not None else np.nan
                                      for p in plans], np.float64),
                latency_ms=np.array([p["latency_ms"] if p["latency_ms"] is not None else np.nan
                                     for p in plans], np.float64),
                blocks=np.stack([p["block"] for p in plans]),
            )
        self._atomic_savez(self.plans_path, arrays)

    def _atomic_savez(self, path: Path, arrays: dict) -> None:
        """Write to a sibling temp file, then replace; readers never see a partial npz."""
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)

    # -- M2 events --------------------------------------------------------
    def decision(self, *, frame, start, delta, latent_norm=None,
                 obs=None, latent=None, base=None, corrected=None) -> None:
        delta = np.asarray(delta, dtype=np.float32)
        with self._lock:
            self._decisions.append(dict(
                frame=int(frame), start=int(start), delta=delta,
                latent_norm=None if latent_norm is None else float(latent_norm),
                obs=None if obs is None else np.asarray(obs, np.float32).copy(),
                latent=None if latent is None else np.asarray(latent, np.float32).copy(),
                base=None if base is None else np.asarray(base, np.float32).copy(),
                corrected=None if corrected is None else np.asarray(corrected, np.float32).copy(),
            ))
        self._dirty.set()

    def _write_decisions(self, decisions: list[dict]) -> None:
        if not decisions:
            arrays = dict(
                frame=np.zeros(0, np.int64), start=np.zeros(0, np.int64),
                latent_norm=np.zeros(0, np.float64), delta=np.zeros((0, 0, 29), np.float32),
                obs=np.zeros((0, 0), np.float32),
                latent=np.zeros((0, 0), np.float32),
                base=np.zeros((0, 0, 36), np.float32),
                corrected=np.zeros((0, 0, 36), np.float32),
            )
        else:
            arrays = dict(
                frame=np.array([d["frame"] for d in decisions], np.int64),
                start=np.array([d["start"] for d in decisions], np.int64),
                latent_norm=np.array([np.nan if d["latent_norm"] is None else d["latent_norm"]
                                      for d in decisions], np.float64),
                delta=np.stack([d["delta"] for d in decisions]),
            )
            for key in ("obs", "latent", "base", "corrected"):
                values = [d.get(key) for d in decisions]
                if all(value is not None for value in values):
                    arrays[key] = np.stack(values)
                elif any(value is not None for value in values):
                    # Mixed old/new decisions; keep an object array rather than
                    # silently dropping the diagnostic payloads.
                    arrays[key] = np.asarray(values, dtype=object)
        self._atomic_savez(self.decisions_path, arrays)

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            self._dirty.wait(0.1)
            if self._stop.is_set():
                break
            self._dirty.clear()
            with self._lock:
                plans = list(self._plans)
                decisions = list(self._decisions)
            try:
                if len(plans) != self._written_plans:
                    self._write_plans(plans)
                    self._written_plans = len(plans)
                if len(decisions) != self._written_decisions:
                    self._write_decisions(decisions)
                    self._written_decisions = len(decisions)
            except Exception as error:  # noqa: BLE001 - recording must not kill control
                self._writer_error = error

    # -- per-frame --------------------------------------------------------
    def frame(self, *, frame, t, action, q, dq, tau, quat, gyro, residual,
              read_ms=0.0, tick_ms=0.0, encode_ms=0.0, consume_ms=0.0, step_ms=0.0,
              action_raw=None, dq_raw=None, target=None, final_target=None,
              action_clip=None, target_limit=None,
              ref_plan=None, ref_corrected=None, gravity=None) -> None:
        row = [int(frame), float(t)]
        row += np.asarray(action, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(q, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(dq, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(tau, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(quat, np.float32).reshape(-1)[:4].tolist()
        row += np.asarray(gyro, np.float32).reshape(-1)[:3].tolist()
        row += [int(bool(residual)), float(read_ms), float(tick_ms), float(encode_ms),
                float(consume_ms), float(step_ms)]
        if len(row) != len(FRAME_COLUMNS):
            raise ValueError("frame row has %d fields, expected %d" % (len(row), len(FRAME_COLUMNS)))
        self._frame_writer.writerow(row)
        self._frame_file.flush()
        self._frames += 1

        # Optional diagnostic payload.  Node deployments that pass the new
        # arguments get <base>.debug.npz; older callers keep the old files.
        if action_raw is not None:
            values = {
                "frame": int(frame),
                "t": float(t),
                "action_raw": np.asarray(action_raw, np.float32).reshape(29),
                "action": np.asarray(action, np.float32).reshape(29),
                "dq_raw": np.asarray(dq_raw if dq_raw is not None else dq, np.float32).reshape(29),
                "dq": np.asarray(dq, np.float32).reshape(29),
                "target": np.asarray(target if target is not None else action, np.float32).reshape(29),
                "final_target": np.asarray(final_target if final_target is not None else target if target is not None else action, np.float32).reshape(29),
                "action_clip": np.asarray(action_clip if action_clip is not None else np.zeros(29), bool).reshape(29),
                "target_limit": np.asarray(target_limit if target_limit is not None else np.zeros(29), bool).reshape(29),
                "ref_plan": np.asarray(ref_plan if ref_plan is not None else np.zeros(36), np.float32).reshape(36),
                "ref_corrected": np.asarray(ref_corrected if ref_corrected is not None else ref_plan if ref_plan is not None else np.zeros(36), np.float32).reshape(36),
                "gravity": np.asarray(gravity if gravity is not None else np.zeros(3), np.float32).reshape(3),
            }
            for key, value in values.items():
                self._debug[key].append(value)
            self._debug_frames += 1

    def _write_debug(self) -> None:
        if self._debug_frames == 0:
            return
        arrays = {}
        for key, values in self._debug.items():
            arrays[key] = np.stack(values)
        self._atomic_savez(self.debug_path, arrays)

    def flush_debug(self) -> None:
        """Persist per-frame diagnostics on a stop/fault, not only on close."""
        with self._lock:
            self._write_debug()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._dirty.set()
        self._writer.join()
        with self._lock:
            plans = list(self._plans)
            decisions = list(self._decisions)
        try:
            self._write_plans(plans)
            self._write_decisions(decisions)
            self._write_debug()
        except Exception as error:  # noqa: BLE001 - keep the shutdown path alive
            self._writer_error = error
        finally:
            self._frame_file.close()
            self.update_meta(
                frames=self._frames,
                plans=len(plans),
                decisions=len(decisions),
                debug_frames=self._debug_frames,
                writer_error=None if self._writer_error is None else repr(self._writer_error),
            )
