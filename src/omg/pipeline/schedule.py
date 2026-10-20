"""One native 50Hz raw-plan timeline for simulation and deployment."""

import math
from dataclasses import dataclass

import numpy as np

FPS = 50
HORIZON = 100
HISTORY = 10
LEAD = 25
PREVIEW = 10


@dataclass(frozen=True)
class Schedule:
    switch_seconds: float = 1.0

    def __post_init__(self):
        frames = self.switch_seconds * FPS
        if (
            not math.isfinite(frames)
            or not LEAD <= frames <= HORIZON
            or not math.isclose(frames, round(frames), abs_tol=1e-9, rel_tol=0)
        ):
            raise ValueError("switch_seconds must be 0.5..2.0 in 0.02s increments")

    @property
    def period(self):
        return round(self.switch_seconds * FPS)

    @property
    def request_phase(self):
        return self.period - LEAD

    @property
    def budget_ms(self):
        return (LEAD - PREVIEW - 1) * 1000 // FPS

    def requests_at(self, frame):
        return frame % self.period == self.request_phase


def motion(value, frames=None):
    q = np.asarray(value, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 36 or (frames is not None and len(q) != frames):
        raise ValueError(f"Expected ({frames or 'T'},36) qpos, got {q.shape}")
    if not len(q) or not np.isfinite(q).all():
        raise ValueError("Motion must be nonempty and finite")
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-4):
        raise ValueError("Root quaternions must be normalized")
    return q.copy()


@dataclass(frozen=True)
class Replan:
    plan_id: int
    request_frame: int
    switch_frame: int
    history: np.ndarray

    @property
    def deadline_frame(self):
        # Preview reaches +10; velocity differences additionally read +1.
        return self.switch_frame - PREVIEW - 1


class PlanTimeline:
    def __init__(self, first, schedule=Schedule()):
        self.schedule = schedule
        self.qpos = motion(first, HORIZON)
        self.raw = [self.qpos.copy()]
        self.pending = None
        self.events = []

    def request(self, frame):
        if not self.schedule.requests_at(frame) or self.pending is not None:
            raise ValueError(
                "Replan must start at the configured request phase with no pending request"
            )
        plan_id = frame // self.schedule.period + 1
        if plan_id != len(self.raw):
            raise ValueError("Nonsequential replan")
        switch = plan_id * self.schedule.period
        history = self.raw[-1][
            self.schedule.period - HISTORY : self.schedule.period
        ].copy()
        np.testing.assert_array_equal(history, self.qpos[switch - HISTORY : switch])
        self.pending = Replan(plan_id, frame, switch, history)
        return self.pending

    def accept(self, request, qpos, *, current_frame):
        if request is not self.pending:
            raise ValueError("Stale or unsolicited plan response")
        if not request.request_frame <= current_frame <= request.deadline_frame:
            raise TimeoutError("Plan arrived after the tracker preview deadline")
        q = motion(qpos, HORIZON)
        self.qpos = np.concatenate([self.qpos[: request.switch_frame], q])
        self.raw.append(q)
        self.events.append(
            {
                "plan_id": request.plan_id,
                "request_frame": request.request_frame,
                "switch_frame": request.switch_frame,
                "accepted_frame": current_frame,
                "preview_deadline_frame": request.deadline_frame,
            }
        )
        self.pending = None

    def check_frame(self, frame):
        if self.pending is not None and frame >= self.pending.deadline_frame:
            raise TimeoutError(f"Missing plan before preview at frame {frame}")
        if not 0 <= frame < len(self.qpos) - PREVIEW - 1:
            raise RuntimeError("Reference does not cover tracker preview")
