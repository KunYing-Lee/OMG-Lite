"""Request/response adapter for the same native timeline used by simulation."""

from concurrent.futures import ThreadPoolExecutor

from omg.pipeline.schedule import Schedule, motion
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest


class PlannerService:
    def __init__(self, planner, schedule=None):
        schedule = Schedule() if schedule is None else schedule
        self.schedule = schedule
        self.planner = planner

    def plan(self, request):
        if request.history_fps != 50 or not request.prompt:
            raise ValueError("Native 50Hz history and text required")
        if request.metadata.get("period_frames") != self.schedule.period:
            raise ValueError("Planner and client schedules must match")
        history = motion(request.qpos_36_history, 10)
        plan_id = int(request.metadata["plan_id"])
        switch = int(request.metadata["switch_frame"])
        if plan_id == 0:
            if switch != 0 or request.tracker_frame != 0:
                raise ValueError("Invalid initialization")
        elif (
            switch != plan_id * self.schedule.period
            or request.tracker_frame
            != switch - self.schedule.period + self.schedule.request_phase
        ):
            raise ValueError("Invalid future-history schedule")
        result = self.planner.plan(history, request.prompt, plan_id)
        return MotionPlanChunk(
            qpos_36=result["qpos"],
            fps=50,
            request_id=request.request_id,
            plan_id=plan_id,
            request_tracker_frame=request.tracker_frame,
            prompt=request.prompt,
            planning_latency_seconds=result["latency_s"],
            metadata={"switch_frame": switch, "period_frames": self.schedule.period},
        )


class RemotePlanner:
    """Persistent connection owned exclusively by a dedicated I/O thread."""

    def __init__(self, connect, schedule=None):
        schedule = Schedule() if schedule is None else schedule
        self.schedule = schedule
        self.connect = connect
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.client = None

    def plan(self, history, text, plan_id):
        return self.pool.submit(self._plan, history, text, plan_id).result()

    def _plan(self, history, text, plan_id):
        from omg.realtime.transport import ZmqPlanClient

        switch = plan_id * self.schedule.period
        request = RobotStateRequest(
            qpos_36_history=motion(history, 10),
            history_fps=50,
            tracker_frame=(
                0
                if plan_id == 0
                else switch - self.schedule.period + self.schedule.request_phase
            ),
            prompt=text,
            metadata={
                "plan_id": plan_id,
                "switch_frame": switch,
                "period_frames": self.schedule.period,
            },
        )
        if self.client is None:
            self.client = ZmqPlanClient(self.connect)
        response = self.client.request_plan(
            request, timeout_ms=30000 if plan_id == 0 else self.schedule.budget_ms
        )
        if (
            response.fps,
            response.plan_id,
            response.request_tracker_frame,
            response.metadata.get("switch_frame"),
            response.metadata.get("period_frames"),
        ) != (50, plan_id, request.tracker_frame, switch, self.schedule.period):
            raise ValueError("Mismatched planner response")
        return {
            "qpos": motion(response.qpos_36, 100),
            "latency_s": response.planning_latency_seconds,
        }

    def _close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def close(self):
        self.pool.submit(self._close).result()
        self.pool.shutdown(wait=True, cancel_futures=True)
