"""Request/response adapter for the same native timeline used by simulation."""

from omg.pipeline.schedule import motion
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest


class PlannerService:
    def __init__(self, planner):
        self.planner = planner

    def plan(self, request):
        if request.history_fps != 50 or not request.prompt:
            raise ValueError("Native 50Hz history and text required")
        history = motion(request.qpos_36_history, 10)
        plan_id = int(request.metadata["plan_id"])
        switch = int(request.metadata["switch_frame"])
        if plan_id == 0:
            if switch != 0 or request.tracker_frame != 0:
                raise ValueError("Invalid initialization")
        elif switch != plan_id * 50 or request.tracker_frame != switch - 25:
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
            metadata={"switch_frame": switch},
        )


class RemotePlanner:
    """Synchronous callable; the async executor calls this on its worker thread.

    ZMQ sockets are created in the calling thread, never shared across threads.
    """

    def __init__(self, connect):
        self.connect = connect

    def plan(self, history, text, plan_id):
        from omg.realtime.transport import ZmqPlanClient

        switch = plan_id * 50
        request = RobotStateRequest(
            qpos_36_history=motion(history, 10),
            history_fps=50,
            tracker_frame=0 if plan_id == 0 else switch - 25,
            prompt=text,
            metadata={"plan_id": plan_id, "switch_frame": switch},
        )
        with ZmqPlanClient(self.connect) as client:
            response = client.request_plan(
                request, timeout_ms=30000 if plan_id == 0 else 280
            )
        if (
            response.fps,
            response.plan_id,
            response.request_tracker_frame,
            response.metadata.get("switch_frame"),
        ) != (50, plan_id, request.tracker_frame, switch):
            raise ValueError("Mismatched planner response")
        return {
            "qpos": motion(response.qpos_36, 100),
            "latency_s": response.planning_latency_seconds,
        }
