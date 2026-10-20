"""Robot-side integration. The existing robot controller owns arming and motors.

Call activate with measured native 50Hz startup history after arming. Call tick
once at each 50Hz control boundary. Any raised error must stop active tracking;
this adapter never holds, extrapolates, skips, or silently changes the schedule.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from omg.pipeline.schedule import PlanTimeline, motion


class DeploymentBridge:
    def __init__(self, planner, text, *, clock=time.monotonic):
        self.planner = planner
        self.text = text
        self.clock = clock
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.timeline = None
        self.future = None
        self.frame = 0

    def activate(self, history):
        if self.timeline is not None:
            raise RuntimeError("Already active")
        # Startup is outside the active control clock; existing controller stays armed but idle.
        first = self.planner.plan(motion(history, 10), self.text, 0)
        self.timeline = PlanTimeline(first["qpos"])
        self.origin = self.clock()

    def _generate(self, request):
        result = self.planner.plan(request.history, self.text, request.plan_id)
        return result, self.clock()

    def tick(self):
        if self.timeline is None:
            raise RuntimeError("Bridge is not activated")
        frame = self.frame
        now = self.clock()
        if now < self.origin + frame / 50 or now >= self.origin + (frame + 1) / 50:
            raise TimeoutError("Control tick outside its 20ms interval")
        if self.future is not None and self.future.done():
            request = self.timeline.pending
            result, ready = self.future.result()
            if ready > self.origin + request.deadline_frame / 50:
                raise TimeoutError("Planner missed preview deadline")
            self.timeline.accept(request, result["qpos"], current_frame=frame)
            self.future = None
        self.timeline.check_frame(frame)
        if frame % 50 == 25:
            request = self.timeline.request(frame)
            self.future = self.pool.submit(self._generate, request)
        self.frame += 1
        return self.timeline.qpos, frame

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        if hasattr(self.planner, "close"):
            self.planner.close()
