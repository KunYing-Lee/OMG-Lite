"""Shared control ordering. Sync pauses simulation; async obeys the preview deadline."""

import time
from concurrent.futures import ThreadPoolExecutor

from omg.pipeline.schedule import FPS, PERIOD, REQUEST, PlanTimeline


def run(planner, tracker, history, text, *, frames=500, mode="sync"):
    if mode not in ("sync", "async") or frames <= 0:
        raise ValueError("Expected sync/async and positive frames")
    first = planner.plan(history, text, 0)
    timeline = PlanTimeline(first["qpos"])
    tracker.activate(timeline.qpos)
    plans = [first]
    future = None
    origin = time.perf_counter()

    def generate(request):
        result = planner.plan(request.history, text, request.plan_id)
        return result, time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as pool:
        for frame in range(frames):
            if mode == "async":
                time.sleep(max(0, origin + frame / FPS - time.perf_counter()))
                if time.perf_counter() > origin + (frame + 1) / FPS:
                    raise TimeoutError(f"Control tick missed at frame {frame}")
            if future is not None and future.done():
                request = timeline.pending
                result, ready = future.result()
                if ready > origin + request.deadline_frame / FPS:
                    raise TimeoutError("Inference missed wall-clock preview deadline")
                timeline.accept(request, result["qpos"], current_frame=frame)
                plans.append(result)
                tracker.activate(timeline.qpos)
                future = None
            timeline.check_frame(frame)
            if frame % PERIOD == REQUEST:
                request = timeline.request(frame)
                if mode == "sync":
                    result, _ = generate(request)
                    timeline.accept(request, result["qpos"], current_frame=frame)
                    plans.append(result)
                    tracker.activate(timeline.qpos)
                else:
                    future = pool.submit(generate, request)
            tracker.step(frame, frame // PERIOD)
    return timeline, plans
