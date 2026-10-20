from omg.pipeline.schedule import PREVIEW, motion
from omg.tracking.holomotion.reference import precompute_reference_features


class Tracker:
    def __init__(self, runner):
        self.runner = runner
        if runner.metadata.n_fut_frames != PREVIEW:
            raise ValueError("Expected tracker future preview of 10 frames")
        if abs(runner.model.opt.timestep * runner.control_substeps - 0.02) > 1e-9:
            raise ValueError("Expected 50Hz tracker")

    def activate(self, qpos):
        self.qpos = motion(qpos)
        self.features = precompute_reference_features(
            self.qpos, fps=50.0, onnx_to_g1=self.runner.holomotion_handles.onnx_to_g1
        )

    def step(self, frame, plan_id):
        return self.runner.run_reference_chunk(
            self.qpos,
            reference_fps=50.0,
            steps=1,
            reference_cursor=frame,
            reference_features=self.features,
            plan_id=plan_id,
        ).qpos_36[0]
