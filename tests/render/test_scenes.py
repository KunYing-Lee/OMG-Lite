import json

import mujoco
import numpy as np
import pytest

from omg.render import mujoco as render
from omg.tracking.holomotion.runtime import build_g1_state_handles, set_g1_qpos

URDF = 'assets/robots/g1/g1_29dof.urdf'


@pytest.mark.parametrize('scene', ['studio', 'minimal', 'unknown'])
def test_removed_scenes_rejected(scene):
    with pytest.raises(ValueError, match='Unsupported'):
        render.load_scene_model(scene, URDF, 320, 240)
    with pytest.raises(ValueError, match='Unsupported'):
        render.build_mjcf({}, 320, 240, scene_preset=scene)


def test_native_replay_uses_tracker_joint_mapping():
    model = render.load_scene_model('tracker_native', URDF, 320, 240)
    kin = render.G1Kinematics('assets/robots/g1/g1_kinematics.json')
    qpos = np.r_[0.1, -0.2, 0.8, 1., 0., 0., 0., np.linspace(-0.2, 0.2, 29)]
    viewer, tracker = mujoco.MjData(model), mujoco.MjData(model)
    render._set_data_qpos(viewer, qpos, render._joint_qpos_addresses(model, kin.joint_order), kin.joint_name_to_qpos_index)
    set_g1_qpos(model, tracker, build_g1_state_handles(model), qpos)
    np.testing.assert_array_equal(viewer.qpos, tracker.qpos)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'groundplane') >= 0


def test_mixed_scene_comparison_preserves_motion_and_metadata(tmp_path, monkeypatch):
    observed = []
    def frames(qpos, body_pos, kin, model, data, **kwargs):
        observed.append((qpos.copy(), model))
        return np.zeros((len(qpos), 16, 16, 3), np.uint8)
    monkeypatch.setattr(render, 'render_qpos_frames', frames)
    monkeypatch.setattr(render, 'save_video', lambda frames, path, fps: path)
    qpos = np.zeros((3, 36), np.float32)
    qpos[:, 2:4] = [0.8, 1.]
    executed = qpos.copy()
    executed[:, 0] = 0.1
    output = tmp_path / 'comparison.mp4'
    render.render_qpos_comparison_video(qpos, executed, output, right_scene_preset='tracker_native')
    np.testing.assert_array_equal(observed[0][0], qpos)
    np.testing.assert_array_equal(observed[1][0], executed)
    assert mujoco.mj_name2id(observed[0][1], mujoco.mjtObj.mjOBJ_MATERIAL, 'grid_minor_mat') >= 0
    assert mujoco.mj_name2id(observed[1][1], mujoco.mjtObj.mjOBJ_MATERIAL, 'groundplane') >= 0
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['scene_preset'] == 'holomotion'
    assert metadata['right_scene_preset'] == 'tracker_native'
