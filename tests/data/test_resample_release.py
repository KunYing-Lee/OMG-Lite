import numpy as np
import pytest

from omg.cli.data.resample_release import coordinates, interpolate, resample_conditions


def test_grid_and_slerp_do_not_cross_episodes():
    lengths = np.array([4, 1, 3])
    q = np.zeros((8, 36), np.float32)
    q[:, 3] = 1
    q[:4, 0] = np.arange(4)
    q[4:, 0] = 100
    q[1::2, 3] = -1  # equivalent rotations, alternating quaternion sign
    out, _, _ = interpolate(q, lengths)
    lens, _, starts, frame, _, _, _ = coordinates(lengths)
    assert lens.tolist() == [6, 1, 4]
    np.testing.assert_allclose(out[:6, 0], np.arange(6) * 0.6)
    np.testing.assert_equal(out[starts[1] :, 0], 100)
    np.testing.assert_allclose(np.abs(out[:, 3]), 1)
    assert frame[starts[1]] == 0


def test_events_are_retimed_and_continuous_channels_interpolated():
    x = np.zeros((4, 35), np.float32)
    x[:, 0] = np.arange(4)
    x[1, 33] = x[3, 34] = 1
    y = resample_conditions(x, np.array([4]), ["aistpp"])
    np.testing.assert_allclose(y[:, 0], np.arange(6) * 0.6)
    assert np.flatnonzero(y[:, 33]).tolist() == [2]
    assert np.flatnonzero(y[:, 34]).tolist() == [5]
    assert set(np.unique(y[:, 33:])) == {0, 1}
    continuous = resample_conditions(x, np.array([4]), ["aioz_gdance"])
    assert continuous[1, 33] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="Undeclared"):
        resample_conditions(x, np.array([4]), ["unknown"])


def test_terminal_event_projects_to_nearest_in_domain_timestamp():
    x = np.zeros((3, 35), np.float32)
    x[-1, 34] = 1
    y = resample_conditions(x, np.array([3]), ["finedance"])
    assert np.flatnonzero(y[:, 34]).tolist() == [3]
    x[-1, 34] = 0.5
    with pytest.raises(ValueError, match="Non-binary"):
        resample_conditions(x, np.array([3]), ["finedance"])


def test_public_release_cli_round_trip(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omg.cli.data import resample_release, validate_resampled_release

    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    q = np.zeros((5, 36), np.float32)
    q[:, 0] = np.arange(5)
    q[:, 3] = 1

    def vectors(a):
        return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), a.shape[1])

    table = pa.table(
        {
            "observation.state": vectors(q),
            "action": vectors(q[[1, 1, 3, 4, 4]]),
            "omg.condition.has_text": [True] * 5,
            "timestamp": np.array([0, 1, 0, 1, 2], np.float32) / 30,
            "frame_index": [0, 1, 0, 1, 2],
            "episode_index": [0, 0, 1, 1, 1],
            "index": np.arange(5),
            "task_index": [0] * 5,
            "omg.audio.feature": vectors(np.zeros((5, 35), np.float32)),
            "omg.humanref.motion": vectors(np.ones((5, 66), np.float32)),
            "omg.condition.has_audio": [False] * 5,
            "omg.condition.has_humanref": [True] * 5,
        }
    )
    pq.write_table(table, source / "data/chunk-000/file-000.parquet")
    episodes = pa.table(
        {
            "episode_index": [0, 1],
            "length": [2, 3],
            "dataset_from_index": [0, 2],
            "dataset_to_index": [2, 5],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "omg/dataset": ["amass_train", "amass_test"],
            "tasks": [["walk"], ["walk"]],
        }
    )
    pq.write_table(episodes, source / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(
        pa.table({"task_index": [0], "task": ["walk"]}), source / "meta/tasks.parquet"
    )
    (source / "meta/info.json").write_text(
        json.dumps(
            dict(
                fps=30,
                total_episodes=2,
                total_frames=5,
                splits={"train": "0:1", "test": "1:2"},
            )
        )
    )
    (source / "meta/omg_manifest.json").write_text(
        json.dumps({"repo_id": "THU-MARS/OMG-Data"})
    )
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    hash_file = tmp_path / "hashes.json"
    hash_file.write_text(json.dumps(hashes))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resample",
            "--source",
            str(source),
            "--dest",
            str(target),
            "--source-hashes",
            str(hash_file),
            "--workers",
            "1",
        ],
    )
    resample_release.main()
    assert (
        json.loads((target / "provenance/source_file_sha256.json").read_text())
        == hashes
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--source",
            str(source),
            "--target",
            str(target),
            "--workers",
            "1",
        ],
    )
    validate_resampled_release.main()
    assert (
        json.loads((target / "provenance/validation_report.json").read_text())["status"]
        == "passed"
    )

    # Metadata is pinned too, not just the frame shards.
    (source / "meta/omg_manifest.json").write_text('{"repo_id":"changed"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resample",
            "--source",
            str(source),
            "--dest",
            str(target),
            "--source-hashes",
            str(hash_file),
            "--workers",
            "1",
            "--resume",
        ],
    )
    with pytest.raises(ValueError, match="Source hash mismatch"):
        resample_release.main()
