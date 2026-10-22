"""Independent full-frame audit of a 30-to-50 Hz public release."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def matrix(table, name):
    a = table[name].combine_chunks()
    return a.values.to_numpy().reshape(len(a), -1)


def check_file(job):
    source, target, rel, episodes, expected_hash = job
    a = pq.read_table(Path(source) / rel)
    b = pq.read_table(Path(target) / rel)
    assert digest(Path(target) / rel) == expected_hash
    assert a.schema == b.schema
    qa, qb = matrix(a, "observation.state"), matrix(b, "observation.state")
    action = matrix(b, "action")
    audio_a, audio_b = matrix(a, "omg.audio.feature"), matrix(b, "omg.audio.feature")
    human_a, human_b = (
        matrix(a, "omg.humanref.motion"),
        matrix(b, "omg.humanref.motion"),
    )
    for x in (qa, qb, action, audio_a, audio_b, human_a, human_b):
        assert np.isfinite(x).all()
    masks = [
        "omg.condition.has_text",
        "omg.condition.has_audio",
        "omg.condition.has_humanref",
        "task_index",
    ]
    am = {k: a[k].to_numpy() for k in masks}
    bm = {k: b[k].to_numpy() for k in masks}
    frame_b = b["frame_index"].to_numpy()
    time_b = b["timestamp"].to_numpy()
    ep_b = b["episode_index"].to_numpy()
    index_b = b["index"].to_numpy()
    cursor_a = cursor_b = 0
    event_count = 0
    for ep in episodes:
        n = int(ep["omg/resample_source_length"])
        m = int(ep["length"])
        assert m == int(np.floor((n - 1) * 50 / 30)) + 1
        sa = slice(cursor_a, cursor_a + n)
        sb = slice(cursor_b, cursor_b + m)
        t = np.arange(m, dtype=np.float64) * 3 / 5
        left = np.floor(t).astype(np.int64)
        right = np.minimum(left + 1, n - 1)
        w = (t - left)[:, None]
        x = qa[sa]
        y = qb[sb]
        for cols in (slice(0, 3), slice(7, 36)):
            expected = x[left, cols] * (1 - w) + x[right, cols] * w
            np.testing.assert_allclose(y[:, cols], expected, rtol=2e-6, atol=2e-6)
        # An independent geometric SLERP check: unit length and angular distance
        # to both endpoints must equal the prescribed shortest-arc fractions.
        u = x[left, 3:7].astype(np.float64)
        v = x[right, 3:7].astype(np.float64)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        z = y[:, 3:7].astype(np.float64)
        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=2e-7, rtol=0)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        angle = np.arccos(np.clip(np.abs(np.sum(u * v, axis=1)), 0, 1))
        for endpoint, fraction in [(u, w[:, 0]), (v, 1 - w[:, 0])]:
            actual = np.arccos(np.clip(np.abs(np.sum(endpoint * z, axis=1)), 0, 1))
            np.testing.assert_allclose(actual, angle * fraction, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(action[sb][:-1], y[1:])
        np.testing.assert_array_equal(action[cursor_b + m - 1], y[-1])
        np.testing.assert_array_equal(frame_b[sb], np.arange(m))
        np.testing.assert_array_equal(
            time_b[sb], (np.arange(m) / 50).astype(np.float32)
        )
        np.testing.assert_array_equal(ep_b[sb], ep["episode_index"])
        np.testing.assert_array_equal(
            index_b[sb], np.arange(ep["dataset_from_index"], ep["dataset_to_index"])
        )
        for name in masks:
            assert np.all(am[name][sa] == am[name][cursor_a])
            assert np.all(bm[name][sb] == am[name][cursor_a])
        for old, new, is_audio in [
            (audio_a[sa], audio_b[sb], True),
            (human_a[sa], human_b[sb], False),
        ]:
            events = is_audio and ep["omg/audio_feature_type"] == "aistpp35_events"
            width = 33 if events else old.shape[1]
            expected = old[left, :width] * (1 - w) + old[right, :width] * w
            np.testing.assert_allclose(new[:, :width], expected, rtol=2e-6, atol=2e-6)
            if events:
                expected = np.zeros((m, 2), np.float32)
                for ch in range(2):
                    frames = np.flatnonzero(old[:, 33 + ch])
                    # Nearest target timestamp, bounded to the declared domain.
                    indices = np.minimum(
                        np.floor(frames / 30 * 50 + 0.5).astype(int), m - 1
                    )
                    expected[indices, ch] = 1
                    event_count += len(frames)
                np.testing.assert_array_equal(new[:, 33:], expected)
        cursor_a += n
        cursor_b += m
    assert cursor_a == len(qa) and cursor_b == len(qb)
    return dict(
        file=rel,
        episodes=len(episodes),
        frames=len(qb),
        binary_audio_events=event_count,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    report = json.loads((args.target / "provenance/build_report.json").read_text())
    old = []
    new = []
    for path in sorted((args.source / "meta/episodes").rglob("*.parquet")):
        a = pq.read_table(path)
        b = pq.read_table(args.target / path.relative_to(args.source))
        for key in a.column_names:
            if key not in ["length", "dataset_from_index", "dataset_to_index"]:
                assert a[key].equals(b[key]), f"Metadata changed: {key}"
        old.extend(a.to_pylist())
        new.extend(b.to_pylist())
    assert digest(args.source / "meta/tasks.parquet") == digest(
        args.target / "meta/tasks.parquet"
    )
    source_info = json.loads((args.source / "meta/info.json").read_text())
    target_info = json.loads((args.target / "meta/info.json").read_text())
    assert source_info["splits"] == target_info["splits"]
    assert target_info["fps"] == 50
    jobs = []
    groups = {}
    for ep in new:
        rel = f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
        groups.setdefault(rel, []).append(ep)
    for shard in report["shards"]:
        rel = shard["file"]
        jobs.append(
            (str(args.source), str(args.target), rel, groups[rel], shard["sha256"])
        )
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(check_file, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
    validation = dict(
        status="passed",
        full_frame_validation=True,
        episodes=len(new),
        frames=sum(r["frames"] for r in results),
        files=results,
        source_revision=report["source_revision"],
        validator_sha256=digest(__file__),
    )
    assert validation["frames"] == target_info["total_frames"]
    (args.target / "provenance/validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    report["status"] = "validated"
    report["validation_report_sha256"] = digest(
        args.target / "provenance/validation_report.json"
    )
    (args.target / "provenance/build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
