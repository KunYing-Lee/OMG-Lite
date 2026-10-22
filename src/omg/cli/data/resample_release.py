#!/usr/bin/env python3
"""Deterministic, episode-local 30 -> 50 Hz OMG LeRobot v3 resampling."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import concurrent.futures as cf
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_REVISION = "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"
EVENT_AUDIO_FAMILIES = {"aistpp", "finedance"}
CONTINUOUS_AUDIO_FAMILIES = {"aioz_gdance", "choreomaster", "compas3d", "opendance"}


def resample_conditions(values, lengths, families=None):
    """Interpolate continuous coordinates; scatter declared binary events by time.

    Nearest in-domain sample is defined by minimising absolute timestamp error.
    No interpolation crosses an episode boundary. No event detector is applied.
    """
    if not np.isfinite(values).all():
        raise ValueError("Non-finite condition values")
    lens, starts, out_starts, frame, left, right, fraction = coordinates(lengths)
    out = np.empty((len(frame), values.shape[1]), np.float32)
    for lo in range(0, len(out), 131072):
        sl = slice(lo, lo + 131072)
        a = fraction[sl, None]
        out[sl] = (1 - a) * values[left[sl]] + a * values[right[sl]]
    if families is not None:
        for start, length, target, count, family in zip(
            starts, lengths, out_starts, lens, families
        ):
            source = values[start : start + length]
            if family in EVENT_AUDIO_FAMILIES:
                events = source[:, 33:35]
                if not np.isin(events, [0, 1]).all():
                    raise ValueError(f"Non-binary event channels in {family}")
                out[target : target + count, 33:35] = 0
                frames, channels = np.nonzero(events)
                nearest = np.minimum((frames * 5 + 1) // 3, count - 1)
                out[target + nearest, 33 + channels] = 1
                if np.count_nonzero(out[target : target + count, 33:35]) != len(frames):
                    raise ValueError("Event collision on target grid")
            elif family not in CONTINUOUS_AUDIO_FAMILIES and np.any(source):
                raise ValueError(f"Undeclared audio semantics for {family}")
    return out


DATASET_CARD = """---
language:
- en
tags:
- robotics
- motion-generation
- multimodal
---
# OMG-Lite-Data: 50 Hz multimodal motion

A deterministic 50 Hz derivative of THU-MARS/OMG-Data at revision
`6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7`. All source episodes, captions,
condition availability, source identities, and train/val/test membership are retained.
Consult the upstream dataset for source dataset licenses and attribution.

## Representation and time

LeRobot v3 public schema retains 36D qpos (XYZ, WXYZ quaternion, 29 joint angles),
next-state action, 35D audio, 66D human reference, and modality masks. OMG-Lite
encodes qpos into 38D (29 joints + XYZ + 6D rotation); 38D is not raw qpos.
Uniform timestamps are j/50 within [0,(N-1)/30]. No extrapolation or time scaling.
Translation, bounded joint angles, and human joint positions use linear interpolation;
root quaternions use shortest-arc SLERP. Action is rebuilt from next target state,
with terminal repetition. Source frame offsets remain in original 30 Hz units.
The last target timestamp can precede the original endpoint by at most 13.334 ms.
This resampling does not create new measured motion detail.

## Conditions

The upstream audio feature definitions differ by source family. AIST++/FineDance
channels 33 and 34 are binary peak/beat events: events are assigned to the nearest
in-domain 50 Hz timestamp. Other channels are linearly interpolated. For AIOZ-GDance,
ChoreoMaster, COMPAS3D, and OpenDance all 35 channels are continuous and linearly
interpolated. These are resampled released features, not features re-extracted from
raw audio. They must not be assumed to share one audio encoder definition.
Human reference is flattened 22x3 Cartesian joint positions in the source convention.
Missing conditions and availability flags are preserved; no conditions are fabricated.

## Reproducibility

`provenance/build_report.json` records algorithms, counts, source/output hashes and
versions. Episode metadata retains original provenance plus resampling lengths and
source FPS. Official splits are unchanged, including any upstream overlap: this
release does not claim recording-level or motion-content deduplication.
`meta/stats.json` contains frame statistics, not 38D model normalization. Compute model
normalization from training windows only. Existing checkpoints retain their own stats.

Read splits through `meta/info.json` and episode metadata. Parquet files can cross split
boundaries; do not treat an automatically inferred Hugging Face `train` split as the
official training split.

Use OMG-Lite's pinned dataset config and compact episode-cache materializer. Compact
cache stores qpos, conditions, masks, captions and episode provenance, without redundant
FK body positions/quaternions. See the [OMG-Lite data guide](https://github.com/KunYing-Lee/OMG-Lite/blob/main/docs/data/multimodal_50hz.md) for commands.
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def matrix(table, name):
    a = table[name].combine_chunks()
    return a.values.to_numpy(zero_copy_only=False).reshape(len(a), -1)


def stats(a):
    total = np.zeros(a.shape[1], np.float64)
    square = np.zeros(a.shape[1], np.float64)
    for lo in range(0, len(a), 131072):
        x = a[lo : lo + 131072].astype(np.float64)
        total += x.sum(axis=0)
        square += np.einsum("ij,ij->j", x, x)
    return dict(
        count=len(a),
        sum=total.tolist(),
        square=square.tolist(),
        min=a.min(axis=0).tolist(),
        max=a.max(axis=0).tolist(),
    )


def coordinates(lengths):
    new_lengths = (lengths - 1) * 5 // 3 + 1
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    new_starts = np.r_[0, np.cumsum(new_lengths)[:-1]]
    frame = np.arange(new_lengths.sum()) - np.repeat(new_starts, new_lengths)
    left = np.repeat(starts, new_lengths) + frame * 3 // 5
    right = np.minimum(left + 1, np.repeat(starts + lengths - 1, new_lengths))
    fraction = ((frame * 3) % 5).astype(np.float64) / 5
    return new_lengths, starts, new_starts, frame, left, right, fraction


def interpolate(q, lengths):
    lens, starts, new_starts, frame, left, right, fraction = coordinates(lengths)
    result = np.empty((len(frame), 36), np.float32)
    quat = q[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(quat, axis=1)
    norm_error = float(np.max(np.abs(norms - 1)))
    if norm_error > 1e-3:
        raise ValueError(f"Invalid source unit quaternions: max error {norm_error}")
    quat /= norms[:, None]
    # Resolve q/-q representation continuously, independently in each episode.
    flip = np.zeros(len(q), dtype=np.int64)
    flip[1:] = np.einsum("ij,ij->i", quat[:-1], quat[1:]) < 0
    flip[starts] = 0
    parity = np.cumsum(flip) % 2
    parity ^= np.repeat(parity[starts], lengths)
    quat *= (1 - 2 * parity)[:, None]
    for lo in range(0, len(frame), 131072):
        sl = slice(lo, lo + 131072)
        a = fraction[sl, None]
        p = left[sl]
        r = right[sl]
        result[sl] = (1 - a) * q[p].astype(np.float64) + a * q[r].astype(np.float64)
        q0 = quat[p]
        q1 = quat[r]
        theta = np.arccos(np.clip(np.einsum("ij,ij->i", q0, q1), -1.0, 1.0))[:, None]
        # Analytic sinc form of SLERP includes the exact theta=0 limit.
        denominator = np.sinc(theta / np.pi)
        w0 = (1 - a) * np.sinc((1 - a) * theta / np.pi) / denominator
        w1 = a * np.sinc(a * theta / np.pi) / denominator
        result[sl, 3:7] = w0 * q0 + w1 * q1
    return result, (lens, starts, new_starts, frame), norm_error


def worker(job):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    (
        source,
        dest,
        rel,
        lengths,
        episodes,
        source_offset,
        target_offset,
        families,
        expected_sha,
    ) = job
    begin = time.monotonic()
    path = Path(source) / rel
    record_path = Path(dest) / "provenance" / f"{Path(rel).stem}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record["source_sha256"] != expected_sha
            or sha(Path(dest) / rel) != record["sha256"]
        ):
            raise ValueError(f"Resume hash mismatch: {rel}")
        return record
    before = path.stat()
    if sha(path) != expected_sha:
        raise ValueError(f"Source hash mismatch: {rel}")
    table = pq.read_table(path)
    lengths = np.asarray(lengths, np.int64)
    episodes = np.asarray(episodes, np.int64)
    assert table.num_rows == int(lengths.sum())
    assert set(table.column_names) == {
        "observation.state",
        "action",
        "omg.condition.has_text",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "omg.audio.feature",
        "omg.humanref.motion",
        "omg.condition.has_audio",
        "omg.condition.has_humanref",
    }
    q = matrix(table, "observation.state")
    action = matrix(table, "action")
    assert np.isfinite(q).all() and np.isfinite(action).all()
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = starts + lengths - 1
    next_row = np.arange(len(q)) + 1
    next_row[ends] = ends
    assert np.array_equal(action, q[next_row]), "Source action semantics differ"
    local_frame = np.arange(len(q)) - np.repeat(starts, lengths)
    assert np.array_equal(table["frame_index"].to_numpy(), local_frame)
    assert np.array_equal(
        table["index"].to_numpy(), np.arange(source_offset, source_offset + len(q))
    )
    assert np.array_equal(
        table["episode_index"].to_numpy(), np.repeat(episodes, lengths)
    )
    assert np.allclose(
        table["timestamp"].to_numpy(),
        local_frame / 30.0,
        atol=0,
        rtol=np.finfo(np.float32).eps,
    )
    constants = {}
    for name in [
        "task_index",
        "omg.condition.has_text",
        "omg.condition.has_audio",
        "omg.condition.has_humanref",
    ]:
        v = table[name].to_numpy()
        assert np.array_equal(v, np.repeat(v[starts], lengths)), f"Nonconstant {name}"
        constants[name] = v[starts]
    out, coords, norm_error = interpolate(q, lengths)
    lens, starts, out_starts, frame = coords
    out_action = np.empty_like(out)
    out_action[:-1] = out[1:]
    out_ends = out_starts + lens - 1
    out_action[out_ends] = out[out_ends]
    assert np.isfinite(out).all()
    assert (
        np.max(np.abs(np.linalg.norm(out[:, 3:7].astype(np.float64), axis=1) - 1))
        < 2e-7
    )
    columns = {
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(out.reshape(-1)), 36
        ),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(out_action.reshape(-1)), 36
        ),
        "omg.condition.has_text": pa.array(
            np.repeat(constants["omg.condition.has_text"], lens)
        ),
        "timestamp": pa.array((frame / 50.0).astype(np.float32)),
        "frame_index": pa.array(frame),
        "episode_index": pa.array(np.repeat(episodes, lens)),
        "index": pa.array(np.arange(target_offset, target_offset + len(out))),
        "task_index": pa.array(np.repeat(constants["task_index"], lens)),
    }
    audio = matrix(table, "omg.audio.feature")
    human = matrix(table, "omg.humanref.motion")
    if audio.shape[1] != 35 or human.shape[1] != 66:
        raise ValueError("Expected 35D audio and 22x3 human joint positions")
    out_audio = resample_conditions(audio, lengths, families=families)
    out_human = resample_conditions(human, lengths)
    for name, values, width in [
        ("omg.audio.feature", out_audio, 35),
        ("omg.humanref.motion", out_human, 66),
    ]:
        columns[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1)), width
        )
    for name in ["omg.condition.has_audio", "omg.condition.has_humanref"]:
        columns[name] = pa.array(np.repeat(constants[name], lens))
    result = pa.table(
        [columns[name] for name in table.column_names], schema=table.schema
    )
    assert result.schema == table.schema
    output = Path(dest) / rel
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        result, output, compression="zstd", compression_level=3, row_group_size=262144
    )
    # Full source and output hashes are provenance, not just metadata fingerprints.
    source_hash = sha(path)
    assert (before.st_size, before.st_mtime_ns) == (
        path.stat().st_size,
        path.stat().st_mtime_ns,
    )
    record = dict(
        file=rel,
        source_rows=len(q),
        rows=len(out),
        episodes=len(lens),
        source_sha256=source_hash,
        sha256=sha(output),
        bytes=output.stat().st_size,
        max_source_quaternion_norm_error=norm_error,
        stats={
            "observation.state": stats(out),
            "action": stats(out_action),
            "omg.audio.feature": stats(out_audio),
            "omg.humanref.motion": stats(out_human),
        },
        seconds=round(time.monotonic() - begin, 2),
    )
    write_json(Path(dest) / "provenance" / f"{Path(rel).stem}.json", record)
    return record


def merge_stats(records):
    result = {}
    for key in records[0]["stats"]:
        ss = [r["stats"][key] for r in records]
        count = sum(s["count"] for s in ss)
        mean = sum(np.array(s["sum"]) for s in ss) / count
        var = sum(np.array(s["square"]) for s in ss) / count - mean**2
        assert (var >= 0).all()
        result[key] = dict(
            count=[count],
            mean=mean.tolist(),
            std=np.sqrt(var).tolist(),
            min=np.min([s["min"] for s in ss], axis=0).tolist(),
            max=np.max([s["max"] for s in ss], axis=0).tolist(),
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--source-hashes",
        type=Path,
        required=True,
        help="Official pinned file SHA256 mapping",
    )
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    expected = json.loads(a.source_hashes.read_text())
    source = Path(a.source).resolve()
    dest = Path(a.dest).resolve()
    if dest == source:
        raise ValueError("Source and destination must differ")
    for rel, expected_digest in expected.items():
        if not rel.startswith("data/") and sha(source / rel) != expected_digest:
            raise ValueError(f"Source hash mismatch: {rel}")
    if dest.exists() and not a.resume:
        raise FileExistsError(dest)
    if dest.exists() and a.resume:
        marker = dest / "provenance/source_file_sha256.json"
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise ValueError("Resume requires the same pinned source hash manifest")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta").mkdir(exist_ok=True)
    (dest / "provenance").mkdir(exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text())
    assert info["fps"] == 30
    episode_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    tables = [pq.read_table(f) for f in episode_paths]
    ep = pa.concat_tables(tables)
    values = lambda k: ep[k].to_numpy()
    lengths = values("length")
    ids = values("episode_index")
    assert np.array_equal(ids, np.arange(info["total_episodes"]))
    assert int(lengths.sum()) == info["total_frames"] and (lengths > 0).all()
    oldstarts = np.r_[0, np.cumsum(lengths)[:-1]]
    assert np.array_equal(values("dataset_from_index"), oldstarts)
    assert np.array_equal(values("dataset_to_index"), np.cumsum(lengths))
    lens = (lengths - 1) * 5 // 3 + 1
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    chunks = values("data/chunk_index")
    files = values("data/file_index")
    jobs = []
    for chunk, file in sorted(set(zip(chunks.tolist(), files.tolist()))):
        mask = (chunks == chunk) & (files == file)
        ix = np.flatnonzero(mask)
        assert np.array_equal(ix, np.arange(ix[0], ix[-1] + 1))
        rel = f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        jobs.append(
            (
                str(source),
                str(dest),
                rel,
                lengths[mask].tolist(),
                ids[mask].tolist(),
                int(oldstarts[ix[0]]),
                int(starts[ix[0]]),
                [str(v).rsplit("_", 1)[0] for v in values("omg/dataset")[mask]],
                expected[rel],
            )
        )
    expected_files = set(
        str(p.relative_to(source)) for p in (source / "data").rglob("*.parquet")
    )
    assert expected_files == {j[2] for j in jobs}
    # Snapshot original metadata and preserve all source labels, splits and source frame offsets.
    write_json(dest / "provenance/source_file_sha256.json", expected)
    source_meta = []
    for f in sorted((source / "meta").rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(source))
            digest = sha(f)
            if rel in expected and digest != expected[rel]:
                raise ValueError(f"Source hash mismatch: {rel}")
            source_meta.append(dict(file=rel, sha256=digest))
    write_json(dest / "provenance/source_metadata.json", source_meta)
    changes = {
        "length": lens,
        "dataset_from_index": starts,
        "dataset_to_index": np.cumsum(lens),
    }
    for k, v in changes.items():
        ep = ep.set_column(ep.schema.get_field_index(k), k, pa.array(v))
    families = [str(v).rsplit("_", 1)[0] for v in values("omg/dataset")]
    audio_types = [
        "aistpp35_events"
        if f in EVENT_AUDIO_FAMILIES
        else ("current35_continuous" if f in CONTINUOUS_AUDIO_FAMILIES else "none")
        for f in families
    ]
    ep = ep.append_column("omg/audio_feature_type", pa.array(audio_types))
    ep = ep.append_column("omg/resample_source_length", pa.array(lengths))
    ep = ep.append_column(
        "omg/resample_source_fps", pa.array(np.full(len(ids), 30, dtype=np.int64))
    )
    ep = ep.append_column(
        "omg/resample_tail_seconds", pa.array((lengths - 1) / 30.0 - (lens - 1) / 50.0)
    )
    off = 0
    for path, table in zip(episode_paths, tables):
        output = dest / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(ep.slice(off, len(table)), output, compression="zstd")
        off += len(table)
    shutil.copyfile(source / "meta/tasks.parquet", dest / "meta/tasks.parquet")
    print(
        json.dumps(
            dict(
                event="planned",
                source_frames=int(lengths.sum()),
                target_frames=int(lens.sum()),
                episodes=len(ids),
                shards=len(jobs),
                workers=a.workers,
            )
        ),
        flush=True,
    )
    records = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for record in pool.map(worker, jobs):
            records.append(record)
            print(
                json.dumps(
                    dict(
                        event="shard_done",
                        file=record["file"],
                        rows=record["rows"],
                        seconds=record["seconds"],
                        completed=len(records),
                        total=len(jobs),
                    )
                ),
                flush=True,
            )
    info["fps"] = 50
    info["total_frames"] = int(lens.sum())
    info["data_files_size_in_mb"] = int(np.ceil(max(r["bytes"] for r in records) / 1e6))
    write_json(dest / "meta/info.json", info)
    write_json(dest / "meta/stats.json", merge_stats(records))
    manifest = json.loads((source / "meta/omg_manifest.json").read_text())
    manifest.update(
        frames=int(lens.sum()),
        fps=50,
        repo_id="THU-MARS/OMG-Lite-Data",
        source_repo_id="THU-MARS/OMG-Data",
        source_revision=SOURCE_REVISION,
        audio_resampling="source-family specific; see provenance/build_report.json",
    )
    write_json(dest / "meta/omg_manifest.json", manifest)
    split_frames = {}
    for k, r in info["splits"].items():
        lo, hi = map(int, r.split(":"))
        split_frames[k] = int(lens[lo:hi].sum())
    report = dict(
        status="built_pending_independent_validation",
        source=str(source),
        destination=str(dest),
        source_repo_id="THU-MARS/OMG-Data",
        source_revision=SOURCE_REVISION,
        source_fps=30,
        target_fps=50,
        source_frames=int(lengths.sum()),
        frames=int(lens.sum()),
        episodes=len(ids),
        splits=info["splits"],
        split_frames=split_frames,
        source_metadata=source_meta,
        shards=[{k: v for k, v in r.items() if k != "stats"} for r in records],
        max_tail_seconds=float(np.max((lengths - 1) / 30.0 - (lens - 1) / 50.0)),
        algorithm=dict(
            grid="t=j/50, j=0..floor((N-1)*50/30); no time scaling or extrapolation",
            translation="piecewise linear",
            joints="piecewise linear on bounded G1 joint coordinates; no wrapping",
            root_rotation="normalized input, episode-local sign continuity, shortest-arc SLERP",
            action="next 50 Hz state; terminal action repeats terminal state",
            labels="episode task and has_text retained; episode identity, split, and source offsets unchanged",
            audio="continuous channels: piecewise linear; aistpp/finedance channels 33,34: binary events at nearest in-domain 50Hz timestamp",
            humanref="22x3 Cartesian joint positions: piecewise linear",
            masks="constant within each source episode; retained exactly",
            derived_features="FK and velocities computed from qpos at 50 Hz",
        ),
        script_sha256=sha(__file__),
        numpy=np.__version__,
        pyarrow=pa.__version__,
    )
    write_json(dest / "provenance/build_report.json", report)
    shutil.copyfile(__file__, dest / "provenance/build_50hz.py")
    (dest / "README.md").write_text(DATASET_CARD)
    print(
        json.dumps(
            dict(event="build_complete", frames=int(lens.sum()), episodes=len(ids))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
