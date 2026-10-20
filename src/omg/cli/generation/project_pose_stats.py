"""Project matching 125D rot6d statistics onto the unchanged 38 pose channels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_FEATURE = "root_pos_local+root_rot_local_rot6d_6+joint_dof_29+body_link_pos_local_29x3"


def project_pose_stats(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    stats = json.loads(raw)
    if stats.get("feature") != SOURCE_FEATURE or stats.get("rotation_representation") != "rot6d":
        raise ValueError("Expected the 125D rot6d feature layout")
    if len(stats["mean"]) != 125 or len(stats["std"]) != 125:
        raise ValueError("Expected 125 mean and std entries")
    stats["mean"] = stats["mean"][:38]
    stats["std"] = stats["std"][:38]
    stats["feature"] = SOURCE_FEATURE.removesuffix("+body_link_pos_local_29x3")
    stats["include_body_positions"] = False
    stats["projection"] = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "feature_indices": list(range(38)),
    }
    # An explicit migration, never an implicit truncation in the training loader.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project_pose_stats(args.source, args.output)


if __name__ == "__main__":
    main()
