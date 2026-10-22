import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import yaml


def test_multimodal_preset_matches_release_statistics(monkeypatch):
    monkeypatch.delenv("OMG_LITE_MULTIMODAL_STATS", raising=False)
    root = Path(__file__).resolve().parents[2]
    stats = json.loads(
        (root / "assets/stats/g1_38d_multimodal_50hz_hist10_stats.json").read_text()
    )
    direct = yaml.safe_load(
        (root / "configs/generation/data/omg_lite_multimodal_50hz.yaml").read_text()
    )
    with initialize_config_dir(
        config_dir=str(root / "configs/generation"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=["exp=multimodal_50hz"])
    assert cfg.model.use_audio and cfg.model.use_human_motion
    assert all(v == 0 for k, v in cfg.loss.items() if k != "_target_")
    assert (
        cfg.callbacks.fixed_motion_evaluation.expected_batches_per_epoch
        == (stats["windows"] + 6143) // 6144
    )
    for split in ("train", "val", "test"):
        a = next(iter(direct["dataset_opts"][split].values()))
        b = next(iter(cfg.data.dataset_opts[split].values()))
        assert a["repo_id"] == b.source_repo_id == stats["source_repo_id"]
        assert a["revision"] == b.source_revision
        assert len(a["revision"]) == 40 and all(
            c in "0123456789abcdef" for c in a["revision"]
        )
        assert a["manifest_sha256"] == stats["source_manifest_sha256"]
    representation = instantiate(cfg.representation)
    assert representation.feat_dim == 38
