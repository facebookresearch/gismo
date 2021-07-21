from omegaconf import MISSING, DictConfig

from inv_cooking.utils.hydra import merge_with_non_missing


def test_merge_with_non_missing():
    base = DictConfig(
        dict(
            batch_size=32,
            num_workers=5,
        )
    )
    replacement = DictConfig(
        dict(
            batch_size=16,
            num_workers=MISSING,
        )
    )
    merged = merge_with_non_missing(base, replacement)
    assert merged["batch_size"] == 16
    assert merged["num_workers"] == 5
