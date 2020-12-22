from omegaconf import OmegaConf

from inv_cooking.config.parsing.experiment import (
    Search,
    _all_variables_assignments,
    _collect_variable_paths,
    _expand_search,
)


def test_config_generation_simple():
    conf = OmegaConf.create(
        {"comment": "hello", "optimization": {"lr": "SEARCH[1.0,2.0]", "wd": 1e-5}}
    )

    paths = _collect_variable_paths(conf)
    assert paths == [("optimization.lr", Search(values=[1.0, 2.0]))]

    assignments = _all_variables_assignments(paths)
    assert len(assignments) == 2

    generated = _expand_search(conf)
    assert len(generated) == 2
    assert generated[0] == OmegaConf.create(
        {"comment": "hello", "optimization": {"lr": 1.0, "wd": 1e-5}}
    )
    assert generated[1] == OmegaConf.create(
        {"comment": "hello", "optimization": {"lr": 2.0, "wd": 1e-5}}
    )


def test_config_generation_with_lists():
    conf = OmegaConf.create(
        {
            "comment": "hello",
            "optimization": [{"lr1": "SEARCH[1.0,2.0]"}, {"lr2": "SEARCH[1.0,2.0]"}],
        }
    )

    paths = _collect_variable_paths(conf)
    assert paths == [
        ("optimization.0.lr1", Search(values=[1.0, 2.0])),
        ("optimization.1.lr2", Search(values=[1.0, 2.0])),
    ]

    assignments = _all_variables_assignments(paths)
    assert len(assignments) == 4

    generated = _expand_search(conf)
    assert len(generated) == 4
