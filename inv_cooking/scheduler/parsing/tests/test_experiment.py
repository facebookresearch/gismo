import pytest
from omegaconf import OmegaConf

from inv_cooking.scheduler.parsing.experiment import (
    Search,
    Experiments,
    _all_variables_assignments,
    _collect_variable_paths,
    _expand_search,
    parse_experiments,
)


def test_hyper_parameter_search_simple():
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


def test_hyper_parameter_search_with_list():
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


def test_parse_experiments():
    conf = OmegaConf.create({
        "im2ingr": {
            "exp1":
                {
                    "comment": "comment1",
                    "optimization": {"lr": 1.0},
                }
        },
        "im2recipe": {
            "exp2":
                {
                    "comment": "comment2",
                    "optimization": {"lr": "SEARCH[1.0,2.0]"},
                }
        },
        "ingr2recipe": {
            "exp3":
                {
                    "comment": "comment3",
                    "optimization": {"lr": 1.0},
                },
            "exp4":
                {
                    "parent": "exp3",
                    "optimization": {"lr": "SEARCH[1.0,2.0]"},
                },
            "bad_exp6":
                {
                    "parent": "exp5",
                },
        }
    })
    schema = OmegaConf.structured(Experiments)
    conf = OmegaConf.merge(schema, conf)

    # Test standard configuration (no hyper-parameters and no inheritance)
    out = parse_experiments(conf, "im2ingr", "exp1")
    assert len(out) == 1
    assert out[0]['name'] == "exp1"
    assert out[0]['comment'] == "comment1"
    assert out[0]['optimization']["lr"] == 1.0

    # Test hyper-parameter search
    out = parse_experiments(conf, "im2recipe", "exp2")
    assert len(out) == 2
    assert out[0]['name'] == "exp2"
    assert out[0]['comment'] == "comment2"
    assert out[0]['optimization']["lr"] == 1.0
    assert out[1]['optimization']["lr"] == 2.0

    # Test inheritance plus hyper-parameter search
    out = parse_experiments(conf, "ingr2recipe", "exp4")
    assert len(out) == 2
    assert out[0]['name'] == "exp4"
    assert out[0]['comment'] == "comment3"
    assert out[0]['optimization']["lr"] == 1.0
    assert out[1]['optimization']["lr"] == 2.0

    # Test missing experiment (as parent or direct target)
    with pytest.raises(ValueError, match="Could not find experiment named exp5"):
        parse_experiments(conf, "ingr2recipe", "exp5")
    with pytest.raises(ValueError, match="Could not find experiment named exp5"):
        parse_experiments(conf, "ingr2recipe", "bad_exp6")

    # Test missing task
    with pytest.raises(ValueError, match="Unknown task bad"):
        parse_experiments(conf, "bad", "exp1")

