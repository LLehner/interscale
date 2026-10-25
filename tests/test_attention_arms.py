"""Generated sweep arms (attention_plan.md, "Test matrix"): every combination the config accepts.

Pinned here: the validators decide what is kept, with the reason for each rejection; every arm sets
the same keys, as ``load_arms`` requires; the file round-trips through the sweep machinery
(``build_sweep_config``, ``load_arms``, ``apply_sweep_config``) into configs that still validate;
and arms whose checkpoints would overwrite each other are named.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

from interscale.config.sweep import (
    GLOBAL_PARAMETERS,
    apply_sweep_config,
    arms_yaml,
    build_sweep_config,
    generate_arms,
    load_arms,
    parse_backend,
    parse_encodings,
    parse_grid,
)

KIND = GLOBAL_PARAMETERS + "attention.kind"
HEAD_DIM = GLOBAL_PARAMETERS + "attention.head_dim"


@pytest.fixture
def base(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(
        "model:\n  n_embed: 64\n  global_component:\n    name: self-attn-transformer\n"
        "dataset:\n  spatial_key: spatial\n  prediction_task: regression\n"
    )
    return [path]


def _generate(base, **kwargs):
    kwargs.setdefault("backends", ["dense", "performer", "linear:attention.head_dim=32", "exphormer"])
    kwargs.setdefault("node", ["none", "lap"])
    kwargs.setdefault("bias", ["none", "distance", "fourier", "spectral"])
    kwargs.setdefault("rotary", ["none", "rope"])
    return generate_arms(base, **kwargs)


# --------------------------------------------------------------------------- specs


def test_encoding_levels_are_none_or_names_joined_by_plus():
    assert parse_encodings("none") == [] and parse_encodings("") == []
    assert parse_encodings("lap+rw") == ["lap", "rw"]
    for bad in ("lap+none", "lap+", "lap+lap"):
        with pytest.raises(ValueError):
            parse_encodings(bad)


def test_backend_specs_take_a_label_and_yaml_settings():
    assert parse_backend("dense") == ("dense", "dense", {})
    assert parse_backend("linear:attention.head_dim=32") == ("linear", "linear", {"attention.head_dim": 32})
    assert parse_backend("wide=linear:attention.head_dim=64,optim.lr=1e-4") == (
        "wide",
        "linear",
        {"attention.head_dim": 64, "optim.lr": 1e-4},
    )
    assert parse_backend("dense:probe.attention_radii_um=[100, 300]")[2] == {"probe.attention_radii_um": [100, 300]}
    for bad in ("", "=dense", "dense:head_dim", "de nse"):
        with pytest.raises(ValueError):
            parse_backend(bad)


def test_grid_specs_are_a_key_and_its_values():
    assert parse_grid("model.n_embed=64,128") == ("model.n_embed", [64, 128])
    assert parse_grid("optim.lr=1e-4,3e-4") == ("optim.lr", [1e-4, 3e-4])
    assert parse_grid("pe.node=[lap],[lap, rw]") == ("pe.node", [["lap"], ["lap", "rw"]])
    for bad in ("model.n_embed", "=64", "model.n_embed="):
        with pytest.raises(ValueError):
            parse_grid(bad)


# --------------------------------------------------------------------------- what is kept


def test_the_validators_decide_what_is_kept(base):
    generated = _generate(base)

    rejected = set(generated.rejected)
    assert {
        "performer_distance",
        "performer_lap_distance_rope",
        "linear_rope",
        "linear_fourier",
        "linear_spectral",
    } <= rejected
    assert {"dense_lap_distance_rope", "performer_lap_spectral_rope", "exphormer_distance", "linear_lap"} <= set(
        generated.arms
    )
    assert not rejected & set(generated.arms)
    assert len(generated.arms) + len(rejected) == 4 * 2 * 4 * 2
    assert "cannot carry the attention bias 'distance'" in generated.rejected["performer_distance"]
    assert "cannot apply pe.rotary" in generated.rejected["linear_rope"]


def test_every_arm_sets_the_same_keys_and_a_backend_without_a_setting_keeps_the_base_value(base):
    generated = _generate(base)

    keys = {frozenset(overrides) for overrides in generated.arms.values()}
    assert len(keys) == 1
    assert generated.arms["linear_lap"][HEAD_DIM] == 32
    assert generated.arms["dense_lap"][HEAD_DIM] == 0
    assert generated.arms["exphormer_lap_fourier_rope"][GLOBAL_PARAMETERS + "pe.bias"] == ["fourier"]
    assert generated.arms["dense"][GLOBAL_PARAMETERS + "pe.node"] == []


def test_a_typo_in_a_setting_fails_instead_of_rejecting_every_combination(base):
    with pytest.raises(KeyError, match="attention.head_dims"):
        _generate(base, backends=["linear:attention.head_dims=32"])


def test_labels_must_tell_backends_apart(base):
    with pytest.raises(ValueError, match="unique"):
        _generate(base, backends=["linear:attention.head_dim=32", "linear:attention.head_dim=64"])
    with pytest.raises(ValueError, match="sets itself"):
        _generate(base, backends=["dense:pe.node=[lap]"])


def test_arms_that_would_share_a_checkpoint_are_named(base):
    """The checkpoint name carries the encodings, the backend and its head width -- not the
    feature count -- so two performers differing only in it would overwrite each other's file."""
    generated = _generate(
        base,
        backends=[
            "performer",
            "many=performer:attention.performer.num_features=512",
            "wide=performer:attention.head_dim=32",
        ],
        node=["none"],
        bias=["none"],
        rotary=["none"],
    )

    assert generated.shared_checkpoints == [["performer", "many"]]
    assert "performer, many" in arms_yaml(generated)


def _one_level(base, backends, grid):
    return _generate(base, backends=backends, node=["none"], bias=["none"], rotary=["none"], grid=grid)


def test_an_arm_is_kept_only_if_it_validates_at_every_grid_point(base):
    """64 is not divisible by 5 heads: dense at the default head width fails there, while linear
    sets its own width."""
    generated = _one_level(base, ["dense", "linear:attention.head_dim=32"], {"n_heads": [4, 5]})

    assert set(generated.arms) == {"linear"}
    assert generated.rejected["dense"].startswith(
        "at model.global_component.parameters.n_heads=5: model.n_embed (64) is not divisible by n_heads (5)"
    )
    assert generated.grid == {GLOBAL_PARAMETERS + "n_heads": [4, 5]}


def test_grid_keys_reach_the_sweep_and_trials_differing_only_in_them_share_a_checkpoint(base):
    from interscale.config import _validate, load_config

    generated = _one_level(base, ["dense", "performer"], {"model.n_embed": [64, 128]})

    assert generated.shared_checkpoints == [
        ["dense (model.n_embed=64)", "dense (model.n_embed=128)"],
        ["performer (model.n_embed=64)", "performer (model.n_embed=128)"],
    ]
    parsed = yaml.safe_load(arms_yaml(generated))
    sweep_config, sweep_params = build_sweep_config(parsed, prediction_task="regression")
    assert sweep_config["parameters"]["model.n_embed"] == {"values": [64, 128]}
    trial = {"arm": "performer", "model.n_embed": 128, "optim.seed": 0}
    cfg, _ = apply_sweep_config(
        load_config(base), "hyperparmeter", trial, sweep_params=sweep_params, arms=load_arms(parsed, sweep_config)
    )
    _validate(cfg)
    assert cfg.model.n_embed == 128 and cfg.model.global_component.parameters.attention.kind == "performer"


@pytest.mark.parametrize("key", ["attention.kind", "attention.head_dim", "optim.seed"])
def test_a_grid_key_the_arms_set_is_refused(base, key):
    with pytest.raises(ValueError, match="set by the arms"):
        _one_level(base, ["linear:attention.head_dim=32"], {key: [1]})


# --------------------------------------------------------------------------- the file


def test_the_file_round_trips_through_the_sweep_machinery(base):
    from interscale.config import _validate, load_config

    generated = _generate(base)
    text = arms_yaml(generated, seeds=[0, 1], header=["made by a test"])
    parsed = yaml.safe_load(text)

    sweep_config, sweep_params = build_sweep_config(parsed, prediction_task="regression")
    arms = load_arms(parsed, sweep_config)
    assert arms == generated.arms
    assert sweep_config["parameters"]["arm"]["values"] == list(generated.arms)
    assert sweep_config["parameters"]["optim.seed"]["values"] == [0, 1]

    for name in generated.arms:
        cfg, applied = apply_sweep_config(
            load_config(base), "hyperparmeter", {"arm": name, "optim.seed": 1}, sweep_params=sweep_params, arms=arms
        )
        _validate(cfg)  # what the sweep trains is a config the validators accept
        assert cfg.model.global_component.parameters.attention.kind == generated.arms[name][KIND]
        assert cfg.optim.seed == 1


def test_the_header_gives_each_rejection_its_reason_once(base):
    text = arms_yaml(_generate(base), header=["made by a test"])
    header = [line for line in text.splitlines() if line.startswith("#")]

    assert header[0] == "# made by a test"
    assert (
        sum("cannot carry the attention bias 'distance': the bias comes in the forms" in line for line in header) == 2
    )
    assert any("performer_distance, performer_distance_rope" in line for line in header)


def test_no_accepted_combination_writes_no_file(base):
    generated = _generate(base, backends=["linear"], bias=["fourier"], node=["none"], rotary=["none"])
    with pytest.raises(ValueError, match="no combination"):
        arms_yaml(generated)


def test_the_script_writes_the_file(base, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_attention_arms.py"
    spec = importlib.util.spec_from_file_location("make_attention_arms", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "arms.yaml"

    argv = ["--cfg", str(base[0]), "--bias", "none", "--bias", "fourier", "--grid", "num_layers=1,2", "--seeds", "3"]
    generated = module.main([*argv, "--out", str(out)])

    parsed = yaml.safe_load(out.read_text())
    assert (
        set(parsed["arms"])
        == set(generated.arms)
        == {
            "dense",
            "dense_fourier",
            "entmax",
            "entmax_fourier",
            "exphormer",
            "exphormer_fourier",
            "fused",
            "fused_fourier",
            "linear",
            "performer",
            "performer_fourier",
            "topk",
            "topk_fourier",
        }
    )
    assert parsed["sweep_config"]["parameters"]["optim.seed"]["values"] == [3]
    assert parsed["sweep_config"]["parameters"][GLOBAL_PARAMETERS + "num_layers"]["values"] == [1, 2]
    assert out.read_text().startswith("# Generated by: scripts/make_attention_arms.py --cfg")
