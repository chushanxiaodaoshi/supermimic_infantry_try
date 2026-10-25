from types import SimpleNamespace

import pytest
import torch

from tools.collect_mjlab_teacher import configure_collection, frame_major_history


def test_history_conversion_preserves_time_and_feature_order() -> None:
    expected = torch.arange(2 * 5 * 25).reshape(2, 5, 25)
    raw = torch.cat(
        [term.flatten(1) for term in expected.split((3, 3, 3, 4, 6, 6), dim=-1)], dim=-1
    )
    assert not torch.equal(raw.reshape(2, 5, 25), expected)
    torch.testing.assert_close(frame_major_history(raw), expected)


def test_collection_removes_unobserved_dynamics() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="plane")),
        events={
            "push_robots": object(),
            "body_torque": object(),
            "body_mass_inertia": SimpleNamespace(
                params={
                    "inertia_scale_range": (0.9, 1.1),
                    "added_mass_range": (-0.7, 1.4),
                }
            ),
            "wheel_friction": SimpleNamespace(
                params={"asset_cfg": SimpleNamespace(geom_names=(".*",))}
            ),
            "restitution": SimpleNamespace(
                params={"asset_cfg": SimpleNamespace(geom_names=(".*",))}
            ),
            "gas_spring_force": object(),
        },
    )
    spring = cfg.events["gas_spring_force"]
    profile = configure_collection(cfg, seed=17)
    assert cfg.seed == 17
    assert not {"push_robots", "body_torque"} & cfg.events.keys()
    assert cfg.events["body_mass_inertia"].params["inertia_scale_range"] == (1.0, 1.0)
    assert cfg.events["body_mass_inertia"].params["added_mass_range"] == (-0.7, 1.4)
    for name in ("wheel_friction", "restitution"):
        assert (
            list(cfg.events[name].params["asset_cfg"].geom_names)
            == profile["randomized_contact_geoms"]
        )
    assert cfg.events["gas_spring_force"] is spring


def test_collection_rejects_terrain_missing_from_model_inputs() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="generator"))
    )
    with pytest.raises(ValueError, match="no terrain input"):
        configure_collection(cfg, seed=42)
