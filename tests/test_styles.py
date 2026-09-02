from __future__ import annotations

import pytest

from pyconnviz.styles import SCIENTIFIC_STYLE_KEYS, get_style


@pytest.mark.parametrize(
    ("name", "cortex_alpha", "depth_minimum"),
    [
        ("paper", 0.24, 0.20),
        ("soft", 0.20, 0.15),
        ("dark", 0.32, 0.28),
    ],
)
def test_builtin_styles_define_translucent_depth_context(
    name: str,
    cortex_alpha: float,
    depth_minimum: float,
) -> None:
    style = get_style(name)

    assert style["cortex_alpha"] == pytest.approx(cortex_alpha)
    assert style["depth_cue_min_alpha"] == pytest.approx(depth_minimum)


def test_builtin_styles_are_visual_only() -> None:
    for name in ("paper", "soft", "dark"):
        style = get_style(name)
        assert SCIENTIFIC_STYLE_KEYS.isdisjoint(style)
        assert "background" in style
        assert "node_size_range" in style
        assert "edge_width_range" in style
        assert style["default_overlay"] == "none"
        assert 0.0 <= style["cortex_alpha"] <= 1.0
        assert 0.0 <= style["depth_cue_min_alpha"] <= 1.0


def test_style_results_are_deep_copy_isolated() -> None:
    first = get_style("paper")
    first["node_size_range"][0] = 999
    second = get_style("paper")
    assert second["node_size_range"][0] != 999


def test_unknown_style_lists_valid_choices() -> None:
    with pytest.raises(ValueError, match=r"paper.*soft.*dark"):
        get_style("neon")
