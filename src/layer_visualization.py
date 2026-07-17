from __future__ import annotations

from typing import Iterator


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    hlen = len(hex_color)
    return tuple(int(hex_color[i : i + hlen // 3], 16) / 255.0 for i in range(0, hlen, hlen // 3))


# Keep the same palette as old/multi0820.py
HEX_LAYER_COLORS = ["00FF80", "00FFFF", "007FFF", "0000FF", "7F00FF"]
RGB_LAYER_COLORS = [hex_to_rgb(c) for c in HEX_LAYER_COLORS]


def layer_color(layer_idx: int) -> tuple[float, float, float]:
    if not RGB_LAYER_COLORS:
        return (0.0, 0.0, 1.0)
    return RGB_LAYER_COLORS[layer_idx % len(RGB_LAYER_COLORS)]


def iter_layered_quadrics(
    list_quadrics: list,
    per_layer_groups: dict[int, list],
    max_layer: int | None = None,
) -> Iterator[tuple[int, object]]:
    idx = 0
    keys = sorted(int(k) for k in per_layer_groups.keys())
    for layer in keys:
        if max_layer is not None and layer > max_layer:
            break
        group_count = len(per_layer_groups[layer])
        for _ in range(group_count):
            if idx >= len(list_quadrics):
                return
            yield layer, list_quadrics[idx]
            idx += 1

    # Safety fallback if layer counts and quadric counts diverge.
    fallback_layer = keys[-1] if keys else 0
    while idx < len(list_quadrics):
        yield fallback_layer, list_quadrics[idx]
        idx += 1
