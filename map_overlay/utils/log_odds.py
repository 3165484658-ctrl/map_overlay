"""Log-odds 栅格与 Bresenham 射线。"""

from __future__ import annotations

import math
from typing import List, Tuple


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


def log_odds_to_prob(lo: float) -> float:
    return 1.0 - 1.0 / (1.0 + math.exp(lo))


def prob_to_log_odds(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def occ_to_log_odds(occ: int, *, unknown_prior: float = 0.5) -> float:
    if occ < 0:
        return prob_to_log_odds(unknown_prior)
    if occ >= 65:
        p = 0.9
    elif occ <= 10:
        p = 0.1
    else:
        p = occ / 100.0
    return prob_to_log_odds(p)


def log_odds_to_occ(
    lo: float,
    *,
    display_mode: str = 'linear',
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
) -> int:
    """log-odds → OccupancyGrid 0–100；-1 仅用于从未观测格，由调用方保留。"""
    p = log_odds_to_prob(lo)
    if display_mode == 'threshold':
        if p >= occupied_thresh:
            return 100
        if p <= free_thresh:
            return 0
    return int(round(min(max(p, 0.0), 1.0) * 100.0))
