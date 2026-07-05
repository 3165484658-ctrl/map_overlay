"""Likelihood field 用的局部距离场（Felzenszwalb 精确欧氏距离变换）。

替换原来的 4 邻接 BFS（曼哈顿距离）：4 邻接把真实欧氏距离高估约 √2 倍，
导致 likelihood 被系统性压低 → 整体分数偏低 → 退化/低分频发 → 漂移。
Felzenszwalb 二维 EDT 是 O(n) 精确欧氏距离，与 AMCL/Hector 的距离场一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


# 无障碍种子时的“无穷远”标记：sqrt 后约 1e10，远超任何 max_dist，
# likelihood_from_distance 会钳到 max_dist 给出极小值；scan_matcher 用 >=1e8 判定回退。
_SQ_INF = 1e20


def likelihood_from_distance(dist_m: float, sigma: float, max_dist: float) -> float:
    d = min(max(dist_m, 0.0), max_dist)
    return math.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)


@dataclass
class MatchFieldContext:
    dist: List[float]
    min_mx: int
    min_my: int
    patch_w: int
    patch_h: int
    map_width: int
    occ_data: List[int]


def _edt_1d_sq(f: List[float]) -> List[float]:
    """Felzenszwalb 1D 平方距离变换。

    输入 f[i]：种子处 0、其余大值；输出 d[i] = min_j (f[j] + (i-j)^2)。
    用抛物线下包络求交点，O(n)。
    """
    n = len(f)
    if n == 0:
        return []
    d = [0.0] * n
    v = [0] * n          # 下包络里各抛物线的顶点位置
    z = [0.0] * (n + 1)  # 相邻抛物线交点
    k = 0
    v[0] = 0
    z[0] = -1e30
    z[1] = 1e30
    for q in range(1, n):
        # 抛物线 (q, f[q]) 与当前栈顶 (v[k], f[v[k]]) 的交点
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 1e30
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        d[q] = float(diff * diff) + f[v[k]]
    return d


def build_local_distance_field(
    occ_data: List[int],
    map_width: int,
    map_height: int,
    resolution: float,
    center_mx: int,
    center_my: int,
    margin_cells: int,
) -> Optional[MatchFieldContext]:
    min_mx = max(0, center_mx - margin_cells)
    max_mx = min(map_width - 1, center_mx + margin_cells)
    min_my = max(0, center_my - margin_cells)
    max_my = min(map_height - 1, center_my + margin_cells)
    patch_w = max_mx - min_mx + 1
    patch_h = max_my - min_my + 1
    if patch_w <= 0 or patch_h <= 0:
        return None

    # 障碍源阈值 50：与建图 display 解耦。新建障碍约 2 次命中（display≈55）即入源，
    # 不必等到底图级的 65（约需 4 次命中），消除新区行进的匹配盲区。
    THRESH = 50

    # 平方距离（以格为单位），种子=障碍格 0，其余 _SQ_INF
    sq = [_SQ_INF] * (patch_w * patch_h)
    for my in range(min_my, max_my + 1):
        row_base = my * map_width
        py = my - min_my
        row_off = py * patch_w
        for mx in range(min_mx, max_mx + 1):
            if occ_data[mx + row_base] >= THRESH:
                sq[(mx - min_mx) + row_off] = 0.0

    # 第一遍：沿 x（每行）做 1D EDT
    for py in range(patch_h):
        base = py * patch_w
        sq[base:base + patch_w] = _edt_1d_sq(sq[base:base + patch_w])
    # 第二遍：沿 y（每列）做 1D EDT，得到完整 2D 平方欧氏距离
    col_buf = [0.0] * patch_h
    for px in range(patch_w):
        for py in range(patch_h):
            col_buf[py] = sq[px + py * patch_w]
        d = _edt_1d_sq(col_buf)
        for py in range(patch_h):
            sq[px + py * patch_w] = d[py]

    # 转米
    dist = [0.0] * (patch_w * patch_h)
    for i in range(patch_w * patch_h):
        v = sq[i]
        dist[i] = math.sqrt(v) * resolution if v < _SQ_INF else 1e10

    return MatchFieldContext(
        dist=dist,
        min_mx=min_mx,
        min_my=min_my,
        patch_w=patch_w,
        patch_h=patch_h,
        map_width=map_width,
        occ_data=occ_data,
    )
