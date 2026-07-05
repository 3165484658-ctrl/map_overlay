"""Scan-to-map 匹配（≈ gmapping scanMatch），numpy 向量化批量打分。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from map_overlay.core.map_grid import MapGrid
from map_overlay.core.types import MatchOutcome, MatchTiming, OverlayConfig, SensorScan
from map_overlay.utils.distance_field import (
    MatchFieldContext,
    build_local_distance_field,
    likelihood_from_distance,
)
from map_overlay.utils.geometry import normalize_angle


def scoring_beam_indices(scan: SensorScan, stride: int, max_beams: int) -> Tuple[int, ...]:
    raw = list(range(0, len(scan.ranges), max(1, stride)))
    if not raw:
        return ()
    if len(raw) <= max_beams:
        return tuple(raw)
    step = max(1, len(raw) // max_beams)
    picked = raw[::step]
    if len(picked) > max_beams:
        picked = picked[:max_beams]
    return tuple(picked)


@dataclass(frozen=True)
class BeamCache:
    """激光端点在 laser 系下的 (bx, by)，避免每 pose 重复 sin/cos(angle)。"""
    beams: Tuple[Tuple[float, float], ...]


class ScanMatcher:
    def __init__(self, config: OverlayConfig, grid: MapGrid) -> None:
        self._cfg = config
        self._grid = grid
        self._field_cache: Optional[MatchFieldContext] = None
        self._field_cache_cmx = -10**9
        self._field_cache_cmy = -10**9

    def invalidate_field_cache(self) -> None:
        self._field_cache = None

    @staticmethod
    def _occ_match_score(occ: int) -> float:
        if occ < 0:
            return 0.5
        return min(max(occ / 100.0, 0.0), 1.0)

    @staticmethod
    def _search_offsets(span: float, steps: int) -> List[float]:
        if steps <= 1:
            return [0.0]
        return [span * (2.0 * i / (steps - 1) - 1.0) for i in range(steps)]

    def build_beam_cache(
        self, scan: SensorScan, beam_indices: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Tuple[int, ...], BeamCache]:
        cfg = self._cfg
        if beam_indices is None:
            beam_indices = scoring_beam_indices(
                scan, cfg.match_beam_stride, cfg.match_max_beams,
            )
        beams: List[Tuple[float, float]] = []
        inc = scan.angle_increment
        min_range = getattr(cfg, 'scan_match_min_range', cfg.range_min)
        for i in beam_indices:
            r = scan.ranges[i]
            angle = scan.angle_min + i * inc
            if math.isfinite(r) and min_range < r < cfg.range_max:
                beams.append((math.cos(angle) * r, math.sin(angle) * r))
        return beam_indices, BeamCache(tuple(beams))

    def _match_margin_cells(self, scan: SensorScan) -> int:
        assert self._grid.grid is not None
        res = self._grid.grid.info.resolution
        return self._cfg.match_margin_cells(res)

    def prepare_match_field(
        self, px: float, py: float, scan: SensorScan,
    ) -> Tuple[Optional[MatchFieldContext], bool]:
        assert self._grid.grid is not None
        cmx, cmy = self._grid.world_to_map(px, py)
        tol = self._cfg.match_field_cache_cells
        if (
            self._field_cache is not None
            and tol > 0
            and abs(cmx - self._field_cache_cmx) <= tol
            and abs(cmy - self._field_cache_cmy) <= tol
        ):
            return self._field_cache, True

        margin = self._match_margin_cells(scan)
        occ = self._grid.match_occ_data()
        info = self._grid.grid.info
        ctx = build_local_distance_field(
            occ, info.width, info.height, info.resolution, cmx, cmy, margin,
        )
        if ctx is not None and tol > 0:
            self._field_cache = ctx
            self._field_cache_cmx = cmx
            self._field_cache_cmy = cmy
        else:
            self._field_cache = None
        return ctx, False

    # ------------------------------------------------------------------ scalar scorer

    def score_beams_at_pose(
        self,
        lx: float,
        ly: float,
        lyaw: float,
        beam_cache: BeamCache,
        ctx: Optional[MatchFieldContext] = None,
    ) -> float:
        """精确标量打分（用于最终阈值判定和无 ctx 回退）。"""
        assert self._grid.grid is not None
        if not beam_cache.beams:
            return 0.0
        cfg = self._cfg
        width = self._grid.grid.info.width
        height = self._grid.grid.info.height
        occ_data = ctx.occ_data if ctx is not None else self._grid.match_occ_data()
        use_likelihood = ctx is not None
        cosy = math.cos(lyaw)
        siny = math.sin(lyaw)
        score = 0.0
        count = 0
        for bx, by in beam_cache.beams:
            ex = lx + cosy * bx - siny * by
            ey = ly + siny * bx + cosy * by
            mx, my = self._grid.world_to_map(ex, ey)
            if not (0 <= mx < width and 0 <= my < height):
                count += 1
                continue
            if use_likelihood and ctx is not None:
                if (
                    mx < ctx.min_mx or mx >= ctx.min_mx + ctx.patch_w
                    or my < ctx.min_my or my >= ctx.min_my + ctx.patch_h
                ):
                    beam_score = 0.02
                else:
                    patch_idx = (mx - ctx.min_mx) + (my - ctx.min_my) * ctx.patch_w
                    dist_m = ctx.dist[patch_idx]
                    if dist_m >= 1e8:
                        # patch 内无障碍源（未知区）：不给奖励，避免偏移位姿靠未知区刷分
                        beam_score = 0.0
                    else:
                        beam_score = likelihood_from_distance(
                            dist_m,
                            cfg.match_likelihood_sigma,
                            cfg.match_likelihood_max_dist,
                        )
                score += beam_score
            else:
                score += self._occ_match_score(occ_data[mx + my * width])
            count += 1
        return score / max(count, 1)

    # ------------------------------------------------------------------ numpy batch scorer

    def _score_batch(
        self,
        cands: np.ndarray,   # (M, 3) [x, y, yaw]
        beams: np.ndarray,   # (N, 2) [bx, by]
        ctx: MatchFieldContext,
    ) -> np.ndarray:
        """numpy 向量化批量打分。返回 (M,) 分数。

        与标量 score_beams_at_pose 保持一致：
        - in-bounds 且 in-patch 且有障碍源：likelihood
        - in-bounds 但 out-of-patch：0.05（与标量一致，避免系统性低估偏移位姿）
        - out-of-bounds：0.0（惩罚把激光甩出地图的位姿）
        分母统一为 N（总 beam 数），与标量 count 一致。
        """
        M = cands.shape[0]
        N = beams.shape[0]
        if N == 0 or M == 0:
            return np.zeros(M, dtype=np.float64)

        info = self._grid.grid.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        width = info.width
        height = info.height

        cx = cands[:, 0]
        cy = cands[:, 1]
        cyaw = cands[:, 2]
        cos_y = np.cos(cyaw)
        sin_y = np.sin(cyaw)

        bx = beams[:, 0]
        by = beams[:, 1]

        # (M, N) 端点坐标
        ex = cx[:, None] + cos_y[:, None] * bx[None, :] - sin_y[:, None] * by[None, :]
        ey = cy[:, None] + sin_y[:, None] * bx[None, :] + cos_y[:, None] * by[None, :]

        mx = np.floor((ex - ox) / res).astype(np.int32)
        my = np.floor((ey - oy) / res).astype(np.int32)

        in_bounds = (mx >= 0) & (mx < width) & (my >= 0) & (my < height)
        in_patch = (
            (mx >= ctx.min_mx)
            & (mx < ctx.min_mx + ctx.patch_w)
            & (my >= ctx.min_my)
            & (my < ctx.min_my + ctx.patch_h)
        )

        mx_c = np.clip(mx, ctx.min_mx, ctx.min_mx + ctx.patch_w - 1)
        my_c = np.clip(my, ctx.min_my, ctx.min_my + ctx.patch_h - 1)
        patch_idx = (mx_c - ctx.min_mx) + (my_c - ctx.min_my) * ctx.patch_w

        dist_np = np.asarray(ctx.dist, dtype=np.float64)
        dist_m = dist_np[patch_idx]

        sigma = self._cfg.match_likelihood_sigma
        max_dist = self._cfg.match_likelihood_max_dist
        d_clamped = np.minimum(dist_m, max_dist)
        likelihood = np.exp(-0.5 * (d_clamped / max(sigma, 1e-6)) ** 2)

        has_obstacle = dist_m < 1e8
        # in-patch 且有障碍源：likelihood；
        # patch 内无障碍源(未知区)：0（不奖励偏移位姿刷未知区分）
        # out-of-patch/out-of-bounds：0（惩罚偏移位姿）
        beam_scores = np.where(
            in_patch & has_obstacle, likelihood,
            0.0,
        )

        return beam_scores.sum(axis=1) / N

    def score_map_odom_particles(
        self,
        odom_x: float,
        odom_y: float,
        odom_yaw: float,
        particles: Sequence[Tuple[float, float, float]],
        scan: SensorScan,
        beam_cache: BeamCache,
        ctx: Optional[MatchFieldContext],
    ) -> List[float]:
        """批量给 map→odom 粒子打分（likelihood field，numpy 向量化）。"""
        from map_overlay.utils.geometry import map_odom_to_laser

        if not particles:
            return []
        if ctx is None or not beam_cache.beams:
            scores: List[float] = []
            for mx, my, myaw in particles:
                lx, ly, lyaw = map_odom_to_laser(odom_x, odom_y, odom_yaw, mx, my, myaw)
                scores.append(self.score_beams_at_pose(lx, ly, lyaw, beam_cache, ctx))
            return scores

        cands_list = []
        for mx, my, myaw in particles:
            lx, ly, lyaw = map_odom_to_laser(odom_x, odom_y, odom_yaw, mx, my, myaw)
            cands_list.append((lx, ly, lyaw))
        cands_np = np.array(cands_list, dtype=np.float64)
        beams_np = np.array(beam_cache.beams, dtype=np.float64)
        return self._score_batch(cands_np, beams_np, ctx).tolist()

    # ------------------------------------------------------------------ fine search

    def _fine_match_hill_climb_batch(
        self,
        best: Tuple[float, float, float],
        beams_np: np.ndarray,
        ctx: MatchFieldContext,
        initial_score: float,
    ) -> Tuple[Tuple[float, float, float], float]:
        """gmapping optimize 式多分辨率爬山（numpy 批量打分，每步 10 候选一次算）。"""
        cfg = self._cfg
        best_score = initial_score
        if cfg.match_fine_iters <= 0:
            return best, best_score

        n_levels = max(1, cfg.match_fine_steps)
        base_xy = cfg.match_fine_xy
        base_yaw = cfg.match_fine_yaw
        for level in range(n_levels):
            f = 1.0 / (1 << level)
            xy_s = base_xy * f
            yaw_s = base_yaw * f
            for _ in range(cfg.match_fine_iters):
                cands = np.array([
                    (best[0] + dx, best[1] + dy, best[2] + dyaw)
                    for dx, dy, dyaw in (
                        (xy_s, 0.0, 0.0), (-xy_s, 0.0, 0.0),
                        (0.0, xy_s, 0.0), (0.0, -xy_s, 0.0),
                        (xy_s, xy_s, 0.0), (-xy_s, -xy_s, 0.0),
                        (xy_s, -xy_s, 0.0), (-xy_s, xy_s, 0.0),
                        (0.0, 0.0, yaw_s), (0.0, 0.0, -yaw_s),
                    )
                ], dtype=np.float64)
                scores = self._score_batch(cands, beams_np, ctx)
                max_idx = int(np.argmax(scores))
                if scores[max_idx] > best_score:
                    best_score = float(scores[max_idx])
                    best = (
                        float(cands[max_idx, 0]),
                        float(cands[max_idx, 1]),
                        normalize_angle(float(cands[max_idx, 2])),
                    )
                else:
                    break
        return best, best_score

    # ------------------------------------------------------------------ main match

    def match(
        self, px: float, py: float, pyaw: float, scan: SensorScan,
    ) -> MatchOutcome:
        """粗搜 grid（numpy 批量）+ 细搜 hill-climb（numpy 批量）+ 精确重打。"""
        t0 = time.perf_counter()
        cfg = self._cfg
        log_timing = cfg.scan_match_log_timing
        beam_indices, beam_cache = self.build_beam_cache(scan)

        t_field0 = time.perf_counter()
        ctx, field_cached = self.prepare_match_field(px, py, scan)
        field_ms = (time.perf_counter() - t_field0) * 1000.0

        if ctx is None:
            score = self.score_beams_at_pose(px, py, pyaw, beam_cache)
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = MatchTiming(
                field_ms=field_ms, total_ms=total_ms, field_cached=field_cached,
            )
            return MatchOutcome(px, py, pyaw, score, timing if log_timing else None)

        beams_np = (
            np.array(beam_cache.beams, dtype=np.float64)
            if beam_cache.beams else np.empty((0, 2), dtype=np.float64)
        )

        # --- 粗搜：一次性批量打分所有候选（含初始位姿）---
        t_coarse0 = time.perf_counter()
        yaw_vals = self._search_offsets(cfg.match_search_yaw, cfg.match_steps)
        xy_vals = self._search_offsets(cfg.match_search_xy, cfg.match_steps)
        cands_list: List[Tuple[float, float, float]] = [(px, py, pyaw)]
        for dyaw in yaw_vals:
            for dx in xy_vals:
                for dy in xy_vals:
                    if dx == 0.0 and dy == 0.0 and dyaw == 0.0:
                        continue
                    cands_list.append((px + dx, py + dy, pyaw + dyaw))
        cands_np = np.array(cands_list, dtype=np.float64)

        batch_scores = self._score_batch(cands_np, beams_np, ctx)
        best_idx = int(np.argmax(batch_scores))
        best = (
            float(cands_np[best_idx, 0]),
            float(cands_np[best_idx, 1]),
            normalize_angle(float(cands_np[best_idx, 2])),
        )
        best_score_batch = float(batch_scores[best_idx])
        coarse_ms = (time.perf_counter() - t_coarse0) * 1000.0

        # --- 细搜：多分辨率爬山，每步 10 候选批量打分 ---
        t_fine0 = time.perf_counter()
        best, best_score_batch = self._fine_match_hill_climb_batch(
            best, beams_np, ctx, best_score_batch,
        )
        fine_ms = (time.perf_counter() - t_fine0) * 1000.0

        # --- 精确重打：用标量 scorer 得到准确分数用于阈值判定 ---
        score = self.score_beams_at_pose(best[0], best[1], best[2], beam_cache, ctx)

        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = MatchTiming(
            field_ms=field_ms,
            coarse_ms=coarse_ms,
            fine_ms=fine_ms,
            search_ms=coarse_ms + fine_ms,
            total_ms=total_ms,
            field_cached=field_cached,
        )
        return MatchOutcome(best[0], best[1], best[2], score, timing if log_timing else None)
