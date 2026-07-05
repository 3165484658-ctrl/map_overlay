"""OccupancyGrid + log-odds 更新（≈ gmapping updateMap）。"""

from __future__ import annotations

import math
import threading
from typing import FrozenSet, List, Optional, Set, Tuple

from nav_msgs.msg import OccupancyGrid

from map_overlay.core.types import IntegrateResult, OverlayConfig, Pose2D, SensorScan
from map_overlay.utils.geometry import transform_point
from map_overlay.utils.log_odds import bresenham, log_odds_to_occ, occ_to_log_odds


class MapGrid:
    def __init__(self, config: OverlayConfig) -> None:
        self._cfg = config
        self._lock = threading.RLock()
        self.grid: Optional[OccupancyGrid] = None
        self.log_odds: Optional[List[float]] = None
        self.match_occ: Optional[List[int]] = None
        self.match_refresh_counter = 0
        self._pending_match_indices: Optional[FrozenSet[int]] = None
        self._hit_counts: Optional[List[int]] = None
        self._miss_counts: Optional[List[int]] = None
        self.ready = False
        self.dirty = False

    def load_grid(self, grid: OccupancyGrid) -> None:
        with self._lock:
            self.grid = grid
            prior = self._cfg.unknown_prior_prob
            self.log_odds = [
                occ_to_log_odds(v, unknown_prior=prior) for v in grid.data
            ]
            cfg = self._cfg
            # 底图里的 occupied/free 栅格视为已经确认过，不需要重新积累
            self._hit_counts = [
                cfg.occ_confirm_hits if v >= 100 else 0 for v in grid.data
            ]
            self._miss_counts = [
                cfg.occ_confirm_misses if 0 <= v <= 10 else 0 for v in grid.data
            ]
            self.ready = True
            self.dirty = True

    def is_large(self) -> bool:
        if self.grid is None:
            return False
        info = self.grid.info
        return info.width * info.height >= self._cfg.large_map_threshold

    def refresh_match_snapshot(self) -> bool:
        """全图匹配快照；大地图返回 False 表示使用 live grid。"""
        if self.grid is None:
            return False
        if self.is_large():
            with self._lock:
                self.match_occ = None
            self.match_refresh_counter = 0
            self._pending_match_indices = None
            return False
        with self._lock:
            self.match_occ = list(self.grid.data)
        self.match_refresh_counter = 0
        self._pending_match_indices = None
        return True

    def match_occ_data(self) -> List[int]:
        assert self.grid is not None
        if self.match_occ is None:
            self.refresh_match_snapshot()
        if self.match_occ is not None:
            return self.match_occ
        return self.grid.data

    def sync_match_indices(self, indices: FrozenSet[int]) -> int:
        """仅把本帧 integrate 改过的栅格同步进匹配快照。"""
        if self.grid is None or self.match_occ is None or not indices:
            return 0
        live = self.grid.data
        match = self.match_occ
        for idx in indices:
            match[idx] = live[idx]
        return len(indices)

    def maybe_refresh_match(self, updated_indices: FrozenSet[int]) -> None:
        """将本帧 integrate 的改动延迟到下一帧再同步进匹配快照。

        这样当前帧激光刚建的障碍物不会立刻被 scan matcher 拿来匹配自己，
        避免自匹配（self-matching）导致定位漂移和地图快速变化。
        """
        if self._cfg.match_map_local_sync:
            # 先同步上一帧保存的改动
            if self._pending_match_indices:
                self.sync_match_indices(self._pending_match_indices)
            # 再把本帧改动存起来，下一帧才同步
            self._pending_match_indices = updated_indices
        every = self._cfg.match_map_full_refresh_every
        if every <= 0:
            return
        self.match_refresh_counter += 1
        if self.match_refresh_counter >= every:
            self.refresh_match_snapshot()

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        assert self.grid is not None
        res = self.grid.info.resolution
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y
        return (
            int(math.floor((wx - ox) / res)),
            int(math.floor((wy - oy) / res)),
        )

    def integrate(self, scan: SensorScan, pose: Pose2D) -> IntegrateResult:
        """Raycast log-odds 更新；栅格范围固定为底图 yaml 大小，越界射线丢弃。"""
        assert self.grid is not None and self.log_odds is not None
        cfg = self._cfg
        info = self.grid.info
        res = info.resolution
        width = info.width
        height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        lx, ly, lyaw = pose.x, pose.y, pose.yaw
        radius_sq = cfg.update_radius * cfg.update_radius
        updated: Set[int] = set()
        lo_free = -cfg.lo_miss if cfg.enable_clearing else 0.0

        def in_bounds(mx: int, my: int) -> bool:
            return 0 <= mx < width and 0 <= my < height

        def world_to_map(wx: float, wy: float) -> Tuple[int, int]:
            return (
                int(math.floor((wx - origin_x) / res)),
                int(math.floor((wy - origin_y) / res)),
            )

        def near_robot(wx: float, wy: float) -> bool:
            dx = wx - lx
            dy = wy - ly
            return dx * dx + dy * dy <= radius_sq

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and cfg.range_min < r < cfg.range_max:
                bx = r * math.cos(angle)
                by = r * math.sin(angle)
                ex, ey = transform_point(bx, by, lx, ly, lyaw)

                if not near_robot(ex, ey) and not near_robot(lx, ly):
                    angle += scan.angle_increment
                    continue

                x0, y0 = world_to_map(lx, ly)
                x1, y1 = world_to_map(ex, ey)
                if not (in_bounds(x0, y0) and in_bounds(x1, y1)):
                    angle += scan.angle_increment
                    continue

                line = bresenham(x0, y0, x1, y1)
                for mx, my in line[:-1]:
                    if not in_bounds(mx, my):
                        continue
                    wx = origin_x + (mx + 0.5) * res
                    wy = origin_y + (my + 0.5) * res
                    if lo_free != 0.0 and near_robot(wx, wy):
                        idx = mx + my * width
                        if self._apply_log_odds(idx, lo_free):
                            updated.add(idx)

                if near_robot(ex, ey):
                    idx = x1 + y1 * width
                    if self._apply_log_odds(idx, cfg.lo_hit):
                        updated.add(idx)

            angle += scan.angle_increment

        changed = bool(updated)
        if changed:
            self.dirty = True
        return IntegrateResult(changed=changed, updated_indices=frozenset(updated))

    def _apply_log_odds(self, idx: int, delta: float) -> bool:
        """返回 True 表示 live 栅格 occupancy 值发生变化。"""
        assert self.log_odds is not None and self.grid is not None
        cfg = self._cfg
        new_lo = min(max(self.log_odds[idx] + delta, cfg.lo_min), cfg.lo_max)
        if abs(new_lo - self.log_odds[idx]) < 1e-6:
            return False
        self.log_odds[idx] = new_lo

        raw_occ = log_odds_to_occ(
            new_lo,
            display_mode=cfg.occ_display_mode,
            occupied_thresh=cfg.occupied_thresh,
            free_thresh=cfg.free_thresh,
        )

        assert self._hit_counts is not None and self._miss_counts is not None
        is_hit = delta > 0
        is_free = delta < 0
        if is_hit:
            self._hit_counts[idx] += 1
            self._miss_counts[idx] = max(0, self._miss_counts[idx] - 1)
        elif is_free:
            self._miss_counts[idx] += 1
            self._hit_counts[idx] = max(0, self._hit_counts[idx] - 1)

        # 多帧确认：连续命中/空闲次数不足时不显示为纯 occupied/free
        if raw_occ == 100 and self._hit_counts[idx] < cfg.occ_confirm_hits:
            display_occ = 50
        elif raw_occ == 0 and self._miss_counts[idx] < cfg.occ_confirm_misses:
            display_occ = 50
        else:
            display_occ = raw_occ

        if self.grid.data[idx] == display_occ:
            return False
        self.grid.data[idx] = display_occ
        return True
