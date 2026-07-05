"""
Overlay SLAM 主循环（≈ gmapping GridSlamProcessor::processScan）。

每帧激光固定顺序：
  1. estimate_pose   — 定位（odom 递推 + scan_match / AMCL TF）
  2. integrate_map   — 建图（log-odds，≈ updateMap）
  3. refresh_match   — 仅 sync 本帧 integrate 改动的格子到匹配快照

不含 ROS：由 map_overlay_node 提供 TF 查询回调。
"""

from __future__ import annotations

import math
import time
from typing import Optional

from map_overlay.core.localization import LocalizationEngine, LookupOdomLaserFn, LookupPoseFn
from map_overlay.core.map_grid import MapGrid
from map_overlay.core.scan_matcher import ScanMatcher
from map_overlay.core.types import OverlayConfig, Pose2D, ProcessScanResult, SensorScan


class OverlayProcessor:
    """地图覆盖 SLAM 算法核（无 rclpy）。"""

    def __init__(self, config: OverlayConfig) -> None:
        self.config = config
        self.grid = MapGrid(config)
        self.matcher = ScanMatcher(config, self.grid)
        self.localization = LocalizationEngine(config, self.matcher)

        self.mapping_active = False
        self.scan_count = 0
        self.update_count = 0
        self._last_integrate = Pose2D(0.0, 0.0, 0.0)
        self._last_integrate_set = False

    @property
    def map_ready(self) -> bool:
        return self.grid.ready

    def begin_mapping_scan_match(self) -> None:
        self.localization.start_scan_match_mode()

    def begin_mapping_frozen(self) -> None:
        self.localization.set_frozen_mode()

    def begin_mapping_amcl_passive(self) -> None:
        self.localization.active_loc_mode = self.config.loc_mode

    def start_mapping(self) -> None:
        self.mapping_active = True
        self.scan_count = 0
        self.update_count = 0
        self._last_integrate_set = False

    def stop_mapping(self) -> None:
        self.mapping_active = False
        self.localization.stop_scan_match_mode()

    def prepare_match_snapshot_large_map(self) -> None:
        self.grid.match_occ = None

    def refresh_match_snapshot(self) -> bool:
        return self.grid.refresh_match_snapshot()

    def _should_integrate(self, pose: Pose2D) -> bool:
        cfg = self.config
        if self.scan_count % cfg.integrate_every_n != 0:
            return False
        if not self._last_integrate_set:
            return True
        dx = pose.x - self._last_integrate.x
        dy = pose.y - self._last_integrate.y
        dyaw = math.atan2(
            math.sin(pose.yaw - self._last_integrate.yaw),
            math.cos(pose.yaw - self._last_integrate.yaw),
        )
        return (
            math.hypot(dx, dy) >= cfg.min_update_d
            or abs(dyaw) >= cfg.min_update_a
        )

    def process_scan(
        self,
        scan: SensorScan,
        lookup_map_laser: LookupPoseFn,
        lookup_odom_laser: LookupOdomLaserFn,
    ) -> ProcessScanResult:
        """
        SLAM 主循环入口（由 laserCallback 调用）。

        对应 gmapping：addScan → drawFromMotion/scanMatch → updateMap
        """
        result = ProcessScanResult(
            scan_count=self.scan_count,
            update_count=self.update_count,
        )
        if not self.mapping_active or not self.grid.ready:
            return result

        t0 = time.perf_counter()

        # --- Step 1: 定位 ---
        pose = self.localization.estimate_pose(
            scan, lookup_map_laser, lookup_odom_laser,
        )
        if pose is None:
            return result

        self.scan_count += 1
        result.scan_count = self.scan_count
        result.pose = pose
        result.accepted = True
        result.match_timing = self.localization.last_match_timing

        if not self._should_integrate(pose):
            result.update_count = self.update_count
            result.process_scan_ms = (time.perf_counter() - t0) * 1000.0
            return result

        # --- Step 2: 建图 (updateMap) ---
        integrate_result = self.grid.integrate(scan, pose)
        if integrate_result.changed:
            self.update_count += 1
            result.map_updated = True
            result.update_count = self.update_count
            self._last_integrate = pose
            self._last_integrate_set = True

            if self.localization.scan_match_active:
                self.matcher.invalidate_field_cache()
                self.grid.maybe_refresh_match(integrate_result.updated_indices)
                result.refresh_match_map = True

            if self.config.publish_on_integrate:
                if self.grid.is_large():
                    result.should_publish_map = self.update_count % 10 == 0
                else:
                    result.should_publish_map = True

        result.update_count = self.update_count
        result.process_scan_ms = (time.perf_counter() - t0) * 1000.0
        return result
