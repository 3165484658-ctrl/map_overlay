"""SLAM 核用的数据结构（无 ROS 依赖）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class MapOdomState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class SensorScan:
    """LaserScan 的算法层表示。"""

    frame_id: str
    stamp_ns: int
    angle_min: float
    angle_increment: float
    ranges: Tuple[float, ...]


@dataclass
class MatchTiming:
    field_ms: float = 0.0
    coarse_ms: float = 0.0
    fine_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0
    field_cached: bool = False


@dataclass
class MatchOutcome:
    lx: float
    ly: float
    yaw: float
    score: float
    timing: Optional['MatchTiming'] = None


@dataclass
class ProcessScanResult:
    """process_scan() 一帧的输出。"""

    accepted: bool = False
    pose: Optional[Pose2D] = None
    map_updated: bool = False
    should_publish_map: bool = False
    refresh_match_map: bool = False
    scan_count: int = 0
    update_count: int = 0
    match_timing: Optional[MatchTiming] = None
    process_scan_ms: float = 0.0


@dataclass
class IntegrateResult:
    changed: bool = False
    updated_indices: FrozenSet[int] = frozenset()


@dataclass
class OverlayConfig:
    """从 ROS 参数解析后的配置快照。"""

    map_frame: str = 'map'
    odom_frame: str = 'odom'
    loc_mode: str = 'amcl'
    use_scan_match_when_mapping: bool = True
    update_radius: float = 5.0
    enable_clearing: bool = True
    integrate_every_n: int = 3
    min_update_d: float = 0.10
    min_update_a: float = 0.10
    lo_hit: float = 0.40
    lo_miss: float = 0.40
    lo_min: float = -4.0
    lo_max: float = 4.0
    unknown_prior_prob: float = 0.35
    occ_display_mode: str = 'linear'
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    occ_confirm_hits: int = 3
    occ_confirm_misses: int = 2
    range_min: float = 0.15
    range_max: float = 12.0
    publish_on_integrate: bool = True
    large_map_threshold: int = 2_000_000
    match_search_xy: float = 0.35
    match_search_yaw: float = 0.20
    match_steps: int = 9
    match_beam_stride: int = 2
    match_max_beams: int = 60
    match_field_cache_cells: int = 3
    match_min_score: float = 0.25
    match_likelihood_sigma: float = 0.15
    match_likelihood_max_dist: float = 2.0
    match_field_radius: float = 0.0
    match_yaw_reach: float = 3.0
    match_fine_iters: int = 3
    match_fine_xy: float = 0.06
    match_fine_yaw: float = 0.04
    match_fine_steps: int = 5
    match_map_local_sync: bool = True
    match_map_full_refresh_every: int = 0
    scan_match_every_n: int = 4
    scan_match_trigger_d: float = 0.15
    scan_match_trigger_a: float = 0.12
    scan_match_log_timing: bool = True
    map_odom_smoothing: float = 0.35
    use_map_odom_pf: bool = True
    pf_num_particles: int = 80
    pf_resample_interval: int = 2
    pf_noise_xy: float = 0.015
    pf_noise_yaw: float = 0.012
    pf_trans_alpha: float = 0.15
    pf_rot_alpha: float = 0.10
    pf_init_xy: float = 0.08
    pf_init_yaw: float = 0.06
    pf_refine_xy: float = 0.04
    pf_refine_yaw: float = 0.03
    # 静止判定阈值：odom 变化小于此值时不做 scan_to_map 测量更新
    loc_update_min_d: float = 0.02
    loc_update_min_a: float = 0.02
    # likelihood 指数温度，越小越锐化
    pf_likelihood_temp: float = 0.15
    # scan-to-map 匹配时忽略的最小距离（过滤机体/地面/腿等近距离反射，缓解退化）
    scan_match_min_range: float = 0.80
    # 静止判定阈值：odom 变化小于此值时不做 scan_to_map 测量更新
    loc_update_min_d: float = 0.02
    loc_update_min_a: float = 0.02
    # likelihood 指数温度，越小越锐化
    pf_likelihood_temp: float = 0.15

    def match_field_reach_meters(self) -> float:
        """
        BFS 距离场 patch 半径（以机器人为中心）。

        与 AMCL 一致：likelihood 只在障碍附近 max_dist 内有意义，
        不应按 laser range_max 铺整张场。pose 搜索窗口单独加在 patch 上。
        """
        if self.match_field_radius > 0.0:
            return self.match_field_radius
        return (
            self.match_likelihood_max_dist
            + self.match_search_xy
            + self.match_search_yaw * self.match_yaw_reach
            + self.match_fine_xy * self.match_fine_iters
            + 0.5
        )

    def match_margin_cells(self, resolution: float) -> int:
        return int(math.ceil(self.match_field_reach_meters() / resolution)) + 2

    @classmethod
    def from_node_params(cls, node) -> 'OverlayConfig':
        return cls(
            map_frame=str(node.get_parameter('map_frame').value),
            odom_frame=str(node.get_parameter('odom_frame').value),
            loc_mode=str(node.get_parameter('localization_mode').value).lower(),
            use_scan_match_when_mapping=bool(
                node.get_parameter('use_scan_match_when_mapping').value,
            ),
            update_radius=float(node.get_parameter('update_radius').value),
            enable_clearing=bool(node.get_parameter('enable_clearing').value),
            integrate_every_n=max(1, int(node.get_parameter('integrate_every_n_scans').value)),
            min_update_d=float(node.get_parameter('min_update_distance').value),
            min_update_a=float(node.get_parameter('min_update_angle').value),
            lo_hit=float(node.get_parameter('log_odds_hit').value),
            lo_miss=float(node.get_parameter('log_odds_miss').value),
            lo_min=float(node.get_parameter('log_odds_min').value),
            lo_max=float(node.get_parameter('log_odds_max').value),
            unknown_prior_prob=float(node.get_parameter('unknown_prior_prob').value),
            occ_display_mode=str(node.get_parameter('map_occ_display_mode').value).lower(),
            occupied_thresh=float(node.get_parameter('occupied_thresh').value),
            free_thresh=float(node.get_parameter('free_thresh').value),
            occ_confirm_hits=max(0, int(node.get_parameter('occ_confirm_hits').value)),
            occ_confirm_misses=max(0, int(node.get_parameter('occ_confirm_misses').value)),
            range_min=float(node.get_parameter('scan_range_min').value),
            range_max=float(node.get_parameter('scan_range_max').value),
            publish_on_integrate=bool(node.get_parameter('publish_map_on_integrate').value),
            large_map_threshold=int(node.get_parameter('large_map_cell_threshold').value),
            match_search_xy=float(node.get_parameter('scan_match_search_xy').value),
            match_search_yaw=float(node.get_parameter('scan_match_search_yaw').value),
            match_steps=max(3, int(node.get_parameter('scan_match_steps').value)),
            match_beam_stride=max(1, int(node.get_parameter('scan_match_beam_stride').value)),
            match_max_beams=max(10, int(node.get_parameter('scan_match_max_beams').value)),
            match_field_cache_cells=max(
                0, int(node.get_parameter('scan_match_field_cache_cells').value),
            ),
            match_min_score=float(node.get_parameter('scan_match_min_score').value),
            match_likelihood_sigma=float(
                node.get_parameter('scan_match_likelihood_sigma').value,
            ),
            match_likelihood_max_dist=float(
                node.get_parameter('scan_match_likelihood_max_dist').value,
            ),
            match_field_radius=float(
                node.get_parameter('scan_match_field_radius').value,
            ),
            match_yaw_reach=float(node.get_parameter('scan_match_yaw_reach').value),
            match_fine_iters=max(
                0, int(node.get_parameter('scan_match_fine_iterations').value),
            ),
            match_fine_xy=float(node.get_parameter('scan_match_fine_xy').value),
            match_fine_yaw=float(node.get_parameter('scan_match_fine_yaw').value),
            match_fine_steps=max(3, int(node.get_parameter('scan_match_fine_steps').value)),
            match_map_local_sync=bool(node.get_parameter('match_map_local_sync').value),
            match_map_full_refresh_every=max(
                0, int(node.get_parameter('match_map_full_refresh_updates').value),
            ),
            scan_match_every_n=max(
                1, int(node.get_parameter('scan_match_every_n_scans').value),
            ),
            scan_match_trigger_d=float(node.get_parameter('scan_match_trigger_d').value),
            scan_match_trigger_a=float(node.get_parameter('scan_match_trigger_a').value),
            scan_match_log_timing=bool(
                node.get_parameter('scan_match_log_timing').value,
            ),
            map_odom_smoothing=float(node.get_parameter('map_odom_smoothing').value),
            use_map_odom_pf=bool(node.get_parameter('use_map_odom_pf').value),
            pf_num_particles=max(10, int(node.get_parameter('pf_num_particles').value)),
            pf_resample_interval=max(
                1, int(node.get_parameter('pf_resample_interval').value),
            ),
            pf_noise_xy=float(node.get_parameter('pf_noise_xy').value),
            pf_noise_yaw=float(node.get_parameter('pf_noise_yaw').value),
            pf_trans_alpha=float(node.get_parameter('pf_trans_alpha').value),
            pf_rot_alpha=float(node.get_parameter('pf_rot_alpha').value),
            pf_init_xy=float(node.get_parameter('pf_init_xy').value),
            pf_init_yaw=float(node.get_parameter('pf_init_yaw').value),
            pf_refine_xy=float(node.get_parameter('pf_refine_xy').value),
            pf_refine_yaw=float(node.get_parameter('pf_refine_yaw').value),
            loc_update_min_d=float(
                node.get_parameter('loc_update_min_d').value
            ),
            loc_update_min_a=float(
                node.get_parameter('loc_update_min_a').value
            ),
            pf_likelihood_temp=float(
                node.get_parameter('pf_likelihood_temp').value
            ),
            scan_match_min_range=float(node.get_parameter('scan_match_min_range').value),
        )


@dataclass
class TfFrozenSnapshot:
    """frozen 模式锁定的 map→odom。"""

    ox: float
    oy: float
    oyaw: float
