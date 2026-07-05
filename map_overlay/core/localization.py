"""定位：map→odom 估计 + 位姿查询（≈ drawFromMotion + scanMatch 的定位部分）。"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

from map_overlay.core.map_odom_pf import MapOdomParticleFilter
from map_overlay.core.scan_matcher import ScanMatcher
from map_overlay.core.types import (
    MapOdomState,
    MatchTiming,
    OverlayConfig,
    Pose2D,
    SensorScan,
    TfFrozenSnapshot,
)
from map_overlay.utils.geometry import (
    laser_to_map_odom,
    map_odom_to_laser,
    normalize_angle,
    transform_point,
)


LookupPoseFn = Callable[[str, int], Optional[Pose2D]]
LookupOdomLaserFn = Callable[[str, int], Optional[Pose2D]]


class LocalizationEngine:
    def __init__(self, config: OverlayConfig, matcher: ScanMatcher) -> None:
        self._cfg = config
        self._matcher = matcher
        self.map_odom = MapOdomState()
        self.base_loc_mode = config.loc_mode
        self.active_loc_mode = config.loc_mode
        self.scan_match_active = False
        self.frozen_snapshot: Optional[TfFrozenSnapshot] = None
        self.scan_match_counter = 0
        self.anchor_map_set = False
        self.anchor_map_laser = (0.0, 0.0, 0.0)
        self.last_match_timing = None
        self._pf = MapOdomParticleFilter(config)
        self._last_odom: Optional[Pose2D] = None
        self._last_match_odom: Optional[Pose2D] = None

    def lock_map_odom(self, x: float, y: float, yaw: float) -> None:
        self.map_odom.x = x
        self.map_odom.y = y
        self.map_odom.yaw = yaw
        self.frozen_snapshot = TfFrozenSnapshot(x, y, yaw)
        if self._cfg.use_map_odom_pf:
            self._pf.initialize(x, y, yaw)

    def reset_scan_match_session(self) -> None:
        self.anchor_map_set = False
        self.scan_match_counter = 0
        self._last_odom = None
        self._last_match_odom = None
        self._pf.reset()

    def start_scan_match_mode(self) -> None:
        self.scan_match_active = True
        self.active_loc_mode = 'scan_match'
        self.reset_scan_match_session()

    def stop_scan_match_mode(self) -> None:
        self.scan_match_active = False
        self.active_loc_mode = self.base_loc_mode
        self.frozen_snapshot = None
        self._pf.reset()
        self._last_odom = None
        self._last_match_odom = None

    def set_frozen_mode(self) -> None:
        self.active_loc_mode = 'frozen'

    def _odom_delta(self, odom_laser: Pose2D) -> Tuple[float, float]:
        if self._last_odom is None:
            return 0.0, 0.0
        dx = odom_laser.x - self._last_odom.x
        dy = odom_laser.y - self._last_odom.y
        da = normalize_angle(odom_laser.yaw - self._last_odom.yaw)
        return math.hypot(dx, dy), abs(da)

    def _should_run_full_match(self, odom_laser: Pose2D) -> bool:
        cfg = self._cfg
        if not self.anchor_map_set and not self._pf.ready:
            return True
        if self.scan_match_counter <= 1:
            return True
        if self.scan_match_counter % cfg.scan_match_every_n == 0:
            return True
        if self._last_match_odom is None:
            return True
        dx = odom_laser.x - self._last_match_odom.x
        dy = odom_laser.y - self._last_match_odom.y
        da = normalize_angle(odom_laser.yaw - self._last_match_odom.yaw)
        if math.hypot(dx, dy) >= cfg.scan_match_trigger_d:
            return True
        if abs(da) >= cfg.scan_match_trigger_a:
            return True
        return False

    def _scan_degeneracy(self, scan: SensorScan, pose: Pose2D) -> Tuple[bool, float, float]:
        """
        检测激光点云是否接近一维分布（如正对长墙、走廊）。
        返回: (是否退化, 主方向角度, 小特征值/大特征值)
        """
        cfg = self._cfg
        # 退化检测只用 scan matcher 实际参与匹配的点，保持一致
        min_range = getattr(cfg, 'scan_match_min_range', cfg.range_min)
        points: List[Tuple[float, float]] = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and min_range < r < cfg.range_max:
                bx = r * math.cos(angle)
                by = r * math.sin(angle)
                ex, ey = transform_point(bx, by, pose.x, pose.y, pose.yaw)
                points.append((ex, ey))
            angle += scan.angle_increment

        n = len(points)
        if n < 10:
            return False, 0.0, 1.0

        mean_x = sum(p[0] for p in points) / n
        mean_y = sum(p[1] for p in points) / n
        cxx = sum((p[0] - mean_x) ** 2 for p in points) / n
        cyy = sum((p[1] - mean_y) ** 2 for p in points) / n
        cxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / n

        trace = cxx + cyy
        det = cxx * cyy - cxy * cxy
        det = max(det, 0.0)
        tmp = math.sqrt(max(trace * trace - 4.0 * det, 0.0))
        eig1 = (trace + tmp) / 2.0
        eig2 = (trace - tmp) / 2.0
        eig1 = max(eig1, 1e-12)
        eig2 = max(eig2, 1e-12)
        ratio = eig2 / eig1

        # 主方向：方差最大的方向
        if abs(cxy) < 1e-9:
            main_angle = 0.0 if cxx >= cyy else math.pi / 2.0
        else:
            main_angle = math.atan2(eig1 - cxx, cxy)

        # 阈值 0.10：真实走廊有门框/转角时点云并非完美一维，ratio 常在 0.05~0.3，
        # 0.05 过严导致轻度退化不触发切向抑制，scan_match 在切向平坦面选错方向而滑移。
        # 0.10 让轻度退化也进入保护路径（切向 gain 已从 0.02 提到 0.10）。
        return ratio < 0.10, main_angle, ratio

    def _constrain_tangent_correction(
        self, nx: float, ny: float, main_angle: float, tangent_gain: float = 0.02,
    ) -> Tuple[float, float]:
        """
        退化场景中，只强烈修正法向（主方向），大幅抑制切向（垂直主方向）修正。

        Args:
            nx (float): 目标修正位置的 x 坐标
            ny (float): 目标修正位置的 y 坐标
            main_angle (float): 主方向（法向）的角度，单位为弧度
            tangent_gain (float): 切向修正保留比例，默认 0.02，值越小切向抑制越强

        Returns:
            Tuple[float, float]: 经切向抑制后的修正位置 (x, y)
        """
        dx = nx - self.map_odom.x
        dy = ny - self.map_odom.y
        # 切向方向 = 主方向 + 90°
        tcx = math.cos(main_angle + math.pi / 2.0)
        tcy = math.sin(main_angle + math.pi / 2.0)
        tangent = dx * tcx + dy * tcy
        # 从总修正中去掉多余的切向分量，只保留 tangent_gain 比例
        nx2 = self.map_odom.x + dx - tangent * tcx * (1.0 - tangent_gain)
        ny2 = self.map_odom.y + dy - tangent * tcy * (1.0 - tangent_gain)
        return nx2, ny2

    def _apply_map_odom_smooth(self, nx: float, ny: float, nyaw: float, score: float,
                                is_degenerate: bool = False) -> None:
        # 按匹配质量选择融合系数，退化场景额外压低
        if is_degenerate:
            # 退化场景（走廊/对墙）：即使分数高也只能保守修正
            if score >= 0.55:
                alpha = 0.12
            else:
                alpha = self._cfg.map_odom_smoothing * 0.4
        elif score >= 0.65:
            # 非退化 + 很高置信度才较快收敛（但不超过 0.30，防单帧误匹配大跳）
            alpha = 0.30
        elif score >= 0.45:
            alpha = max(self._cfg.map_odom_smoothing, 0.15)
        else:
            alpha = self._cfg.map_odom_smoothing * 0.4
        # 保留一个保守上限
        alpha = min(0.30, alpha)
        dyaw = normalize_angle(nyaw - self.map_odom.yaw)
        self.map_odom.x = alpha * nx + (1.0 - alpha) * self.map_odom.x
        self.map_odom.y = alpha * ny + (1.0 - alpha) * self.map_odom.y
        self.map_odom.yaw = normalize_angle(self.map_odom.yaw + alpha * dyaw)




    def _sync_map_odom_from_pf(self, odom_laser: Pose2D, score: float = 0.5,
                                 is_degenerate: bool = False) -> Pose2D:
        nx, ny, nyaw = self._pf.estimate()
        self._apply_map_odom_smooth(nx, ny, nyaw, score, is_degenerate)
        lx, ly, lyaw = map_odom_to_laser(
            odom_laser.x, odom_laser.y, odom_laser.yaw,
            self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
        )
        return Pose2D(lx, ly, lyaw)

    def _pf_update(
        self,
        scan: SensorScan,
        odom_laser: Pose2D,
        run_full_match: bool,
    ) -> Pose2D:
        cfg = self._cfg
        ox, oy, oyaw = odom_laser.x, odom_laser.y, odom_laser.yaw
        delta_d, delta_a = self._odom_delta(odom_laser)

        if not self._pf.ready:
            self._pf.initialize(self.map_odom.x, self.map_odom.y, self.map_odom.yaw)

        self._pf.predict(delta_d, delta_a)

        est_lx, est_ly, est_lyaw = map_odom_to_laser(
            ox, oy, oyaw, self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
        )
        _, beam_cache = self._matcher.build_beam_cache(scan)
        ctx, _ = self._matcher.prepare_match_field(est_lx, est_ly, scan)
        scores = self._matcher.score_map_odom_particles(
            ox, oy, oyaw, self._pf.particle_map_odoms(), scan, beam_cache, ctx,
        )
        self._pf.set_weights(scores)
        self._pf.maybe_resample()

        match_score = max(scores) if scores else 0.0

        if run_full_match:
            est = self._pf.estimate_laser_pose(ox, oy, oyaw)
            outcome = self._matcher.match(est.x, est.y, est.yaw, scan)
            self.last_match_timing = outcome.timing
            is_degen, main_angle, _ = self._scan_degeneracy(
                scan, Pose2D(outcome.lx, outcome.ly, outcome.yaw),
            )
            # ── 跳变检测：scan_match 结果与当前 estimate 偏差过大则拒绝 ──
            # 防止"每次同一位置误匹配到同一个错误位姿"
            jump_d = math.hypot(outcome.lx - est.x, outcome.ly - est.y)
            jump_a = abs(normalize_angle(outcome.yaw - est.yaw))
            MAX_JUMP_D = 0.30   # 30cm，超过视为可疑
            MAX_JUMP_A = 0.25   # 14°，超过视为可疑
            if jump_d > MAX_JUMP_D or jump_a > MAX_JUMP_A:
                print(
                    f'[map_overlay] scan_match 跳变过大 '
                    f'(d={jump_d:.2f}m a={math.degrees(jump_a):.1f}° '
                    f'score={outcome.score:.2f})，拒绝采纳，保守递推',
                    flush=True)
                # 跳变时 PF 已被坏数据加权污染，重置回当前 map_odom 防持续偏移
                self._pf.initialize(
                    self.map_odom.x, self.map_odom.y, self.map_odom.yaw)
                # 保守更新（低 alpha），不覆盖 anchor
                self._apply_map_odom_smooth(
                    self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
                    0.0, is_degen)
                self._last_match_odom = Pose2D(ox, oy, oyaw)
                lx, ly, lyaw = map_odom_to_laser(
                    ox, oy, oyaw, self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
                )
                return Pose2D(lx, ly, lyaw)

            if outcome.score >= cfg.match_min_score:
                # 高分：scan_match 结果收紧 PF + 平滑更新 map_odom
                mo_x, mo_y, mo_yaw = laser_to_map_odom(
                    outcome.lx, outcome.ly, outcome.yaw, ox, oy, oyaw,
                )
                if is_degen:
                    mo_x, mo_y = self._constrain_tangent_correction(mo_x, mo_y, main_angle)
                self._pf.refine(mo_x, mo_y, mo_yaw)
                self._apply_map_odom_smooth(mo_x, mo_y, mo_yaw, outcome.score, is_degen)
                self.anchor_map_set = True
                self.anchor_map_laser = (outcome.lx, outcome.ly, outcome.yaw)
                self._last_match_odom = Pose2D(ox, oy, oyaw)
                return Pose2D(outcome.lx, outcome.ly, outcome.yaw)
            # 低分：不再丢弃 scan_match / 退化为纯 odom 递推。
            # 用 PF estimate 做保守平滑更新，保留 likelihood field 对粒子的加权信息，
            # 避免退化场景下 scan_match 一旦低分就完全靠 odom 递推而持续漂移。
            self._last_match_odom = Pose2D(ox, oy, oyaw)
            est_mo_x, est_mo_y, est_mo_yaw = self._pf.estimate()
            if is_degen:
                est_mo_x, est_mo_y = self._constrain_tangent_correction(
                    est_mo_x, est_mo_y, main_angle, tangent_gain=0.10,
                )
            self._apply_map_odom_smooth(est_mo_x, est_mo_y, est_mo_yaw, outcome.score, is_degen)
            lx, ly, lyaw = map_odom_to_laser(
                ox, oy, oyaw, self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
            )
            return Pose2D(lx, ly, lyaw)

        # 中间帧：用 PF estimate 保守平滑更新 map_odom，不再纯 anchor+odom 覆盖。
        # PF 已做 predict（odom 增量）+ likelihood 加权，estimate 即「odom 递推 + 测量修正」，
        # 比直接 anchor+odom 覆盖更能抑制 odom 漂移。
        self.last_match_timing = None
        est_mo_x, est_mo_y, est_mo_yaw = self._pf.estimate()
        is_degen, main_angle, _ = self._scan_degeneracy(
            scan, Pose2D(est_lx, est_ly, est_lyaw),
        )
        if is_degen:
            est_mo_x, est_mo_y = self._constrain_tangent_correction(
                est_mo_x, est_mo_y, main_angle, tangent_gain=0.10,
            )
        self._apply_map_odom_smooth(est_mo_x, est_mo_y, est_mo_yaw, match_score, is_degen)
        lx, ly, lyaw = map_odom_to_laser(
            ox, oy, oyaw, self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
        )
        return Pose2D(lx, ly, lyaw)

    def _scan_match_pose_legacy(
        self,
        scan: SensorScan,
        odom_laser: Pose2D,
        run_full_match: bool,
    ) -> Pose2D:
        ox, oy, oyaw = odom_laser.x, odom_laser.y, odom_laser.yaw
        mo = self.map_odom

        if run_full_match:
            px, py, pyaw = map_odom_to_laser(ox, oy, oyaw, mo.x, mo.y, mo.yaw)
            outcome = self._matcher.match(px, py, pyaw, scan)
            self.last_match_timing = outcome.timing
            if outcome.score >= self._cfg.match_min_score:
                nx, ny, nyaw = laser_to_map_odom(
                    outcome.lx, outcome.ly, outcome.yaw, ox, oy, oyaw,
                )
                is_degen, main_angle, _ = self._scan_degeneracy(
                    scan, Pose2D(outcome.lx, outcome.ly, outcome.yaw),
                )
                if is_degen:
                    nx, ny = self._constrain_tangent_correction(nx, ny, main_angle)
                self._apply_map_odom_smooth(nx, ny, nyaw, outcome.score, is_degen)
                self.anchor_map_laser = (outcome.lx, outcome.ly, outcome.yaw)
                self.anchor_map_set = True
                self._last_match_odom = Pose2D(ox, oy, oyaw)
                return Pose2D(outcome.lx, outcome.ly, outcome.yaw)
        else:
            self.last_match_timing = None

        if self.anchor_map_set:
            ax, ay, ayaw = self.anchor_map_laser
            nx, ny, nyaw = laser_to_map_odom(ax, ay, ayaw, ox, oy, oyaw)
            mo.x, mo.y, mo.yaw = nx, ny, nyaw
            lx, ly, lyaw = map_odom_to_laser(ox, oy, oyaw, mo.x, mo.y, mo.yaw)
            return Pose2D(lx, ly, lyaw)

        lx, ly, lyaw = map_odom_to_laser(ox, oy, oyaw, mo.x, mo.y, mo.yaw)
        return Pose2D(lx, ly, lyaw)

    def _scan_match_pose(
        self,
        scan: SensorScan,
        odom_laser: Pose2D,
    ) -> Pose2D:
        cfg = self._cfg
        ox, oy, oyaw = odom_laser.x, odom_laser.y, odom_laser.yaw

        # AMCL 式静止判定：odom 变化足够小时跳过本次测量更新，避免原地反复修正导致漂移
        if self._last_odom is not None and self.anchor_map_set:
            delta_d, delta_a = self._odom_delta(odom_laser)
            if delta_d < cfg.loc_update_min_d and delta_a < cfg.loc_update_min_a:
                # 必须刷新 _last_odom：否则静止期间 _odom_delta 持续累积，
                # 一旦重新移动会一次性释放大跳变，把 PF 粒子云 predict 甩飞。
                self._last_odom = Pose2D(odom_laser.x, odom_laser.y, odom_laser.yaw)
                ax, ay, ayaw = self.anchor_map_laser
                nx, ny, nyaw = laser_to_map_odom(ax, ay, ayaw, ox, oy, oyaw)
                self.map_odom.x, self.map_odom.y, self.map_odom.yaw = nx, ny, nyaw
                lx, ly, lyaw = map_odom_to_laser(
                    ox, oy, oyaw, self.map_odom.x, self.map_odom.y, self.map_odom.yaw,
                )
                return Pose2D(lx, ly, lyaw)

        self.scan_match_counter += 1
        run_full_match = self._should_run_full_match(odom_laser)

        if self._cfg.use_map_odom_pf:
            pose = self._pf_update(scan, odom_laser, run_full_match)
        else:
            pose = self._scan_match_pose_legacy(scan, odom_laser, run_full_match)

        self._last_odom = Pose2D(
            odom_laser.x, odom_laser.y, odom_laser.yaw,
        )
        return pose

    def _frozen_pose(
        self,
        laser_frame: str,
        stamp_ns: int,
        lookup_odom_laser: LookupOdomLaserFn,
    ) -> Optional[Pose2D]:
        if self.frozen_snapshot is None:
            return None
        odom_laser = lookup_odom_laser(laser_frame, stamp_ns)
        if odom_laser is None:
            return None
        snap = self.frozen_snapshot
        lx, ly = transform_point(odom_laser.x, odom_laser.y, snap.ox, snap.oy, snap.oyaw)
        lyaw = snap.oyaw + odom_laser.yaw
        return Pose2D(lx, ly, lyaw)

    def estimate_pose(
        self,
        scan: SensorScan,
        lookup_map_laser: LookupPoseFn,
        lookup_odom_laser: LookupOdomLaserFn,
    ) -> Optional[Pose2D]:
        if self.active_loc_mode == 'scan_match' and self.scan_match_active:
            odom_laser = lookup_odom_laser(scan.frame_id, scan.stamp_ns)
            if odom_laser is None:
                return None
            return self._scan_match_pose(scan, odom_laser)

        if self.active_loc_mode == 'frozen':
            return self._frozen_pose(scan.frame_id, scan.stamp_ns, lookup_odom_laser)

        return lookup_map_laser(scan.frame_id, scan.stamp_ns)
