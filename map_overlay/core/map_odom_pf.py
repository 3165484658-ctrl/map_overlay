"""map→odom 轻量粒子滤波（AMCL 式 likelihood 更新，无每帧 grid search）。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from map_overlay.core.types import OverlayConfig, Pose2D
from map_overlay.utils.geometry import map_odom_to_laser, normalize_angle


@dataclass
class _Particle:
    x: float
    y: float
    yaw: float
    weight: float = 1.0


class MapOdomParticleFilter:
    """粒子表示 map→odom；每帧 odom 递推 + likelihood field 加权。"""

    def __init__(self, config: OverlayConfig) -> None:
        self._cfg = config
        self._particles: List[_Particle] = []
        self._ready = False
        self._update_count = 0

    @property
    def ready(self) -> bool:
        return self._ready and bool(self._particles)

    def reset(self) -> None:
        self._particles.clear()
        self._ready = False
        self._update_count = 0

    def initialize(self, mo_x: float, mo_y: float, mo_yaw: float) -> None:
        cfg = self._cfg
        n = cfg.pf_num_particles
        spread_xy = cfg.pf_init_xy
        spread_yaw = cfg.pf_init_yaw
        self._particles = []
        for _ in range(n):
            self._particles.append(_Particle(
                mo_x + random.gauss(0.0, spread_xy),
                mo_y + random.gauss(0.0, spread_xy),
                normalize_angle(mo_yaw + random.gauss(0.0, spread_yaw)),
                1.0 / n,
            ))
        self._ready = True
        self._update_count = 0

    def refine(self, mo_x: float, mo_y: float, mo_yaw: float) -> None:
        """scan_match 后收紧粒子云。"""
        cfg = self._cfg
        n = cfg.pf_num_particles
        spread_xy = cfg.pf_refine_xy
        spread_yaw = cfg.pf_refine_yaw
        self._particles = []
        for _ in range(n):
            self._particles.append(_Particle(
                mo_x + random.gauss(0.0, spread_xy),
                mo_y + random.gauss(0.0, spread_xy),
                normalize_angle(mo_yaw + random.gauss(0.0, spread_yaw)),
                1.0 / n,
            ))
        self._ready = True

    def predict(self, odom_delta_d: float, odom_delta_a: float) -> None:
        """gmapping drawFromMotion 风格运动模型（适配 map→odom 粒子）。

        关键修正：粒子表示 map→odom（准静态量），【不应】随 odom 平移。
        旧实现把粒子当机器人位姿、按 odom 增量前推，会让 map→odom 跟着
        里程计一起飘——这正是退化场景持续漂移的根因之一。

        正确做法：odom 运动只带来 map→odom 的“漂移不确定性”，故仅施加扩散噪声；
        噪声幅度按 gmapping drawFromMotion 随运动量缩放并带平移-旋转耦合。
        测量更新（likelihood 加权 + scan_match refine）负责把 map→odom 拉回真值。
        """
        if not self._particles:
            return
        cfg = self._cfg
        ad = abs(odom_delta_d)
        aa = abs(odom_delta_a)
        srr = cfg.pf_trans_alpha          # 平移→平移
        rtt = cfg.pf_rot_alpha            # 旋转→旋转
        srt = 0.1 * srr                   # 平移→旋转耦合（gmapping 式）
        # sigma_t = srr*|d| + rtt*|theta|；sigma_r = rtt*|theta| + srt*|d|
        sigma_xy = cfg.pf_noise_xy + srr * ad + rtt * aa
        sigma_yaw = cfg.pf_noise_yaw + rtt * aa + srt * ad
        for p in self._particles:
            # 仅扩散，不平移：map→odom 是准静态量
            p.x += random.gauss(0.0, sigma_xy)
            p.y += random.gauss(0.0, sigma_xy)
            p.yaw = normalize_angle(p.yaw + random.gauss(0.0, sigma_yaw))

    def set_weights(self, scores: List[float]) -> None:
        if len(scores) != len(self._particles):
            return
        # AMCL 式：把平均 likelihood 通过指数锐化，拉大好坏 pose 的权重差距
        tau = max(self._cfg.pf_likelihood_temp, 1e-6)
        clipped = [max(s, 1e-9) for s in scores]
        log_weights = [math.log(s) / tau for s in clipped]
        max_lw = max(log_weights)
        unnormalized = [math.exp(lw - max_lw) for lw in log_weights]
        total = sum(unnormalized)
        if total <= 1e-12:
            w = 1.0 / len(self._particles)
            for p in self._particles:
                p.weight = w
            return
        for p, w in zip(self._particles, unnormalized):
            p.weight = w / total
        self._update_count += 1

    def maybe_resample(self) -> None:
        # 只在达到间隔时考虑重采样
        if self._update_count % self._cfg.pf_resample_interval != 0:
            return
        n = len(self._particles)
        if n == 0:
            return
        # 有效粒子数低于一半时才重采样，避免粒子贫化
        neff = 1.0 / max(sum(p.weight * p.weight for p in self._particles), 1e-12)
        if neff < 0.5 * n:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        n = len(self._particles)
        if n == 0:
            return
        positions = [(random.random() + i) / n for i in range(n)]
        cumulative = []
        acc = 0.0
        for p in self._particles:
            acc += p.weight
            cumulative.append(acc)
        new_particles: List[_Particle] = []
        idx = 0
        for pos in positions:
            while idx < n - 1 and cumulative[idx] < pos:
                idx += 1
            src = self._particles[idx]
            new_particles.append(_Particle(src.x, src.y, src.yaw, 1.0 / n))
        self._particles = new_particles

    def estimate(self) -> Tuple[float, float, float]:
        if not self._particles:
            return 0.0, 0.0, 0.0
        sx = sy = 0.0
        sin_sum = cos_sum = 0.0
        for p in self._particles:
            sx += p.weight * p.x
            sy += p.weight * p.y
            sin_sum += p.weight * math.sin(p.yaw)
            cos_sum += p.weight * math.cos(p.yaw)
        return sx, sy, normalize_angle(math.atan2(sin_sum, cos_sum))

    def estimate_laser_pose(
        self, odom_x: float, odom_y: float, odom_yaw: float,
    ) -> Pose2D:
        mx, my, myaw = self.estimate()
        lx, ly, lyaw = map_odom_to_laser(odom_x, odom_y, odom_yaw, mx, my, myaw)
        return Pose2D(lx, ly, lyaw)

    def particle_map_odoms(self) -> List[Tuple[float, float, float]]:
        return [(p.x, p.y, p.yaw) for p in self._particles]
