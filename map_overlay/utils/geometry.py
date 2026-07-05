"""2D 几何工具（无 ROS 依赖）。"""

from __future__ import annotations

import math
from typing import Tuple


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_point(x: float, y: float, tx: float, ty: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y + tx, s * x + c * y + ty


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def laser_to_map_odom(
    lx: float, ly: float, lyaw: float,
    ox: float, oy: float, oyaw: float,
) -> Tuple[float, float, float]:
    myaw = normalize_angle(lyaw - oyaw)
    c = math.cos(myaw)
    s = math.sin(myaw)
    mx = lx - c * ox + s * oy
    my = ly - s * ox - c * oy
    return mx, my, myaw


def map_odom_to_laser(
    ox: float, oy: float, oyaw: float,
    map_odom_x: float, map_odom_y: float, map_odom_yaw: float,
) -> Tuple[float, float, float]:
    lx, ly = transform_point(ox, oy, map_odom_x, map_odom_y, map_odom_yaw)
    lyaw = normalize_angle(map_odom_yaw + oyaw)
    return lx, ly, lyaw
