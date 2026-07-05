#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map_overlay ROS 节点（薄封装）。

算法核：core.overlay_processor.OverlayProcessor（≈ gmapping GridSlamProcessor）
  process_scan(): 定位 → 建图 → 匹配图同步

保存：独立 map_saver_node

典型流程：
  ros2 launch map_overlay overlay_mapping.launch.py
  RViz 2D Pose Estimate → start_mapping → 慢速走 → save_map
"""

from __future__ import annotations

import math
import os
import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener, TransformException

from map_overlay.core.overlay_processor import OverlayProcessor
from map_overlay.core.types import OverlayConfig, Pose2D, SensorScan
from map_overlay.ros.amcl_lifecycle import AmclLifecycle
from map_overlay.ros.map_loader import load_occupancy_grid_from_yaml
from map_overlay.utils.geometry import quaternion_from_yaw, yaw_from_quaternion


def scan_from_ros(msg: LaserScan) -> SensorScan:
    stamp = rclpy.time.Time.from_msg(msg.header.stamp)
    return SensorScan(
        frame_id=msg.header.frame_id,
        stamp_ns=stamp.nanoseconds,
        angle_min=msg.angle_min,
        angle_increment=msg.angle_increment,
        ranges=tuple(msg.ranges),
    )


class MapOverlayNode(Node):
    """≈ gmapping SlamGMapping：ROS 接口 + 调度 OverlayProcessor。"""

    MAP_QOS = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    def __init__(self) -> None:
        super().__init__('map_overlay')
        self._declare_parameters()
        cfg = OverlayConfig.from_node_params(self)
        self._processor = OverlayProcessor(cfg)
        self._cfg = cfg

        self._tf_timeout = float(self.get_parameter('tf_timeout').value)
        self._map_odom_pub_rate = float(self.get_parameter('map_odom_publish_rate').value)
        self._auto_deactivate_amcl = bool(self.get_parameter('auto_deactivate_amcl_on_start').value)
        self._reactivate_amcl_on_stop = bool(self.get_parameter('reactivate_amcl_on_stop').value)
        self._use_scan_match = cfg.use_scan_match_when_mapping
        self._loc_mode = cfg.loc_mode

        self._tf_hint_logged = False
        self._frozen_map_odom_msg: Optional[TransformStamped] = None

        self._amcl = AmclLifecycle(
            str(self.get_parameter('amcl_node_name').value),
            str(self.get_parameter('lifecycle_manager_name').value),
            float(self.get_parameter('ros2_cmd_timeout').value),
            self.get_logger().info,
            self.get_logger().warn,
        )

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        map_out = self.get_parameter('map_topic_out').value
        self._map_pub = self.create_publisher(OccupancyGrid, map_out, self.MAP_QOS)

        scan_topic = self.get_parameter('scan_topic').value
        map_yaml = str(self.get_parameter('map_yaml_path').value).strip()
        if map_yaml:
            self._load_map_from_yaml(os.path.expanduser(map_yaml))
        else:
            map_in = self.get_parameter('map_topic_in').value
            self.create_subscription(OccupancyGrid, map_in, self._on_map, self.MAP_QOS)

        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

        pub_rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / pub_rate, self._publish_map)
        if self._map_odom_pub_rate > 0.0:
            self.create_timer(1.0 / self._map_odom_pub_rate, self._publish_map_odom_tf)

        self.create_service(Trigger, '/map_overlay/start_mapping', self._start_mapping_cb)
        self.create_service(Trigger, '/map_overlay/stop_mapping', self._stop_mapping_cb)

        if bool(self.get_parameter('start_mapping_on_launch').value):
            self._processor.start_mapping()

        self._log_startup(map_yaml)
        self._publish_map(force=True)

    def _declare_parameters(self) -> None:
        defaults = {
            'scan_topic': '/scan',
            'map_yaml_path': '',
            'map_topic_in': '/map',
            'map_topic_out': '/map',
            'map_frame': 'map',
            'odom_frame': 'odom',
            'localization_mode': 'amcl',
            'use_scan_match_when_mapping': True,
            'scan_match_search_xy': 0.20,
            'scan_match_search_yaw': 0.12,
            'scan_match_steps': 15,
            'scan_match_beam_stride': 1,
            'scan_match_max_beams': 300,
            'scan_match_field_cache_cells': 5,
            'scan_match_min_score': 0.60,
            'scan_match_likelihood_sigma': 0.30,
            'scan_match_likelihood_max_dist': 2.0,
            'scan_match_field_radius': 0.0,
            'scan_match_yaw_reach': 3.0,
            'scan_match_fine_iterations': 8,
            'scan_match_fine_xy': 0.03,
            'scan_match_fine_yaw': 0.02,
            'scan_match_fine_steps': 5,
            'match_map_local_sync': True,
            'match_map_full_refresh_updates': 0,
            'scan_match_every_n_scans': 1,
            'scan_match_trigger_d': 0.05,
            'scan_match_trigger_a': 0.03,
            'scan_match_log_timing': True,
            'map_odom_smoothing': 0.08,
            'use_map_odom_pf': True,
            'pf_num_particles': 1000,
            'pf_resample_interval': 5,
            'pf_noise_xy': 0.002,
            'pf_noise_yaw': 0.002,
            'pf_trans_alpha': 0.04,
            'pf_rot_alpha': 0.02,
            'pf_init_xy': 0.02,
            'pf_init_yaw': 0.015,
            'pf_refine_xy': 0.01,
            'pf_refine_yaw': 0.008,
            'loc_update_min_d': 0.005,
            'loc_update_min_a': 0.005,
            'pf_likelihood_temp': 0.30,
            'large_map_cell_threshold': 2_000_000,
            'ros2_cmd_timeout': 3.0,
            'map_odom_publish_rate': 30.0,
            'update_radius': 5.0,
            'enable_clearing': True,
            'integrate_every_n_scans': 3,
            'min_update_distance': 0.10,
            'min_update_angle': 0.10,
            'log_odds_hit': 0.40,
            'log_odds_miss': 0.40,
            'log_odds_min': -4.0,
            'log_odds_max': 4.0,
            'unknown_prior_prob': 0.35,
            'map_occ_display_mode': 'linear',
            'occupied_thresh': 0.65,
            'free_thresh': 0.196,
            'occ_confirm_hits': 3,
            'occ_confirm_misses': 2,
            'scan_range_min': 0.15,
            'scan_range_max': 12.0,
            'scan_match_min_range': 0.80,
            'publish_rate': 5.0,
            'publish_map_on_integrate': True,
            'start_mapping_on_launch': False,
            'wait_for_map_timeout': 30.0,
            'tf_timeout': 0.3,
            'auto_deactivate_amcl_on_start': True,
            'reactivate_amcl_on_stop': False,
            'amcl_node_name': 'amcl',
            'lifecycle_manager_name': 'lifecycle_manager_localization',
        }
        for name, value in defaults.items():
            if isinstance(value, bool):
                self.declare_parameter(name, value)
            elif isinstance(value, int):
                self.declare_parameter(name, value)
            elif isinstance(value, float):
                self.declare_parameter(name, value)
            else:
                self.declare_parameter(name, value)

    def _log_startup(self, map_yaml: str) -> None:
        proc = self._processor
        self.get_logger().info(
            f'map_overlay ready  loc={self._loc_mode}  '
            f'scan_match_on_start={self._use_scan_match}  '
            f'processor=OverlayProcessor  '
            f'mapping={"ON" if proc.mapping_active else "OFF(等 start_mapping)"}'
        )
        if not proc.mapping_active:
            self.get_logger().info(
                '【定位阶段】仅发布底图 /map，不融合激光。'
                '请 RViz「2D Pose Estimate」调整 AMCL 后：\n'
                '  ros2 service call /map_overlay/start_mapping std_srvs/srv/Trigger'
            )
        self.get_logger().info(
            '保存: ros2 service call /map_overlay/save_map std_srvs/srv/Trigger (map_saver 节点)'
        )
        if map_yaml:
            self.get_logger().info(f'底图 map_yaml_path={map_yaml}')

    # ------------------------------------------------------------------ map IO

    def _publish_map(self, force: bool = False) -> None:
        grid = self._processor.grid
        if not grid.ready or grid.grid is None:
            return
        if not force and not grid.dirty:
            return
        grid.grid.header.stamp = self.get_clock().now().to_msg()
        self._map_pub.publish(grid.grid)
        grid.dirty = False

    def _load_map_from_yaml(self, yaml_path: str) -> None:
        grid = load_occupancy_grid_from_yaml(yaml_path, self._cfg.map_frame)
        self._processor.grid.load_grid(grid)
        self._processor.refresh_match_snapshot()
        info = grid.info
        self.get_logger().info(
            f'loaded map yaml={yaml_path}  size={info.width}x{info.height}  '
            f'res={info.resolution:.3f}'
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        if self._processor.map_ready:
            return
        grid = OccupancyGrid()
        grid.header = msg.header
        grid.info = msg.info
        grid.data = list(msg.data)
        self._processor.grid.load_grid(grid)
        self._processor.refresh_match_snapshot()
        self.get_logger().info(
            f'loaded base map {msg.info.width}x{msg.info.height}  '
            f'res={msg.info.resolution:.3f}'
        )
        self._publish_map(force=True)

    # ------------------------------------------------------------------ services

    def _start_mapping_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        proc = self._processor
        if proc.mapping_active:
            response.success = True
            response.message = 'mapping already active'
            return response
        if not proc.map_ready:
            response.success = False
            response.message = 'map not ready'
            return response

        if self._loc_mode == 'frozen':
            self._frozen_map_odom_msg = None
            if not self._init_map_odom_from_tf():
                response.success = False
                response.message = 'cannot lock map->odom'
                return response
            proc.begin_mapping_frozen()
        elif self._use_scan_match or self._loc_mode == 'scan_match':
            if not self._init_map_odom_from_tf():
                response.success = False
                response.message = 'map->odom TF not ready (请先完成 AMCL 定位)'
                return response
            proc.begin_mapping_scan_match()
            if self._auto_deactivate_amcl:
                self._amcl.set_tf_broadcast(False)
        else:
            if not self._init_map_odom_from_tf(check_only=True):
                response.success = False
                response.message = 'map->odom TF not ready (请先完成 AMCL 定位)'
                return response
            proc.begin_mapping_amcl_passive()

        proc.start_mapping()
        self._tf_hint_logged = False
        response.success = True
        response.message = 'mapping started'
        self.get_logger().info(
            f'【建图阶段】loc={proc.localization.active_loc_mode}；'
            '完成后 save_map 或 stop_mapping'
        )

        if proc.localization.scan_match_active:
            def _start_bg() -> None:
                if self._auto_deactivate_amcl:
                    if not self._amcl.pause_via_manager() and not self._amcl.deactivate():
                        self.get_logger().warn(
                            'AMCL lifecycle pause 未完成，已依赖 tf_broadcast=false'
                        )
                if proc.grid.is_large():
                    proc.prepare_match_snapshot_large_map()
                    self.get_logger().info('大地图：匹配使用实时栅格')
                else:
                    proc.refresh_match_snapshot()

            threading.Thread(target=_start_bg, daemon=True).start()

        return response

    def _stop_mapping_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        proc = self._processor
        if not proc.mapping_active:
            response.success = True
            response.message = 'mapping already stopped'
            return response
        was_scan_match = proc.localization.scan_match_active
        proc.stop_mapping()
        self._frozen_map_odom_msg = None
        if was_scan_match and self._reactivate_amcl_on_stop:
            self._amcl.resume()
        response.success = True
        response.message = 'mapping stopped'
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------ TF helpers

    def _init_map_odom_from_tf(self, check_only: bool = False) -> bool:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._cfg.map_frame, self._cfg.odom_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
            )
        except TransformException as exc:
            self.get_logger().error(f'map->odom unavailable: {exc}')
            return False
        if check_only:
            return True
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._processor.localization.lock_map_odom(t.x, t.y, yaw)
        self._frozen_map_odom_msg = tf
        self.get_logger().info(
            f'locked map->odom ({t.x:.3f}, {t.y:.3f}, {math.degrees(yaw):.1f}°)'
        )
        return True

    def _publish_map_odom_tf(self) -> None:
        if not self._processor.localization.scan_match_active:
            return
        mo = self._processor.localization.map_odom
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._cfg.map_frame
        t.child_frame_id = self._cfg.odom_frame
        t.transform.translation.x = mo.x
        t.transform.translation.y = mo.y
        qx, qy, qz, qw = quaternion_from_yaw(mo.yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(t)

    def _lookup_odom_laser(self, laser_frame: str, stamp_ns: int) -> Optional[Pose2D]:
        stamp = rclpy.time.Time(nanoseconds=stamp_ns) if stamp_ns else rclpy.time.Time()
        try:
            tf = self._tf_buffer.lookup_transform(
                self._cfg.odom_frame, laser_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
            )
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return Pose2D(t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _lookup_map_laser(self, laser_frame: str, stamp_ns: int) -> Optional[Pose2D]:
        stamp = rclpy.time.Time(nanoseconds=stamp_ns) if stamp_ns else rclpy.time.Time()
        for try_stamp in (rclpy.time.Time(), stamp):
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._cfg.map_frame, laser_frame, try_stamp,
                    timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
                )
            except TransformException as exc:
                if try_stamp.nanoseconds != 0:
                    continue
                if not self._tf_hint_logged:
                    self._tf_hint_logged = True
                    self.get_logger().error(
                        f'TF {self._cfg.map_frame}->{laser_frame} 不可用: {exc}'
                    )
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            return Pose2D(t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        return None

    # ------------------------------------------------------------------ laserCallback → process_scan

    def _on_scan(self, scan: LaserScan) -> None:
        """≈ gmapping laserCallback：转调 OverlayProcessor.process_scan。"""
        sensor = scan_from_ros(scan)

        def lookup_map(frame: str, stamp_ns: int) -> Optional[Pose2D]:
            return self._lookup_map_laser(frame, stamp_ns)

        result = self._processor.process_scan(
            sensor,
            lookup_map_laser=lookup_map,
            lookup_odom_laser=self._lookup_odom_laser,
        )

        if result.should_publish_map:
            self._publish_map()

        if result.match_timing is not None:
            t = result.match_timing
            cache_note = ' cached' if t.field_cached else ''
            self.get_logger().info(
                f'scan_match  field={t.field_ms:.0f}ms{cache_note}  '
                f'coarse={t.coarse_ms:.0f}ms  fine={t.fine_ms:.0f}ms  '
                f'total={t.total_ms:.0f}ms  process_scan={result.process_scan_ms:.0f}ms',
                throttle_duration_sec=2.0,
            )

        if result.accepted and result.scan_count % 50 == 0 and result.pose is not None:
            p = result.pose
            extra = ''
            if self._processor.localization.scan_match_active:
                mo = self._processor.localization.map_odom
                extra = (
                    f'  map_odom=({mo.x:.2f},{mo.y:.2f},'
                    f'{math.degrees(mo.yaw):.1f}°)'
                )
            self.get_logger().info(
                f'scans={result.scan_count}  integrated={result.update_count}  '
                f'pose=({p.x:.2f},{p.y:.2f},{math.degrees(p.yaw):.1f}°){extra}',
                throttle_duration_sec=5.0,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapOverlayNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
