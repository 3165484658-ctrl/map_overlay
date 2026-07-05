#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立地图保存节点：订阅 /map，后台写 PGM/YAML，不阻塞 map_overlay。

用法：
  ros2 run map_overlay map_saver_node
  ros2 service call /map_overlay/save_map std_srvs/srv/Trigger
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from map_overlay.map_io import write_occupancy_grid_to_files


class MapSaverNode(Node):
    MAP_QOS = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    def __init__(self) -> None:
        super().__init__('map_saver')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_save_path', '~/maps/overlay_map')
        self.declare_parameter('save_service_name', '/map_overlay/save_map')
        self.declare_parameter('large_map_cell_threshold', 2_000_000)

        map_topic = str(self.get_parameter('map_topic').value)
        self._map_save_path = os.path.expanduser(str(self.get_parameter('map_save_path').value))
        save_srv = str(self.get_parameter('save_service_name').value)
        self._large_map_threshold = int(self.get_parameter('large_map_cell_threshold').value)

        self._map_lock = threading.RLock()
        self._map_ready = False
        self._save_in_progress = False
        self._grid_info: Optional[MapMetaData] = None
        self._grid_data: Optional[List[int]] = None

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, self.MAP_QOS)
        self.create_service(Trigger, save_srv, self._save_map_cb)

        self.get_logger().info(
            f'map_saver ready  topic={map_topic}  save={save_srv}\n'
            f'  → {self._map_save_path}.yaml + .pgm'
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        with self._map_lock:
            self._grid_info = msg.info
            self._grid_data = list(msg.data)
            self._map_ready = True

    def _save_map_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        with self._map_lock:
            if not self._map_ready or self._grid_info is None or self._grid_data is None:
                response.success = False
                response.message = 'no /map received yet'
                return response
            if self._save_in_progress:
                response.success = False
                response.message = 'save already in progress, please wait'
                return response
            info = self._grid_info
            data = list(self._grid_data)
            self._save_in_progress = True

        prefix = self._map_save_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        except OSError as exc:
            with self._map_lock:
                self._save_in_progress = False
            response.success = False
            response.message = str(exc)
            return response

        cells = info.width * info.height

        def _save_worker() -> None:
            try:
                pgm_path, yaml_path = write_occupancy_grid_to_files(prefix, info, data)
                self.get_logger().info(f'saved {yaml_path} and {pgm_path}')
            except OSError as exc:
                self.get_logger().error(f'save_map failed: {exc}')
            finally:
                with self._map_lock:
                    self._save_in_progress = False

        threading.Thread(target=_save_worker, daemon=True).start()
        response.success = True
        if cells >= self._large_map_threshold:
            response.message = (
                f'saving {cells // 1_000_000}M cells to {prefix} in background '
                f'(watch map_saver log)'
            )
        else:
            response.message = f'saving to {prefix}.yaml in background (map_saver)'
        self.get_logger().info(response.message)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapSaverNode()
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
