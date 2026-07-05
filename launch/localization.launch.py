#!/usr/bin/env python3
"""
AMCL + map_server 定位。

改地图：
  ros2 launch map_overlay localization.launch.py map_id:=68.yaml
  ros2 launch map_overlay localization.launch.py map_yaml:=/home/ztl/ros2_ws/src/robot_mapping/map/68.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MAP_DIR = '/home/ztl/ros2_ws/src/robot_mapping/map'


def _resolve_map_yaml(context) -> str:
    explicit = LaunchConfiguration('map_yaml').perform(context).strip()
    if explicit:
        return explicit
    map_id = LaunchConfiguration('map_id').perform(context).strip()
    return os.path.join(MAP_DIR, map_id)


def _launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('map_overlay')
    amcl_config = os.path.join(pkg_dir, 'config', 'amcl_params.yaml')
    map_yaml_path = _resolve_map_yaml(context)

    return [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_yaml_path,
                'use_sim_time': False,
            }],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[amcl_config, {'use_sim_time': False}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['map_server', 'amcl'],
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_id', default_value='68.yaml'),
        DeclareLaunchArgument('map_yaml', default_value=''),
        OpaqueFunction(function=_launch_setup),
    ])
