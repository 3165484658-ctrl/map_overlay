#!/usr/bin/env python3
"""
局部地图覆盖建图 + AMCL 定位。

启动后【仅定位】：发布底图 /map + AMCL，不融合激光。
定位调好后手动开始建图：
  ros2 service call /map_overlay/start_mapping std_srvs/srv/Trigger

保存：
  ros2 service call /map_overlay/save_map std_srvs/srv/Trigger

用法：
  ros2 launch map_overlay overlay_mapping.launch.py map_id:=68.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
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
    default_amcl_config = os.path.join(pkg_dir, 'config', 'amcl_params.yaml')
    default_overlay_config = os.path.join(pkg_dir, 'config', 'map_overlay_params.yaml')
    default_saver_config = os.path.join(pkg_dir, 'config', 'map_saver_params.yaml')

    map_yaml_path = _resolve_map_yaml(context)
    with_amcl = LaunchConfiguration('with_amcl').perform(context)
    start_on_launch = LaunchConfiguration('start_mapping_on_launch').perform(context)

    overlay_params = [
        default_overlay_config,
        {
            'map_yaml_path': map_yaml_path,
            'localization_mode': LaunchConfiguration('localization_mode').perform(context),
            'update_radius': float(LaunchConfiguration('update_radius').perform(context)),
            'enable_clearing': LaunchConfiguration('enable_clearing').perform(context) == 'true',
            'start_mapping_on_launch': start_on_launch == 'true',
        },
    ]

    overlay_node = Node(
        package='map_overlay',
        executable='map_overlay_node',
        name='map_overlay',
        output='screen',
        parameters=overlay_params,
    )

    map_saver_node = Node(
        package='map_overlay',
        executable='map_saver_node',
        name='map_saver',
        output='screen',
        parameters=[
            default_saver_config,
            {
                'map_save_path': os.path.splitext(map_yaml_path)[0] + '_overlay',
            },
        ],
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[default_amcl_config, {'use_sim_time': False}],
    )

    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['amcl'],
        }],
    )

    actions = [overlay_node, map_saver_node]
    if with_amcl == 'true':
        actions.append(TimerAction(
            period=1.0,
            actions=[amcl_node, lifecycle_manager_localization],
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_id', default_value='68.yaml'),
        DeclareLaunchArgument('map_yaml', default_value=''),
        DeclareLaunchArgument('with_amcl', default_value='true'),
        DeclareLaunchArgument('start_mapping_on_launch', default_value='false'),
        DeclareLaunchArgument('localization_mode', default_value='amcl'),
        DeclareLaunchArgument('update_radius', default_value='25.0'),
        DeclareLaunchArgument('enable_clearing', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
