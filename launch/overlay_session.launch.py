#!/usr/bin/env python3
"""等同于 overlay_mapping.launch.py（with_amcl 默认 true）。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_dir = get_package_share_directory('map_overlay')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'overlay_mapping.launch.py'),
            ),
        ),
    ])
