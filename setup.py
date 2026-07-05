from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'map_overlay'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ztl',
    maintainer_email='ztl@todo.todo',
    description='Local occupancy-grid overlay mapping on an existing AMCL map',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_overlay_node = map_overlay.map_overlay_node:main',
            'map_saver_node = map_overlay.map_saver_node:main',
        ],
    },
)
