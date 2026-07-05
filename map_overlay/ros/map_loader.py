"""从 yaml/pgm 加载 OccupancyGrid。"""

from __future__ import annotations

import os

from nav_msgs.msg import OccupancyGrid

from map_overlay.utils.log_odds import occ_to_log_odds


def load_occupancy_grid_from_yaml(yaml_path: str, map_frame: str) -> OccupancyGrid:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'map yaml not found: {yaml_path}')

    meta: dict = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            meta[key.strip()] = val.strip()

    image_name = meta['image']
    pgm_path = image_name if os.path.isabs(image_name) else os.path.join(
        os.path.dirname(yaml_path), image_name,
    )
    resolution = float(meta['resolution'])
    origin_parts = meta['origin'].strip('[]').split(',')
    origin_x = float(origin_parts[0].strip())
    origin_y = float(origin_parts[1].strip())

    with open(pgm_path, 'rb') as f:
        header = f.readline().strip()
        if header != b'P5':
            raise ValueError(f'unsupported PGM format: {pgm_path}')
        dims = f.readline()
        while dims.startswith(b'#'):
            dims = f.readline()
        width, height = [int(x) for x in dims.split()]
        maxval = int(f.readline().strip())
        if maxval != 255:
            raise ValueError(f'expected 8-bit PGM: {pgm_path}')
        raw = f.read()

    grid = OccupancyGrid()
    grid.header.frame_id = map_frame
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = origin_x
    grid.info.origin.position.y = origin_y
    grid.info.origin.orientation.w = 1.0

    data = [-1] * (width * height)
    for row in range(height):
        src_row = height - 1 - row
        for col in range(width):
            pgm_val = raw[src_row * width + col]
            if pgm_val >= 250:
                occ = 0
            elif pgm_val <= 10:
                occ = 100
            else:
                occ = -1
            data[col + row * width] = occ
    grid.data = data
    return grid


def grid_log_odds(grid: OccupancyGrid) -> list:
    return [occ_to_log_odds(v) for v in grid.data]
