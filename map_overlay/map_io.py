#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OccupancyGrid 读写工具（map_overlay / map_saver 共用）。"""

from __future__ import annotations

import os
from typing import List, Tuple


def write_occupancy_grid_to_files(
    prefix: str,
    info,
    data: List[int],
) -> Tuple[str, str]:
    """将栅格数据写入 prefix.pgm + prefix.yaml，返回 (pgm_path, yaml_path)。"""
    width = info.width
    height = info.height
    pgm_path = prefix + '.pgm'
    yaml_path = prefix + '.yaml'

    with open(pgm_path, 'wb') as f:
        f.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
        for my in range(height - 1, -1, -1):
            row = bytearray()
            row_base = my * width
            for mx in range(width):
                occ = data[mx + row_base]
                if occ < 0:
                    row.append(205)
                elif occ >= 50:
                    row.append(0)
                else:
                    row.append(254)
            f.write(row)

    origin = info.origin
    yaml_text = (
        f'image: {os.path.basename(pgm_path)}\n'
        f'resolution: {info.resolution}\n'
        f'origin: [{origin.position.x}, {origin.position.y}, 0.0]\n'
        f'negate: 0\n'
        f'occupied_thresh: 0.65\n'
        f'free_thresh: 0.196\n'
    )
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)
    return pgm_path, yaml_path
