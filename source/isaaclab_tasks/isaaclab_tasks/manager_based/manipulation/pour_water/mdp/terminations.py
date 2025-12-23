# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_bottle_pour(
    env: ManagerBasedRLEnv,
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    mouth_offset: float = 0.10,
    max_horizontal_distance: float = 0.35,  # Greatly relaxed from 0.15 to 0.35
    min_height_difference: float = -0.05,  # Allow mouth slightly below cup rim
) -> torch.Tensor:
    """Success when bottle is tilted AND bottle is near cup (relaxed conditions for generation)."""

    bottle: RigidObject = env.scene[bottle_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]

    # Bottle and cup positions
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins
    cup_pos = cup.data.root_pos_w - env.scene.env_origins

    # Bottle local +Z (up) in world frame
    rot_mats = math_utils.matrix_from_quat(bottle.data.root_quat_w)
    bottle_up = rot_mats[:, :, 2]

    # Approximate mouth and bottom positions using bottle_up direction
    mouth_pos = bottle_pos + bottle_up * mouth_offset
    bottom_pos = bottle_pos - bottle_up * mouth_offset

    # Condition 1: bottle is tilted (bottom higher than mouth)
    bottle_tilted = bottom_pos[:, 2] > mouth_pos[:, 2]

    # Condition 2: bottle mouth is above cup and within horizontal range
    height_diff = mouth_pos[:, 2] - cup_pos[:, 2]  # mouth height - cup height
    horizontal_dist = torch.linalg.norm(mouth_pos[:, :2] - cup_pos[:, :2], dim=1)

    mouth_above_cup = height_diff > min_height_difference
    mouth_within_range = horizontal_dist < max_horizontal_distance

    # Both conditions must be satisfied
    done = torch.logical_and(bottle_tilted, mouth_above_cup)
    done = torch.logical_and(done, mouth_within_range)

    return done
