# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_robot_joint_state(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    # hand_joint_names is a list of regex, use find_joints
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    robot_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return robot_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


def grasp_bottle(
    env: ManagerBasedRLEnv,
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    initial_bottle_height: float = 1.09,
    lift_threshold: float = 0.03,
) -> torch.Tensor:
    """Detect if the bottle is grasped by checking if bottle is lifted above its initial position.
    
    The bottle is considered grasped when it's lifted above the table surface.
    We detect this by checking if the bottle center height is higher than its initial resting height.
    
    Args:
        env: The environment.
        bottle_cfg: The bottle scene entity configuration.
        initial_bottle_height: The initial height of bottle center when resting on table (default 1.09m).
        lift_threshold: How much above initial height to count as grasped (default 0.03m).
    
    Returns:
        Boolean tensor indicating if bottle is lifted (grasped).
    """
    bottle: RigidObject = env.scene[bottle_cfg.name]
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins
    bottle_height = bottle_pos[:, 2]
    is_lifted = bottle_height > (initial_bottle_height + lift_threshold)
    return is_lifted.unsqueeze(-1)


def bottle_above_cup_right(
    env: ManagerBasedRLEnv,
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    right_offset_threshold: float = 0.05,
    height_threshold: float = 0.10,
) -> torch.Tensor:
    """Detect if bottle is positioned at the right-upper side of the cup.
    
    This indicates the move phase is complete and ready for pouring.
    The bottle should be:
    1. Above the cup (higher in z)
    2. To the right of the cup (positive x offset)
    
    Args:
        env: The environment.
        bottle_cfg: The bottle scene entity configuration.
        cup_cfg: The cup scene entity configuration.
        right_offset_threshold: Minimum x-offset to the right of cup center.
        height_threshold: Minimum height above cup.
    
    Returns:
        Boolean tensor indicating if bottle is at right-upper position.
    """
    bottle: RigidObject = env.scene[bottle_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins
    cup_pos = cup.data.root_pos_w - env.scene.env_origins
    
    # Check if bottle is to the right of cup (positive x offset)
    x_offset = bottle_pos[:, 0] - cup_pos[:, 0]
    is_right = x_offset > right_offset_threshold
    
    # Check if bottle is above cup
    z_offset = bottle_pos[:, 2] - cup_pos[:, 2]
    is_above = z_offset > height_threshold
    
    return (is_right & is_above).unsqueeze(-1)


