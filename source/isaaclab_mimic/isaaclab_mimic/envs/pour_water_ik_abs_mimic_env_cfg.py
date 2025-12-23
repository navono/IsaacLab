# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pour_water.pourwater_unitree_g1_inspire_hand_env_cfg import PourWaterG1InspireFTPEnvCfg


@configclass
class PourWaterG1InspireFTPMimicEnvCfg(PourWaterG1InspireFTPEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Pour Water G1 Inspire FTP env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_pour_water_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the pour water task.
        # object_ref is used for both annotation/demo selection AND coordinate transformation
        # to adapt trajectories to new object positions in generated scenes
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="bottle",  # Used for annotation, demo selection, and coordinate transformation
                subtask_term_signal="grasp_bottle",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 3),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cup",  # Used for annotation, demo selection, and coordinate transformation
                subtask_term_signal="move",
                subtask_term_offset_range=(0, 3),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cup",  # Used for annotation, demo selection, and coordinate transformation
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs
        
        # Add left hand subtask configs to track left hand movements
        # Left hand moves to cup to stabilize it during grasping
        left_hand_subtask_configs = []
        left_hand_subtask_configs.append(
            SubTaskConfig(
                object_ref="cup",  # Left hand moves toward cup
                subtask_term_signal="grasp_bottle",  # Track during bottle grasp phase
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 3),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        left_hand_subtask_configs.append(
            SubTaskConfig(
                object_ref="cup",  # Left hand stays near cup
                subtask_term_signal="move",
                subtask_term_offset_range=(0, 3),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        left_hand_subtask_configs.append(
            SubTaskConfig(
                object_ref="cup",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = left_hand_subtask_configs