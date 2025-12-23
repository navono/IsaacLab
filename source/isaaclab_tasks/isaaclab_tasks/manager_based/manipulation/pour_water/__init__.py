# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import (
    agents,
    pourwater_unitree_g1_inspire_hand_env_cfg,
)

gym.register(
    id="Isaac-PourWater-G1-InspireFTP-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pourwater_unitree_g1_inspire_hand_env_cfg.PourWaterG1InspireFTPEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)
