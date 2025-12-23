# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class PourWaterAbsMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Pour Water task with Absolute Pose Control.

    This MimicEnv is used when all observations are in the robot base frame.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose.
        
        Args:
            eef_name: Name of the end effector ("left" or "right").
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
            
        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Use dynamic observation name lookup like working implementations
        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        target_wrist_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_wrist_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target pose to action.

        For PourWater G1 Inspire task with Pink IK controller, action format is:
        - Left wrist position (3) + quaternion (4) = 7
        - Right wrist position (3) + quaternion (4) = 7
        - Hand joints (24)
        - Total: 38 dimensions

        Args:
            target_eef_pose_dict: Dict with "left" and "right" keys containing 4x4 pose matrices.
            gripper_action_dict: Dict with "left" and "right" keys for gripper actions.
            action_noise_dict: Dict with noise values for each EEF. If None, no noise is added.
            env_id: Environment index.

        Returns:
            Action tensor of shape (38,).
        """
        # target position and rotation
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # gripper actions
        left_gripper_action = gripper_action_dict["left"]
        right_gripper_action = gripper_action_dict["right"]

        # 调试输出：只在开始时打印几次
        if not hasattr(self, '_debug_print_count'):
            self._debug_print_count = 0

        if self._debug_print_count < 3:
            print(f"[DEBUG] Env {env_id} - Target Right EEF: pos={target_right_eef_pos.numpy()}, quat={target_right_eef_rot_quat.numpy()}")
            print(f"[DEBUG] Env {env_id} - Right Gripper: shape={right_gripper_action.shape}, range=[{right_gripper_action.min():.3f}, {right_gripper_action.max():.3f}]")

            # 获取当前实际右手姿态
            try:
                current_right_pose = self.get_robot_eef_pose("right", [env_id])
                # current_right_pose 应该是 4x4 变换矩阵
                if current_right_pose.shape[-1] == 4 and current_right_pose.shape[-2] == 4:
                    # 这是 4x4 变换矩阵
                    current_pos = PoseUtils.unmake_pose(current_right_pose)[0]
                else:
                    # 尝试其他格式
                    current_pos = current_right_pose[:3]
                print(f"[DEBUG] Env {env_id} - Current Right EEF: pos={current_pos.numpy()}")
                print(f"[DEBUG] Env {env_id} - Target Z: {target_right_eef_pos[2]:.3f}, Current Z: {current_pos[2]:.3f}")
            except Exception as e:
                print(f"[DEBUG] Could not get current EEF pose: {e}")

            self._debug_print_count += 1

        if action_noise_dict is not None:
            pos_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            pos_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_pos)
            quat_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_rot_quat)
            quat_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_rot_quat)

            target_left_eef_pos += pos_noise_left
            target_right_eef_pos += pos_noise_right
            target_left_eef_rot_quat += quat_noise_left
            target_right_eef_rot_quat += quat_noise_right

        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                left_gripper_action,
                right_gripper_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Converts action to target poses for the end effector controllers.
        
        Action format (38 dims):
        - Left wrist pos (3) + quat (4) = indices 0-6
        - Right wrist pos (3) + quat (4) = indices 7-13
        - Hand joints (24) = indices 14-37
        
        Args:
            action: Action tensor of shape (num_envs, 38) or (T, 38) for trajectory.
            
        Returns:
            Dictionary with "left" and "right" keys containing 4x4 pose matrices.
        """
        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left

        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extracts the gripper actuation part from a sequence of env actions.
        
        Action format (38 dims):
        - Left wrist pos (3) + quat (4) = indices 0-6
        - Right wrist pos (3) + quat (4) = indices 7-13
        - Left hand joints (12) = indices 14-25
        - Right hand joints (12) = indices 26-37
        
        Args:
            actions: Action tensor of shape (T, 38) for trajectory.
            
        Returns:
            Dictionary with "left" and "right" keys for gripper actions.
        """
        return {"left": actions[:, 14:26], "right": actions[:, 26:38]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object (including rigid objects and articulated objects) in the robot base frame.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in robot base frame (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        # 调试输出：只在第一次调用时打印
        if not hasattr(self, '_object_pose_debug_count'):
            self._object_pose_debug_count = 0

        # Get scene state
        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        # Get robot root pose
        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = dict()

        # Process rigid objects
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        # Process articulated objects (except robot)
        for art_name, art_state in articulation_states.items():
            if art_name != "robot":  # Skip robot
                pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                    root_pos, root_quat, art_state["root_pose"][env_ids, :3], art_state["root_pose"][env_ids, 3:7]
                )
                rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
                object_pose_matrix[art_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        # 调试输出：打印物体位置
        if self._object_pose_debug_count < 1:
            print("[DEBUG OBJECT] Current scene object positions:")
            for obj_name, pose_matrix in object_pose_matrix.items():
                pos = pose_matrix[0, :3]
                print(f"  {obj_name}: pos={pos.numpy()}")
            self._object_pose_debug_count += 1

        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)
        
        signals = dict()
        
        # Read subtask termination signals from observation buffer
        if "subtask_terms" in self.obs_buf:
            subtask_obs = self.obs_buf["subtask_terms"]
            if "grasp_bottle" in subtask_obs:
                signals["grasp_bottle"] = subtask_obs["grasp_bottle"][env_ids].squeeze(-1).float()
            if "move" in subtask_obs:
                signals["move"] = subtask_obs["move"][env_ids].squeeze(-1).float()
        
        return signals