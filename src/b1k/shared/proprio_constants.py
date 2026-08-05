"""Proprioception / action index constants for the BEHAVIOR-1K R1Pro robot.

Vendored so training can run without the omnigibson (BEHAVIOR-1K simulator) install.

IMPORTANT — the layout changed between challenge years, and the two are NOT
compatible: 2025 packed 256 dims with sin/cos encodings of every joint, while 2026
packs 61 dims with no sin/cos and adds end-effector pose. Every field the model needs
exists in both, at completely different offsets, so applying the wrong table silently
produces garbage rather than an IndexError.

Do not import PROPRIOCEPTION_INDICES from omnigibson to slice dataset rows: the pinned
BEHAVIOR-1K submodule is the *2025* release and would hand back the 256-D table for
61-D data. Select by observed width instead — see PROPRIO_SPECS_BY_WIDTH and
b1k.policies.b1k_policy.extract_state_from_proprio.

Sources:
  2026: BEHAVIOR-1K origin/main  OmniGibson/omnigibson/eval/utils/eval_utils.py
  2025: pinned submodule        OmniGibson/omnigibson/learning/utils/eval_utils.py
"""
from collections import OrderedDict

import numpy as np

# --- 2026 challenge (LeRobot v3.0 dataset: observation.state has 61 dims) ---
PROPRIOCEPTION_INDICES_2026 = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}
PROPRIO_WIDTH_2026 = 61

# --- 2025 challenge (LeRobot v2.1 dataset: observation.state has 256 dims) ---
PROPRIOCEPTION_INDICES_2025 = {
    "A1": OrderedDict(
        {
            "joint_qpos": np.s_[0:8],
            "joint_qpos_sin": np.s_[8:16],
            "joint_qpos_cos": np.s_[16:24],
            "joint_qvel": np.s_[24:32],
            "joint_qeffort": np.s_[32:40],
            "eef_0_pos": np.s_[40:43],
            "eef_0_quat": np.s_[43:47],
            "grasp_0": np.s_[47:48],
            "gripper_0_qpos": np.s_[48:50],
            "gripper_0_qvel": np.s_[50:52],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],
            "base_qpos_sin": np.s_[247:250],
            "base_qpos_cos": np.s_[250:253],
            "base_qvel": np.s_[253:256],
        }
    ),
}
PROPRIO_WIDTH_2025 = 256

# Width -> spec. extract_state_from_proprio dispatches on the trailing dim so the same
# code reads either year's data without a caller-supplied flag.
PROPRIO_SPECS_BY_WIDTH = {
    PROPRIO_WIDTH_2026: PROPRIOCEPTION_INDICES_2026,
    PROPRIO_WIDTH_2025: PROPRIOCEPTION_INDICES_2025,
}

# Default = the year we train on.
PROPRIOCEPTION_INDICES = PROPRIOCEPTION_INDICES_2026

# Official 23-D action layout (2026 upstream ACTION_QPOS_INDICES). The reduced state
# vector the model consumes mirrors this exactly, which is why the two are compared
# dimension-for-dimension by the delta-action mask.
ACTION_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    ),
}
STATE_ACTION_DIM = 23
