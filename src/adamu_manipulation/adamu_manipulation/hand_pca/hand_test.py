#!/usr/bin/env python3
"""
按键说明（V4.0新增3种姿态，V5.0新增协同导纳控制）：
  基础：
    1 → open      (完全张开)
    2 → relax     (自然放松)
  
  抓握：
    3 → power     (力量抓握)
    4 → 4-hook    (四指钩)
    5 → 5s        (五指握小)
    6 → 5l        (五指握大)
  
  精准：
    7 → 2-pinch   (两指捏)
    8 → tripod    (三指抓)
    9 → lateral   (指尖侧压)
  
  特殊：
    0 → l-hook    (L型钩底)
  
  控制：
    l → 切换左手
    r → 切换右手
    b → 双手同步
    z → 切换控制模式 (原生位置 12D / 协同导纳 4D)  <-- 新增
    h → 显示帮助
    q → 退出
"""

import sys
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import os
import time

# ════════════════════════════════════════════════════════════════════
#  10种典型手部姿态字典
# ════════════════════════════════════════════════════════════════════
JOINT_LIMITS = {
    "min": np.array([0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]),
    "max": np.array([1.1,   0.5,   1.0,   1.2,   1.7,   1.6,   1.7,   1.6,   1.7,   1.6,   1.7,   1.6]),
}

POSE_ANGLES = {
    "open":   np.array([0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.    ]),
    "relax":  np.array([0.154,  0.3075, 0.15,   0.18,   0.255,  0.24,   0.255,  0.24,   0.255,  0.24,   0.255,  0.24  ]),
    "power":  np.array([1.1, 0.0, 1.0, 1.2, 1.7, 1.49, 1.7, 1.27, 1.7, 1.42, 1.7, 1.6]),
    "4hook":  np.array([0.,     0.,     0.,     0.,     1.088,  0.72,   1.122,  0.664,  1.071,  0.688,  0.9945, 0.672 ]),
    "5small": np.array([1.1,    0.29,   0.215,  0.144,  0.578,  0.864,  1.105,  0.328,  0.782,  0.848,  1.02,   0.944 ]),
    "5large": np.array([1.1,    0.0239, 0.0572, 0.0764, 0.0606, 0.7489, 0.3698, 0.6522, 0.3031, 0.6923, 0.2982, 0.5249]),
    "2pinch": np.array([1.1,    0.5,    0.,     0.,     0.561,  0.696,  1.7,    1.6,    1.7,    1.6,    1.7,    1.6   ]),
    "tripod": np.array([1.1,    0.4,    0.,     0.,     0.7395, 0.112,  0.782,  0.088,  1.7,    1.6,    1.7,    1.6   ]),
    "lateral":np.array([0.,     0.,     0.,     0.,     0.,     0.72,   0.,     0.72,   0.,     0.712,  0.,     0.712 ]),
    "lhook":  np.array([0.,     0.,     0.,     0.,     0.,     1.36,   0.,     1.536,  0.,     1.456,  0.,     1.256 ]),
}

for key in POSE_ANGLES:
    POSE_ANGLES[key] = np.clip(POSE_ANGLES[key], JOINT_LIMITS["min"], JOINT_LIMITS["max"])

KEY_TO_POSE = {
    '1': ('open',     '完全张开'), '2': ('relax',    '自然放松'),
    '3': ('power',    '力量抓握'), '4': ('4hook',    '四指钩'),
    '5': ('5small',   '五指握小'), '6': ('5large',   '五指握大'),
    '7': ('2pinch',   '两指捏'),   '8': ('tripod',   '三指抓'),
    '9': ('lateral',  '指尖侧压'), '0': ('lhook',    'L型钩底'),
}

POSE_CATEGORIES = {
    'open': {'cat': '基础', 'icon': '✋'}, 'relax': {'cat': '基础', 'icon': '😌'},
    'power': {'cat': '抓握', 'icon': '✊'}, '4hook': {'cat': '抓握', 'icon': '☞'},
    '5small': {'cat': '抓握', 'icon': '👊'}, '5large': {'cat': '抓握', 'icon': '👐'},
    '2pinch': {'cat': '精准', 'icon': '🤏'}, 'tripod': {'cat': '精准', 'icon': '✌️'},
    'lateral': {'cat': '精准', 'icon': '👌'}, 'lhook': {'cat': '特殊', 'icon': '☝️'},
}

class SimpleTaskPlanner(Node):
    """Adam_U 任务规划节点 V5.0 (支持 Synergy 导纳双模切换)"""

    def __init__(self):
        super().__init__('simple_task_planner_v5')
        self.get_logger().info("🎮 任务规划测试节点 V5.0 已启动")

        # =========================================================
        #  1. 定义发布者 (两套：一套原生12D，一套协同4D)
        # =========================================================
        self._pub_joint = {
            'left':  self.create_publisher(Float64MultiArray, '/left_hand_controller/commands', 10),
            'right': self.create_publisher(Float64MultiArray, '/right_hand_controller/commands', 10),
        }
        
        self._pub_synergy = {
            'left':  self.create_publisher(Float64MultiArray, '/adam/left_synergy_command', 10),
            'right': self.create_publisher(Float64MultiArray, '/adam/right_synergy_command', 10),
        }

        # =========================================================
        #  2. 初始化状态变量
        # =========================================================
        self._active_side  = 'both'
        self._current_pose = {'left': 'relax', 'right': 'relax'}
        self._control_mode = 'synergy'  # 默认启动导纳模式 ('joint' 或 'synergy')

        # =========================================================
        #  3. 加载 PCA 模型和约束
        # =========================================================
        model_path = self.declare_parameter('model_path', 'adam_synergy_v4_model.npz').get_parameter_value().string_value

        self._pca_ready = False
        try:
            npz = np.load(model_path)
            self._joint_limits_min = npz.get('joint_limits_min', JOINT_LIMITS["min"])
            self._joint_limits_max = npz.get('joint_limits_max', JOINT_LIMITS["max"])
            self._pca_mean       = npz['pca_mean']
            self._pca_components = npz['pca_components']
            self._scaler_mean    = npz['scaler_mean']
            self._scaler_scale   = npz['scaler_scale']
            self._n_pc           = self._pca_components.shape[0]
            self._pca_ready      = True
            self.get_logger().info(f"✅ PCA 模型加载成功，支持 {self._n_pc} 维导纳控制。")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 无法加载 PCA 模型 ({e})，强制降级为原生位置模式。")
            self._control_mode = 'joint'
            self._joint_limits_min = JOINT_LIMITS["min"]
            self._joint_limits_max = JOINT_LIMITS["max"]

        # 内部状态记录 (用于插值起点)
        self._current_angles = {'left': POSE_ANGLES['relax'].copy(), 'right': POSE_ANGLES['relax'].copy()}
        if self._pca_ready:
            self._current_z = {
                'left': self._q_to_z(POSE_ANGLES['relax']),
                'right': self._q_to_z(POSE_ANGLES['relax'])
            }

        # =========================================================
        #  4. 启动输入线程
        # =========================================================
        self._print_banner()
        self._input_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------
    # 数学工具：将 12 维物理角度映射为 4 维大脑意图
    # ------------------------------------------------------------
    def _q_to_z(self, q_rad):
        """核心逆映射：q(12D) -> 标准化 -> z(4D)"""
        q_scaled = (q_rad - self._scaler_mean) / (self._scaler_scale + 1e-6)
        z = (q_scaled - self._pca_mean) @ self._pca_components.T
        return np.clip(z, -4.0, 4.0)

    # ------------------------------------------------------------
    # 运动发布逻辑
    # ------------------------------------------------------------
    def _publish_pose(self, side: str, pose_name: str, duration: float = 1.5):
        """发布带有 S 型缓动 (S-Curve) 的控制指令"""
        if pose_name not in POSE_ANGLES: return
        
        target_angles = np.clip(POSE_ANGLES[pose_name], self._joint_limits_min, self._joint_limits_max)
        start_angles = self._current_angles[side].copy()
        
        if self._control_mode == 'synergy' and self._pca_ready:
            target_z = self._q_to_z(target_angles)
            start_z = self._current_z[side].copy()
        
        cat = POSE_CATEGORIES.get(pose_name, {}).get('cat', '?')
        icon = POSE_CATEGORIES.get(pose_name, {}).get('icon', '?')
        side_name = '左手' if side == 'left' else '右手'
        mode_str = "🧬 协同导纳 (4D)" if self._control_mode == 'synergy' else "⚙️ 原生位置 (12D)"
        
        self.get_logger().info(f"  [{side_name}] {icon} 过渡到 {pose_name:8} | 模式: {mode_str}")

        hz = 100  # 100Hz 密集插值
        steps = int(duration * hz)
        msg = Float64MultiArray()

        for i in range(1, steps + 1):
            t = i / float(steps)
            s = (1.0 - np.cos(t * np.pi)) / 2.0  # S型插值系数
            
            if self._control_mode == 'synergy' and self._pca_ready:
                # 🌟 [模式 A] 发布协同意图 z
                interpolated_z = start_z + s * (target_z - start_z)
                msg.data = interpolated_z.tolist()
                self._pub_synergy[side].publish(msg)
            else:
                # 🌟 [模式 B] 发布原生关节角度 q
                interpolated_angles = start_angles + s * (target_angles - start_angles)
                msg.data = interpolated_angles.tolist()
                self._pub_joint[side].publish(msg)
            
            time.sleep(1.0 / hz)

        # 状态更新记忆
        self._current_angles[side] = target_angles.copy()
        self._current_pose[side] = pose_name
        if self._control_mode == 'synergy' and self._pca_ready:
            self._current_z[side] = target_z.copy()

    def _keyboard_loop(self):
        while rclpy.ok():
            try:
                key = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key == 'q':
                self.get_logger().info("👋 退出程序")
                break
            elif key == 'l':
                self._active_side = 'left'
                self.get_logger().info("🤚 控制切换为: 左手")
            elif key == 'r':
                self._active_side = 'right'
                self.get_logger().info("🤚 控制切换为: 右手")
            elif key == 'b':
                self._active_side = 'both'
                self.get_logger().info("🙌 控制切换为: 双手同步")
            elif key == 'z':
                if not self._pca_ready:
                    self.get_logger().error("❌ 未加载 PCA 模型，无法切换到协同模式！")
                    continue
                # 模式切换
                self._control_mode = 'joint' if self._control_mode == 'synergy' else 'synergy'
                mode_str = "🧬 协同导纳模式 (Synergy Z-Space)" if self._control_mode == 'synergy' else "⚙️ 原生位置模式 (Joint Q-Space)"
                self.get_logger().info(f"🔄 模式已切换为: {mode_str}")
            elif key in KEY_TO_POSE:
                pose_name = KEY_TO_POSE[key][0]
                if self._active_side == 'both':
                    self._publish_pose('left', pose_name)
                    self._publish_pose('right', pose_name)
                else:
                    self._publish_pose(self._active_side, pose_name)
            elif key == 'h' or key == 'm':
                self._print_banner()

    def _print_banner(self):
        side_dict = {'left': '左手', 'right': '右手', 'both': '双手'}
        mode_str = "协同导纳(4D)" if self._control_mode == 'synergy' else "原生位置(12D)"
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         Adam_U 任务规划控制台 V5.0 (双模融合版)                ║
║                                                               ║
║  当前控制肢体: {side_dict[self._active_side]:<6}                                  ║
║  当前下发模式: {mode_str:<12} [按 'z' 切换]                ║
╠═══════════════════════════════════════════════════════════════╣
║  📋 姿态切换 (按数字键)           🎮 系统控制                    ║
║  ─────────────────────────────────────────────────────────  ║
║  基础:                            l → 左手                       ║
║    1 → open    完全张开           r → 右手                       ║
║    2 → relax   自然放松           b → 双手同步                   ║
║  抓握:                                                        ║
║    3 → power   力量抓握           z → 切换下发模式 (Joint/Synergy)║
║    4 → 4hook   四指钩             h → 显示本帮助页面             ║
║    5 → 5small  五指握小           q → 退出                       ║
║    6 → 5large  五指握大                                        ║
║  精准:                                                        ║
║    7 → 2pinch  两指捏                                          ║
║    8 → tripod  三指抓                                          ║
║    9 → lateral 指尖侧压                                        ║
║  特殊:                                                        ║
║    0 → lhook   L型钩底                                         ║
╚═══════════════════════════════════════════════════════════════╝
""")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleTaskPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⚠️ 收到中断信号")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()