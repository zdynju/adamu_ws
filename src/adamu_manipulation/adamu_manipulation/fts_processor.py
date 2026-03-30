import asyncio
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class FTSProcessor(Node):
    """
    统一的力矩传感器处理节点 (无重映射版)
    负责：数据订阅、去皮(Tare)、重力补偿、低通滤波、100Hz连续发布、指腹FTS管理
    """
    def __init__(self, tool_mass=0.54, filter_alpha=0.1):
        super().__init__('fts_processor_node')
        self.get_logger().info('初始化 FTS 处理器 (包含双臂预处理与指腹传感器)...')
        
        # --- 参数 ---
        self.tool_mass = tool_mass
        self.g = 9.81
        self.alpha = filter_alpha  # 低通滤波系数
        
        # --- 1. 手腕 FTS 变量与锁 ---
        self._left_fts_lock = threading.Lock()
        self._right_fts_lock = threading.Lock()
        self._left_raw_wrench = None
        self._right_raw_wrench = None

        # 滤波器状态与零偏 (1x6 数组: fx, fy, fz, tx, ty, tz)
        self._left_bias = np.zeros(6)
        self._right_bias = np.zeros(6)
        self._left_filtered = np.zeros(6)
        self._right_filtered = np.zeros(6)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- 2. 指腹 FTS 变量与锁 ---
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self._finger_locks = {'L': {}, 'R': {}}
        self._finger_wrenches = {'L': {}, 'R': {}}

        # --- 3. 注册订阅者 (直接对接 Broadcaster 的默认话题) ---
        self.create_subscription(
            WrenchStamped, 
            '/left_fts_broadcaster/wrench', 
            self._left_cb, 
            10
        )
        self.create_subscription(
            WrenchStamped, 
            '/right_fts_broadcaster/wrench', 
            self._right_cb, 
            10
        )

        for side_prefix in ['L', 'R']:
            for finger in self.fingers:
                self._finger_locks[side_prefix][finger] = threading.Lock()
                self._finger_wrenches[side_prefix][finger] = None
                topic_name = f'/{side_prefix}_{finger}_pad_broadcaster/wrench'
                self.create_subscription(
                    WrenchStamped, 
                    topic_name, 
                    lambda msg, s=side_prefix, f=finger: self._finger_cb(msg, s, f), 
                    10
                )

        # --- 4. 注册发布者 (直接发给 Controller 的默认监听话题) ---
        self._left_pub = self.create_publisher(
            WrenchStamped,
            '/left_arm_cartesian_compliance_controller/ft_sensor_wrench', 
            10
        )
        self._right_pub = self.create_publisher(
            WrenchStamped,
            '/right_arm_cartesian_compliance_controller/ft_sensor_wrench', 
            10
        )

        # --- 5. 100Hz 处理定时器 ---
        self._timer = self.create_timer(0.01, self._process_loop)
        self.get_logger().info('WrenchPreprocessor 初始化完成，等待 tare...')

    # ==================== 回调函数 ====================
    def _left_cb(self, msg: WrenchStamped):
        with self._left_fts_lock:
            self._left_raw_wrench = msg

    def _right_cb(self, msg: WrenchStamped):
        with self._right_fts_lock:
            self._right_raw_wrench = msg

    def _finger_cb(self, msg: WrenchStamped, side_prefix: str, finger: str):
        with self._finger_locks[side_prefix][finger]:
            self._finger_wrenches[side_prefix][finger] = msg


    # ==================== 核心处理循环 (100Hz) ====================
    def _process_loop(self):
        self._process_single_arm('left', self._left_raw_wrench, self._left_fts_lock, 
                                 self._left_bias, self._left_filtered, self._left_pub)
        
        self._process_single_arm('right', self._right_raw_wrench, self._right_fts_lock, 
                                 self._right_bias, self._right_filtered, self._right_pub)

    def _process_single_arm(self, side, raw_wrench, lock, bias, filtered_state, publisher):
        with lock:
            raw = raw_wrench
            
        if raw is None:
            return

        # [1] 转成 numpy
        w = np.array([
            raw.wrench.force.x, raw.wrench.force.y, raw.wrench.force.z,
            raw.wrench.torque.x, raw.wrench.torque.y, raw.wrench.torque.z,
        ])

        # [2] 减去 tare 零偏
        w = w - bias

        # [3] 重力补偿 (仅补偿力)
        grav_comp = self._compute_gravity_compensation(side)
        if grav_comp is not None:
            w[:3] -= grav_comp

        # [4] 一阶低通滤波
        new_filtered = self.alpha * w + (1.0 - self.alpha) * filtered_state
        np.copyto(filtered_state, new_filtered)

        # [5] 发布
        out = WrenchStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = f'{side}_hand_fts' 
        
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = filtered_state[0:3]
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = filtered_state[3:6]
        
        publisher.publish(out)

    # ==================== 重力补偿与 Tare ====================
    def _compute_gravity_compensation(self, side: str) -> np.ndarray | None:
        try:
            target_frame = f'{side}_hand_fts'
            transform = self.tf_buffer.lookup_transform(
                target_frame, 'world', rclpy.time.Time()
            )
            gravity_world = Vector3Stamped()
            gravity_world.vector.z = -(self.tool_mass * self.g)

            gravity_sensor = tf2_geometry_msgs.do_transform_vector3(gravity_world, transform)
            return np.array([
                gravity_sensor.vector.x,
                gravity_sensor.vector.y,
                gravity_sensor.vector.z,
            ])
        except Exception as e:
            return None

    def do_tare(self, side: str, n_samples: int = 200) -> bool:
        """
        采集 n_samples 帧计算零偏并记录。
        """
        self.get_logger().info(f'开始 {side} 臂 tare，采集 {n_samples} 帧...')
        samples = []
        
        lock = self._left_fts_lock if side == 'left' else self._right_fts_lock
        
        for _ in range(n_samples):
            with lock:
                raw = self._left_raw_wrench if side == 'left' else self._right_raw_wrench
                
            if raw is None:
                time.sleep(0.01)
                continue

            grav = self._compute_gravity_compensation(side)
            if grav is None:
                time.sleep(0.01)
                continue

            w = np.array([
                raw.wrench.force.x - grav[0],
                raw.wrench.force.y - grav[1],
                raw.wrench.force.z - grav[2],
                raw.wrench.torque.x,
                raw.wrench.torque.y,
                raw.wrench.torque.z,
            ])
            samples.append(w)
            time.sleep(0.01) 

        if len(samples) < n_samples * 0.8:
            self.get_logger().error(f'{side} 臂 tare 失败：有效样本不足')
            return False

        bias = np.mean(samples, axis=0)
        
        if side == 'left':
            self._left_bias = bias
            self._left_filtered = np.zeros(6)
        else:
            self._right_bias = bias
            self._right_filtered = np.zeros(6)
            
        self.get_logger().info(f'{side} 臂 tare 完成，零偏: fx={bias[0]:.3f} fy={bias[1]:.3f} fz={bias[2]:.3f} N')
        return True

    # ==================== 指腹传感器接口 ====================
    def get_all_finger_forces(self, side: str):
        prefix = 'L' if side == 'left' else 'R'
        finger_forces = {}
        for finger in self.fingers:
            with self._finger_locks[prefix][finger]:
                raw = self._finger_wrenches[prefix][finger]
                if raw is not None:
                    finger_forces[finger] = {
                        'force': np.array([raw.wrench.force.x, raw.wrench.force.y, raw.wrench.force.z]),
                        'frame': raw.header.frame_id
                    }
                else:
                    finger_forces[finger] = {
                        'force': np.zeros(3),
                        'frame': f'{prefix}_{finger}_pad_link' 
                    }
        return finger_forces
    # ==================== 对外手腕数据接口 ====================
    def get_raw_wrist_force(self, side: str):
        """
        对外接口：获取手腕 FTS 的原始受力 (Fx, Fy, Fz)
        """
        lock = self._left_fts_lock if side == 'left' else self._right_fts_lock
        
        with lock:
            raw = self._left_raw_wrench if side == 'left' else self._right_raw_wrench
            
        if raw is None:
            return None
            
        return [raw.wrench.force.x, raw.wrench.force.y, raw.wrench.force.z]

    def get_processed_wrist_force(self, side: str):
        """
        对外接口：获取手腕 FTS 处理后的受力 (Fx, Fy, Fz)
        已包含：去皮(Tare) + 重力补偿 + 低通滤波
        """
        lock = self._left_fts_lock if side == 'left' else self._right_fts_lock
        
        with lock:
            raw = self._left_raw_wrench if side == 'left' else self._right_raw_wrench
            # 如果连原始数据都没有，说明还没开始收到传感器信息，返回 None
            if raw is None:
                return None
                
            filtered_state = self._left_filtered if side == 'left' else self._right_filtered
            # 取前三个元素 (力 X, Y, Z) 并转为标准 Python 列表
            return filtered_state[0:3].tolist()