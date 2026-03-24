import asyncio
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3Stamped
import numpy as np

class FTSProcessor(Node):
    """
    统一的力矩传感器处理
    负责：数据订阅、线程锁、去皮(Tare)、坐标重映射
    包含：手腕/手掌 FTS (带重力补偿) + 10个指腹 FTS (原始接触力)
    """
    def __init__(self, mass):
        super().__init__('fts_processor_node')
        self.get_logger().info('初始化 FTS 处理器 (包含指腹传感器)...')
        self.mass = mass
        self.g = 9.8
        
        # --- 1. 手腕 FTS 变量与锁 ---
        self._left_fts_lock = threading.Lock()
        self._right_fts_lock = threading.Lock()
        self._left_wrench = None
        self._right_wrench = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self._left_bias = [0.0] * 6
        self._right_bias = [0.0] * 6
        
        # --- 2. 指腹 FTS 变量与锁 (使用字典统一管理) ---
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        # 结构: {'L': {'thumb': Lock, ...}, 'R': {...}}
        self._finger_locks = {'L': {}, 'R': {}}
        self._finger_wrenches = {'L': {}, 'R': {}}

        # --- 3. 注册订阅者 ---
        # 订阅手腕
        self.create_subscription(WrenchStamped, '/left_fts_broadcaster/wrench', self._left_cb, 10)
        self.create_subscription(WrenchStamped, '/right_fts_broadcaster/wrench', self._right_cb, 10)

        # 批量初始化并订阅10个手指
        for side_prefix in ['L', 'R']:
            for finger in self.fingers:
                self._finger_locks[side_prefix][finger] = threading.Lock()
                self._finger_wrenches[side_prefix][finger] = None
                
                topic_name = f'/{side_prefix}_{finger}_pad_broadcaster/wrench'
                
                # 注意：Python 在 for 循环中使用 lambda 需要捕获变量的值 (s=side_prefix, f=finger)
                self.create_subscription(
                    WrenchStamped, 
                    topic_name, 
                    lambda msg, s=side_prefix, f=finger: self._finger_cb(msg, s, f), 
                    10
                )

    # ==================== 回调函数 ====================
    def _left_cb(self, msg: WrenchStamped):
        with self._left_fts_lock:
            self._left_wrench = msg

    def _right_cb(self, msg: WrenchStamped):
        with self._right_fts_lock:
            self._right_wrench = msg

    def _finger_cb(self, msg: WrenchStamped, side_prefix: str, finger: str):
        """统一的指腹传感器回调"""
        with self._finger_locks[side_prefix][finger]:
            self._finger_wrenches[side_prefix][finger] = msg


    # ==================== 对外统一接口 ====================
    
    def get_original_force(self, side: str):
        """
        获取手腕 FTS 的原始受力 (不带重力补偿)
        """
        raw_wrench = self._left_wrench if side == 'left' else self._right_wrench
        if raw_wrench is None: 
            return None
            
        return [
            raw_wrench.wrench.force.x,
            raw_wrench.wrench.force.y,
            raw_wrench.wrench.force.z
        ]

    def get_standardized_force(self, side: str):
        """
        获取手腕 FTS 经过 TF 重力补偿后的受力 (原标准接口)
        """
        raw_wrench = self._left_wrench if side == 'left' else self._right_wrench
        if raw_wrench is None: 
            return None

        try:
            target_frame = f'{side}_hand_fts'
            transform = self.tf_buffer.lookup_transform(
                target_frame, 'world', rclpy.time.Time()
            )
            
            gravity_world = Vector3Stamped()
            gravity_world.vector.z = - (self.mass * self.g)
            
            gravity_tcp = tf2_geometry_msgs.do_transform_vector3(gravity_world, transform)
            
            clean_fx = raw_wrench.wrench.force.x - gravity_tcp.vector.x
            clean_fy = raw_wrench.wrench.force.y - gravity_tcp.vector.y
            clean_fz = raw_wrench.wrench.force.z - gravity_tcp.vector.z
            
            return [clean_fx, clean_fy, clean_fz]
            
        except Exception as e:
            self.get_logger().warn(f"TF 重力补偿失败: {e}", throttle_duration_sec=2.0)
            return None

    def get_all_finger_forces(self, side: str):
        """
        获取单手所有指腹的接触力 (统一接口)
        🌟 修正点：返回力向量 np.array 的同时，务必携带 frame_id
        """
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
                    # 未收到数据时的安全默认值
                    finger_forces[finger] = {
                        'force': np.zeros(3),
                        'frame': f'{prefix}_{finger}_pad_link' 
                    }
                    
        return finger_forces