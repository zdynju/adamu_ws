#!/usr/bin/env python3
"""
简易双臂维持控制器
右臂：锁定当前位姿（位置控制）
左臂：锁定当前位姿 + Y方向施加 -5N 恒力（柔顺力控）
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
import copy

class SimpleMaintainController(Node):
    def __init__(self):
        super().__init__('simple_maintain_controller')

        # 状态变量：用于存储刚启动时抓取到的“初始位姿”
        self.locked_right_pose = None
        self.locked_left_pose = None

        # 实时回调变量
        self.current_right_pose = None
        self.current_left_pose = None

        # ================= 订阅器 =================
        self.right_pose_sub = self.create_subscription(
            PoseStamped,
            '/right_arm_cartesian_motion_controller/current_pose',
            self.right_pose_callback,
            10
        )
        self.left_pose_sub = self.create_subscription(
            PoseStamped,
            '/left_arm_cartesian_compliance_controller/current_pose',
            self.left_pose_callback,
            10
        )

        # ================= 发布器 =================
        self.right_pose_pub = self.create_publisher(
            PoseStamped,
            '/right_arm_cartesian_motion_controller/target_frame',
            10
        )
        self.left_pose_pub = self.create_publisher(
            PoseStamped,
            '/left_arm_cartesian_compliance_controller/target_frame',
            10
        )
        self.left_wrench_pub = self.create_publisher(
            WrenchStamped,
            '/left_arm_cartesian_compliance_controller/target_wrench',
            10
        )

        # 控制循环定时器 (100 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info('简易维持节点已启动，正在获取当前位姿...')

    def right_pose_callback(self, msg: PoseStamped):
        self.current_right_pose = msg

    def left_pose_callback(self, msg: PoseStamped):
        self.current_left_pose = msg

    def control_loop(self):
        # 1. 如果还没获取到传感器数据，直接返回等待
        if self.current_right_pose is None or self.current_left_pose is None:
            return

        # 2. 第一次进入控制循环时，锁定当前位姿作为永久目标
        if self.locked_right_pose is None:
            self.locked_right_pose = copy.deepcopy(self.current_right_pose)
            self.locked_left_pose = copy.deepcopy(self.current_left_pose)
            self.get_logger().info('✅ 成功锁定双臂初始位姿！开始输出 -5N 保持力...')

        # 3. 发布右臂目标位姿（保持锁定位置）
        self.locked_right_pose.header.stamp = self.get_clock().now().to_msg()
        self.right_pose_pub.publish(self.locked_right_pose)

        # 4. 发布左臂目标位姿（保持锁定位置，抵抗其他方向的漂移）
        self.locked_left_pose.header.stamp = self.get_clock().now().to_msg()
        self.left_pose_pub.publish(self.locked_left_pose)

        # 5. 发布左臂目标力（Y轴 -5N）
        target_wrench = WrenchStamped()
        target_wrench.header.stamp = self.get_clock().now().to_msg()
        target_wrench.header.frame_id = 'left_hand_tcp'  # 确保与你的末端连杆名称一致
        target_wrench.wrench.force.y = -5.0  # Y轴施加 -5N 的恒力

        
        # 保持其他方向力/力矩为0
        target_wrench.wrench.force.x = 0.0
        target_wrench.wrench.force.z = 0.0
        target_wrench.wrench.torque.x = 0.0
        target_wrench.wrench.torque.y = 0.0
        target_wrench.wrench.torque.z = 0.0
        
        self.left_wrench_pub.publish(target_wrench)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMaintainController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n维持任务终止。")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()