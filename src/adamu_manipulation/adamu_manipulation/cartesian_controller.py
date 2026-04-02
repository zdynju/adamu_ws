#!/usr/bin/env python3
"""
双臂协作夹取提升控制器 (终极完美版)
核心机制：
1. 双重锁定：稳定阶段严格锁定双臂绝对位置，彻底消除"微小退让"引发的反馈漂移。
2. 绝对刚性：右臂(Master)作为叹息之墙绝不后退，迫使左臂(Slave)建立真实夹取力。
3. 时序闭环：提升阶段右臂按时间生成轨迹，左臂严格按右臂实际反馈+初始高度差进行跟随。
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from controller_manager_msgs.srv import SwitchController
import time
import copy
from enum import Enum
from dataclasses import dataclass

class GraspState(Enum):
    IDLE = 0
    STABILIZING_FORCE = 1  # 建立稳定夹取力 (双臂严格位置锁定)
    LIFTING = 2            # 闭环同步提升 (左臂姿态锁定+高度跟随)
    COMPLETED = 3
    ERROR = 4

@dataclass
class ControlConfig:
    # ---------------- 力控参数 ----------------
    left_target_force_y: float = -20.0  # N (目标力)
    force_tolerance: float = 9.0  # N
    force_stabilize_time: float = 1.5   # s (需要保持稳定的时间)
    force_stabilize_timeout: float = 15.0 # s (超时保护)
    
    # ---------------- 运动参数 ----------------
    lift_height: float = 0.25           # m
    lift_velocity: float = 0.02         # m/s
    
    # ---------------- 安全阈值 ----------------
    max_force: float = 60.0             # N
    min_force_drop: float = 2.0         # N (掉落检测：力低于此值认为物体脱落)
    
    # ---------------- 硬件接口 ----------------
    control_rate: float = 100.0         # Hz
    left_ee_frame: str = "left_hand_tcp" # 必须与YAML配置一致

class DualArmSyncController(Node):
    def __init__(self):
        super().__init__('dual_arm_sync_controller')
        
        self.config = ControlConfig()
        self.state = GraspState.IDLE
        
        # 实时数据缓存
        self.current_left_wrench = None
        self.current_right_pose = None
        self.current_left_pose = None
        
        # 计时器与状态记录
        self.force_stable_start_time = None
        self.stabilize_phase_start_time = None  
        self.lift_start_time = None             
        self.lift_start_z = None
        self.initial_z_offset = None       
        
        # --- 🛡️ 核心锁定锚点变量 ---
        self.left_stabilize_target = None   # 稳定期左臂位置锁
        self.right_stabilize_target = None  # 稳定期右臂位置锁 (解决无限后退的核心)
        self.master_ideal_pose = None       # 提升期右臂轨迹锚点
        self.left_initial_pose = None       # 提升期左臂姿态锁

        # ================= 1. 发布器 =================
        self.left_pose_pub = self.create_publisher(
            PoseStamped, '/left_arm_cartesian_compliance_controller/target_frame', 10)
        self.left_wrench_pub = self.create_publisher(
            WrenchStamped, '/left_arm_cartesian_compliance_controller/target_wrench', 10)
        self.right_pose_pub = self.create_publisher(
            PoseStamped, '/right_arm_cartesian_motion_controller/target_frame', 10)

        # ================= 2. 订阅器 =================
        self.create_subscription(
            WrenchStamped, '/left_arm_cartesian_compliance_controller/ft_sensor_wrench', 
            self._wrench_cb, 10)
        self.create_subscription(
            PoseStamped, '/left_arm_cartesian_compliance_controller/current_pose', 
            self._left_pose_cb, 10)
        self.create_subscription(
            PoseStamped, '/right_arm_cartesian_motion_controller/current_pose', 
            self._right_pose_cb, 10)

        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.timer = self.create_timer(1.0 / self.config.control_rate, self.control_loop)
        self.get_logger().info('>>> 同步提升控制器已就绪 (终极完美版)')

    def _wrench_cb(self, msg): self.current_left_wrench = msg
    def _left_pose_cb(self, msg): self.current_left_pose = msg
    def _right_pose_cb(self, msg): self.current_right_pose = msg

    def control_loop(self):
        if self.state == GraspState.IDLE: return
        if not self._all_data_received(): return

        if self.state == GraspState.STABILIZING_FORCE:
            self._stabilize_logic()
        elif self.state == GraspState.LIFTING:
            self._sync_lift_logic()
        elif self.state == GraspState.ERROR:
            self._handle_error()
        elif self.state == GraspState.COMPLETED:
            self._handle_completed()

    def _all_data_received(self):
        if any(v is None for v in [self.current_left_wrench, self.current_left_pose, self.current_right_pose]):
            return False
        return True

    def start_task(self):
        """开始任务时，抓取双臂的当前绝对位置作为第一阶段的死锁目标"""
        if not self._all_data_received():
            self.get_logger().error('❌ 数据未就绪，无法开始任务')
            return
        
        # 🔑 终极修复：同时死锁左臂和右臂的初始位置！
        self.left_stabilize_target = copy.deepcopy(self.current_left_pose)
        self.right_stabilize_target = copy.deepcopy(self.current_right_pose)
        self.stabilize_phase_start_time = time.time()
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('🛡️ 已开启双臂位置死锁，开始建立夹取力...')
        self.get_logger().info(f'   右臂(Master)绝对锁定Y: {self.right_stabilize_target.pose.position.y:.4f}')
        self.get_logger().info(f'   目标夹取力大小: {abs(self.config.left_target_force_y):.1f}N')
        self.get_logger().info('=' * 60)
        
        self.state = GraspState.STABILIZING_FORCE

    def _stabilize_logic(self):
        """阶段 1: 建立稳定的夹持力 (双臂绝对静止，仅左臂Y轴开启力控)"""
        now = self.get_clock().now().to_msg()
        
        # 1. 维持左臂锁定位置 (Y轴会被力控覆盖，其他轴保持刚性)
        self.left_stabilize_target.header.stamp = now
        self.left_pose_pub.publish(self.left_stabilize_target)
        
        # 2. 🛑 核心修复：右臂死死咬住刚开始的锁定位置，绝不使用反馈更新！
        right_hold = copy.deepcopy(self.right_stabilize_target)
        right_hold.header.stamp = now
        self.right_pose_pub.publish(right_hold)

        # 3. 发布目标力
        wrench = WrenchStamped()
        wrench.header.stamp = now
        wrench.header.frame_id = self.config.left_ee_frame
        wrench.wrench.force.y = self.config.left_target_force_y
        self.left_wrench_pub.publish(wrench)

        # 4. 力误差计算 (采用绝对量级相减，完美规避不同坐标系下的正负号Bug)
        curr_f = self.current_left_wrench.wrench.force.y
        force_error = abs(abs(curr_f) - abs(self.config.left_target_force_y))
        
        if int(time.time() * 10) % 10 == 0:
            self.get_logger().info(
                f'🔧 发力中: 传感器读数={curr_f:.2f}N | 量级误差={force_error:.2f}N'
            )
        
        # 超时检测
        if time.time() - self.stabilize_phase_start_time > self.config.force_stabilize_timeout:
            self.get_logger().error(f'❌ 夹取失败：在 {self.config.force_stabilize_timeout}s 内无法达到目标力！')
            self.state = GraspState.ERROR
            return
        
        # 力稳定判定
        if force_error < self.config.force_tolerance:
            if self.force_stable_start_time is None:
                self.force_stable_start_time = time.time()
                self.get_logger().info(f'⏱️ 力达标，保持稳定 {self.config.force_stabilize_time}s...')
            elif time.time() - self.force_stable_start_time > self.config.force_stabilize_time:
                # 🔑 阶段切换：记录提升阶段需要的各项锚点参数
                self.get_logger().info('=' * 60)
                self.get_logger().info('✅ 夹取力彻底稳定，开启闭环同步提升！')
                
                self.left_initial_pose = copy.deepcopy(self.current_left_pose)
                self.master_ideal_pose = copy.deepcopy(self.current_right_pose)
                self.lift_start_z = self.current_right_pose.pose.position.z
                self.lift_start_time = time.time()
                self.initial_z_offset = self.left_initial_pose.pose.position.z - self.lift_start_z
                
                self.state = GraspState.LIFTING
        else:
            self.force_stable_start_time = None

    def _sync_lift_logic(self):
        """阶段 2: 闭环同步提升 (基于时间生成轨迹，左臂严格跟随)"""
        now = self.get_clock().now().to_msg()
        curr_f = abs(self.current_left_wrench.wrench.force.y)
        
        # 1. 安全监控
        if curr_f > self.config.max_force:
            self.get_logger().error(f'❌ 过载断开：力达到 {curr_f:.1f}N')
            self.state = GraspState.ERROR
            return
        if curr_f < self.config.min_force_drop:
            self.get_logger().error(f'❌ 物体滑脱：力掉至 {curr_f:.1f}N')
            self.state = GraspState.ERROR
            return
        
        # 2. 时间基准轨迹生成
        elapsed_time = time.time() - self.lift_start_time
        planned_lift = min(self.config.lift_velocity * elapsed_time, self.config.lift_height)
        target_z = self.lift_start_z + planned_lift
        
        actual_master_z = self.current_right_pose.pose.position.z
        actual_lift = actual_master_z - self.lift_start_z
        
        if int(time.time() * 2) % 2 == 0:
            self.get_logger().info(f'📈 提升中: {actual_lift*1000:.0f}mm / {self.config.lift_height*1000:.0f}mm | 夹持力: {curr_f:.1f}N')
        
        if actual_lift >= self.config.lift_height:
            self.get_logger().info('🎉 提升到达目标高度，任务圆满完成！')
            self.state = GraspState.COMPLETED
            return

        # 3. Master(右臂) 下发理想轨迹
        self.master_ideal_pose.header.stamp = now
        self.master_ideal_pose.pose.position.z = target_z
        self.right_pose_pub.publish(self.master_ideal_pose)

        # 4. Slave(左臂) 严格对齐实际高度并维持初始姿态
        slave_target_pose = copy.deepcopy(self.left_initial_pose)
        slave_target_pose.header.stamp = now
        slave_target_pose.pose.position.z = actual_master_z + self.initial_z_offset
        self.left_pose_pub.publish(slave_target_pose)

        # 5. 持续下发目标力
        wrench = WrenchStamped()
        wrench.header.stamp = now
        wrench.header.frame_id = self.config.left_ee_frame
        wrench.wrench.force.y = self.config.left_target_force_y
        self.left_wrench_pub.publish(wrench)

    def _handle_error(self):
        """发生错误时，紧急制动并保持当前绝对位置"""
        now = self.get_clock().now().to_msg()
        if self.current_left_pose and self.current_right_pose:
            left_hold = copy.deepcopy(self.current_left_pose)
            left_hold.header.stamp = now
            self.left_pose_pub.publish(left_hold)
            
            right_hold = copy.deepcopy(self.current_right_pose)
            right_hold.header.stamp = now
            self.right_pose_pub.publish(right_hold)
            
            # 撤销力指令
            wrench = WrenchStamped()
            wrench.header.stamp = now
            wrench.header.frame_id = self.config.left_ee_frame
            wrench.wrench.force.y = 0.0
            self.left_wrench_pub.publish(wrench)

    def _handle_completed(self):
        """完成后死锁当前位置"""
        now = self.get_clock().now().to_msg()
        if self.current_left_pose and self.current_right_pose:
            left_hold = copy.deepcopy(self.current_left_pose)
            left_hold.header.stamp = now
            self.left_pose_pub.publish(left_hold)
            
            right_hold = copy.deepcopy(self.current_right_pose)
            right_hold.header.stamp = now
            self.right_pose_pub.publish(right_hold)
            
            # 依然保持力夹取防止掉落
            wrench = WrenchStamped()
            wrench.header.stamp = now
            wrench.header.frame_id = self.config.left_ee_frame
            wrench.wrench.force.y = self.config.left_target_force_y
            self.left_wrench_pub.publish(wrench)

    def switch_mode(self):
        req = SwitchController.Request()
        req.activate_controllers = [
            'left_arm_cartesian_compliance_controller',
            'right_arm_cartesian_motion_controller'
        ]
        req.deactivate_controllers = ['left_arm_controller', 'right_arm_controller']
        req.strictness = SwitchController.Request.STRICT
        return self.switch_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = DualArmSyncController()
    
    print("\n" + "=" * 60)
    print(" 🚀 双臂协作夹取提升系统启动 (终极锁定位姿版)")
    print("=" * 60)
    print("正在接管底层控制器...")
    
    future = node.switch_mode()
    rclpy.spin_until_future_complete(node, future)
    
    if future.result().ok:
        node.get_logger().info('✅ 控制器接管成功！')
        input("\n⚠️  请确认机械臂与物体已就位。按【回车键】立即执行夹取任务...")
        node.start_task()
    else:
        node.get_logger().error('❌ 控制器接管失败，请检查 controller_manager')
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n⛔ 收到中断信号，正在紧急制动...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()