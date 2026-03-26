import asyncio
import copy
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor

from adamu_manipulation.arm_controller import AdamuDualArmController
from adamu_manipulation.simple_hand_controller import DualHandController
# left_hand_pose = Pose()

# # 平移 (Translation)
# left_hand_pose.position.x = 0.348
# left_hand_pose.position.y = 0.113
# left_hand_pose.position.z = 1.267

# # 旋转 (Quaternion)
# left_hand_pose.orientation.x = -0.048
# left_hand_pose.orientation.y = -0.593
# left_hand_pose.orientation.z = -0.014
# left_hand_pose.orientation.w = 0.804


# # ==========================================
# # 右手位姿 (Right Hand TCP Pose)
# # ==========================================
# right_hand_pose = Pose()

# # 平移 (Translation)
# right_hand_pose.position.x = 0.349
# right_hand_pose.position.y = -0.119
# right_hand_pose.position.z = 1.266

# # 旋转 (Quaternion)
# right_hand_pose.orientation.x = 0.038
# right_hand_pose.orientation.y = -0.593
# right_hand_pose.orientation.z = 0.026
# right_hand_pose.orientation.w = 0.804

def make_pose(
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float
) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """
    使用 RPY 构造四元数，便于明确表达“手指朝下”姿态。
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def offset_pose(src: Pose, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Pose:
    out = copy.deepcopy(src)
    out.position.x += dx
    out.position.y += dy
    out.position.z += dz
    return out
# 0.401, 0.138, 1.278
# -0.010, -0.687, -0.180, 0.703

# 0.416, -0.165, 1.281
# -0.016, -0.650, 0.186, 0.736
async def run_top_ridge_grasp_basic(
    arm_ctrl: AdamuDualArmController,
    hands: DualHandController,
) -> None:
    """
    仅使用“基础关节规划接口（send_single_arm_goal）”的上棱抓取流程。
    不使用 Pilz 直线规划。
    """
    if not await arm_ctrl.wait_for_services():
        arm_ctrl.get_logger().error('运动服务未就绪，退出')
        return
    left_hand_pose = Pose()

    # 平移 (Translation)
    left_hand_pose.position.x = 0.348
    left_hand_pose.position.y = 0.113
    left_hand_pose.position.z = 1.267

    #
    left_hand_pose.orientation.x = -0.048
    left_hand_pose.orientation.y = -0.593
    left_hand_pose.orientation.z = -0.014
    left_hand_pose.orientation.w = 0.804


    # ==========================================
    # 右手位姿 (Right Hand TCP Pose)
    # ==========================================
    right_hand_pose = Pose()

    # 平移 (Translation)
    right_hand_pose.position.x = 0.349
    right_hand_pose.position.y = -0.119
    right_hand_pose.position.z = 1.266

    # 旋转 (Quaternion)
    right_hand_pose.orientation.x = 0.038
    right_hand_pose.orientation.y = -0.593
    right_hand_pose.orientation.z = 0.026
    right_hand_pose.orientation.w = 0.804
    right_hand = hands.get('right')

    # # Step 1) 预抓取手型：先张开
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 1: 右手张开 ===')
    # await right_hand.open(duration=1.0)
    # await asyncio.sleep(0.2)

    # # Step 2) 到达上棱上方预抓取位姿（请按你的工位标定微调）
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 2: 到达上方预抓取位姿 ===')
    # # 手指朝下：默认让工具轴沿 -Z 方向（可根据你末端坐标定义微调 yaw）
    # # qx, qy, qz, qw = quaternion_from_rpy(0.0, math.pi, 0.0)
    # pregrasp = make_pose(0.35, -0.2, 1.281, -0.016, -0.650, 0.186, 0.736)
    # if not await arm_ctrl.send_single_arm_goal('right', pregrasp):
    #     arm_ctrl.get_logger().error('预抓取位姿失败')
    #     return
    # pregrasp = make_pose(0.35, 0.138, 1.281, -0.010, -0.687, -0.18, 0.703)
    # if not await arm_ctrl.send_single_arm_goal('left', pregrasp):
    #     arm_ctrl.get_logger().error('预抓取位姿失败')
    #     return
    if not await arm_ctrl.send_dual_arm_goal(left_hand_pose,right_hand_pose):
        arm_ctrl.get_logger().error('失败')
        return
    left_hand_pose_delta = Pose()

    # 平移 (Translation)
    left_hand_pose_delta.position.x = 0.348
    left_hand_pose_delta.position.y = 0.1
    left_hand_pose_delta.position.z = 1.3

    # 旋转 (Quaternion)
    left_hand_pose_delta.orientation.x = -0.048
    left_hand_pose_delta.orientation.y = -0.593
    left_hand_pose_delta.orientation.z = -0.014
    left_hand_pose_delta.orientation.w = 0.804


    # ==========================================
    # 右手位姿 (Right Hand TCP Pose)
    # ==========================================
    right_hand_pose_delta = Pose()

    # 平移 (Translation)
    right_hand_pose_delta.position.x = 0.349
    right_hand_pose_delta.position.y = -0.125
    right_hand_pose_delta.position.z = 1.2

    # 旋转 (Quaternion)
    right_hand_pose_delta.orientation.x = 0.038
    right_hand_pose_delta.orientation.y = -0.593
    right_hand_pose_delta.orientation.z = 0.026
    right_hand_pose_delta.orientation.w = 0.804
    if not await arm_ctrl.send_dual_arm_goal(left_hand_pose_delta,right_hand_pose_delta):
        arm_ctrl.get_logger().error('失败')
        return
    
    # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((0.0, 0.0, 0.1), (0.0, 0.0, 0.05)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return
    # (Step 3) 手型预成型（卡上棱）
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 3: 手型预成型 ===')
    # await right_hand.edge_grasp(duration=1.2)
    # await asyncio.sleep(0.2)

    # # Step 4) 仅用基础规划做“上棱方向微调”
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 4: 上棱方向微调 ===')
    # align_pose = offset_pose(pregrasp, dx=0.015, dy=0.0, dz=0.0)
    # if not await arm_ctrl.send_single_arm_goal('right', align_pose):
    #     arm_ctrl.get_logger().error('上棱方向微调失败')
    #     return

    # # Step 5) 仅用基础规划做“上方下探”
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 5: 上方下探到抓取高度 ===')
    # descend_pose = offset_pose(align_pose, dx=0.0, dy=0.0, dz=-0.035)
    # if not await arm_ctrl.send_single_arm_goal('right', descend_pose):
    #     arm_ctrl.get_logger().error('上方下探失败')
    #     return

    # # Step 6) 分段闭合（先预压再闭合）
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 6: 分段闭合 ===')
    # pre_close = np.array([0.24, 0.40, 0.65, 0.50, 0.95, 0.65, 0.95, 0.65, 0.78, 0.55, 0.65, 0.50])
    # full_close = np.array([0.32, 0.52, 0.82, 0.62, 1.22, 0.82, 1.22, 0.82, 1.02, 0.72, 0.84, 0.62])
    # await right_hand.set_joints(pre_close, duration=0.8)
    # await asyncio.sleep(0.3)
    # await right_hand.set_joints(full_close, duration=0.6)

    # # Step 7) 仅用基础规划上提验证抓稳
    # arm_ctrl.get_logger().info('=== [Basic Top Ridge] Step 7: 上提验证 ===')
    # lift_pose = offset_pose(descend_pose, dz=0.05)
    # if not await arm_ctrl.send_single_arm_goal('right', lift_pose):
    #     arm_ctrl.get_logger().error('上提失败')
    #     return

    arm_ctrl.get_logger().info('✅ [Basic Top Ridge] 抓取流程完成（基础规划版本）')


def main(args=None):
    rclpy.init(args=args)

    arm_ctrl = AdamuDualArmController()
    hands = DualHandController()

    executor = MultiThreadedExecutor()
    executor.add_node(arm_ctrl)
    executor.add_node(hands.left)
    executor.add_node(hands.right)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        loop.run_until_complete(run_top_ridge_grasp_basic(arm_ctrl, hands))
    except KeyboardInterrupt:
        arm_ctrl.get_logger().info('用户手动中止 top_ridge_grasp_basic')
    finally:
        executor.shutdown()
        hands.left.destroy_node()
        hands.right.destroy_node()
        arm_ctrl.destroy_node()
        loop.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
