import asyncio
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor

from adamu_manipulation.arm_controller import AdamuDualArmController
from adamu_manipulation.simple_hand_controller import DualHandController


def build_right_pregrasp_pose() -> Pose:
    """
    构造“箱子上方直角棱抓取”的右臂预抓取位姿。
    这里给的是一个模板姿态，使用时请按你的场景微调 xyz / 四元数。
    """
    pose = Pose()
    pose.position.x = 0.42
    pose.position.y = -0.18
    pose.position.z = 1.28
    pose.orientation.x = 0.093
    pose.orientation.y = -0.690
    pose.orientation.z = 0.009
    pose.orientation.w = 0.715
    return pose


async def run_edge_top_grasp_demo(
    arm_ctrl: AdamuDualArmController,
    hands: DualHandController,
) -> None:
    """
    演示流程（右手）：
      1) 手张开
      2) MoveIt 到预抓取位姿
      3) 手预成型（edge_grasp）
      4) Pilz 直线贴近 + 下探
      5) 分段闭合
      6) 直线上提
    """
    if not await arm_ctrl.wait_for_services():
        arm_ctrl.get_logger().error('运动服务未就绪，退出演示')
        return

    right_hand = hands.get('right')

    arm_ctrl.get_logger().info('=== [Demo] Step 1: 右手张开 ===')
    await right_hand.open(duration=1.0)
    await asyncio.sleep(0.2)

    arm_ctrl.get_logger().info('=== [Demo] Step 2: 右臂到预抓取位姿 ===')
    pose_pre = build_right_pregrasp_pose()
    if not await arm_ctrl.send_single_arm_goal('right', pose_pre):
        arm_ctrl.get_logger().error('右臂预抓取定位失败')
        return

    arm_ctrl.get_logger().info('=== [Demo] Step 3: 手型预成型（卡棱）===')
    await right_hand.edge_grasp(duration=1.2)
    await asyncio.sleep(0.2)

    arm_ctrl.get_logger().info('=== [Demo] Step 4: 直线贴近箱体侧面 ===')
    # 先沿 +Y（右臂视角）贴近棱边
    if not await arm_ctrl.execute_single_arm_straight_line('right', (0.0, 0.03, 0.0)):
        arm_ctrl.get_logger().error('贴近阶段失败')
        return

    arm_ctrl.get_logger().info('=== [Demo] Step 5: 直线下探至上棱位置 ===')
    if not await arm_ctrl.execute_single_arm_straight_line('right', (0.0, 0.0, -0.02)):
        arm_ctrl.get_logger().error('下探阶段失败')
        return

    arm_ctrl.get_logger().info('=== [Demo] Step 6: 分段闭合（先 80%，再补压）===')
    pre_close = np.array([0.24, 0.40, 0.65, 0.50, 0.95, 0.65, 0.95, 0.65, 0.78, 0.55, 0.65, 0.50])
    full_close = np.array([0.32, 0.52, 0.82, 0.62, 1.22, 0.82, 1.22, 0.82, 1.02, 0.72, 0.84, 0.62])
    await right_hand.set_joints(pre_close, duration=0.8)
    await asyncio.sleep(0.3)
    await right_hand.set_joints(full_close, duration=0.6)

    arm_ctrl.get_logger().info('=== [Demo] Step 7: 轻微上提验证抓稳 ===')
    if not await arm_ctrl.execute_single_arm_straight_line('right', (0.0, 0.0, 0.05)):
        arm_ctrl.get_logger().error('上提失败，建议检查手型和接近偏置')
        return

    arm_ctrl.get_logger().info('✅ [Demo] 直角棱抓取演示流程完成')


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
        loop.run_until_complete(run_edge_top_grasp_demo(arm_ctrl, hands))
    except KeyboardInterrupt:
        arm_ctrl.get_logger().info('用户手动中止 edge_top_grasp_demo')
    finally:
        executor.shutdown()
        hands.left.destroy_node()
        hands.right.destroy_node()
        arm_ctrl.destroy_node()
        loop.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()