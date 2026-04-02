import rclpy
import asyncio
from rclpy.node import Node
from geometry_msgs.msg import Pose
import time

from adamu_manipulation.hand_controller import AdamuHandController
from adamu_manipulation.arm_controller import AdamuDualArmController
from adamu_manipulation.adamu_manipulation.servo_controller import AdamuServoController
from rclpy.executors import SingleThreadedExecutor




async def main_async(arm_ctrl, hand_ctrl, servo_ctrl):
    if not await servo_ctrl.wait_for_services():
        return

    # ── 阶段1 ────────────────────────────────────────────────────────────
    arm_ctrl.get_logger().info('=== 阶段1：移动到预位 ===')
    await hand_ctrl.open_hands()
    

    
    left_pose = Pose()
    right_pose = Pose()










    if not await arm_ctrl.send_dual_arm_goal(left_pose, right_pose):
        return

    # ── 阶段2 ────────────────────────────────────────────────────────────
    arm_ctrl.get_logger().info('=== 阶段2：激活双臂 Servo ===')
    if not await servo_ctrl.activate_both_servo():
        return


    arm_ctrl.get_logger().info("横向接近")
    if not await arm_ctrl.execute_dual_arm_straight_line((0, -0.10, 0), (0, 0.10, 0)):
        arm_ctrl.get_logger().error("横向接近失败！")
        return
    await asyncio.sleep(2.0)
    
    def stop_condition():
        
        return False 
    # ── 阶段4 ────────────────────────────────────────────────────────────
    arm_ctrl.get_logger().info('=== 阶段4：双臂接触逼近 ===')
    if not await servo_ctrl.servo_cartesian('left', vx=0.0, vy=-0.10, vz=0.0, rate_hz=50.0,duration=10.0):
        return
    

def main(args=None):
    rclpy.init(args=args)

    arm_ctrl   = AdamuDualArmController()
    hand_ctrl  = AdamuHandController()
    servo_ctrl = AdamuServoController()

    executor = SingleThreadedExecutor()
    executor.add_node(arm_ctrl)
    executor.add_node(hand_ctrl)
    executor.add_node(servo_ctrl)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(asyncio.gather(
            main_async(arm_ctrl, hand_ctrl, servo_ctrl),
            loop.run_in_executor(None, executor.spin),
        ))
    except KeyboardInterrupt:
        arm_ctrl.get_logger().info('用户中断，安全退出...')
    finally:
        executor.shutdown()
        arm_ctrl.destroy_node()
        hand_ctrl.destroy_node()
        servo_ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()































