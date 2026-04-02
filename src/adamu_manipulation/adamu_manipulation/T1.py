import rclpy
import asyncio
import threading
from rclpy.node import Node
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor

# 假设这些是你自己的模块
from adamu_manipulation.arm_controller import AdamuDualArmController
from adamu_manipulation.fts_processor import FTSProcessor
# 引入我们上一轮写好的 Servo 控制器
async def main_async(arm_ctrl, fts_prc, servo_ctrl):
    # ── 等待所有底层服务就绪 ──────────────────────────────────────────────
    if not await arm_ctrl.wait_for_services():
        return
    if not await servo_ctrl.wait_for_services():
        return

    # ── 阶段1：MoveGroup 移动到预抓取位置 (开环大范围运动) ────────────────
    arm_ctrl.get_logger().info('=== 阶段1：MoveGroup 定位到箱子两侧 ===')
    # pose_left = Pose()
    # pose_right = Pose()

    # # ⬅️ 左 (Left)
    # pose_left.position.x = 0.35
    # pose_left.position.y = 0.22
    # pose_left.position.z = 1.28
    # pose_left.orientation.x = -0.088
    # pose_left.orientation.y = -0.673
    # pose_left.orientation.z = -0.003
    # pose_left.orientation.w = 0.734

    # #➡️ 右 (Right)
    # pose_right.position.x = 0.35
    # pose_right.position.y = -0.22
    # pose_right.position.z = 1.28
    # pose_right.orientation.x = 0.093
    # pose_right.orientation.y = -0.690
    # pose_right.orientation.z = 0.009
    # pose_right.orientation.w = 0.715

    pose_left = Pose()
    pose_right = Pose()

    # ⬅️ 左 (Left)
    pose_left.position.x = 0.3
    pose_left.position.y = 0.3
    pose_left.position.z = 1.28
    pose_left.orientation.x = 0.114
    pose_left.orientation.y = -0.738
    pose_left.orientation.z = -0.424
    pose_left.orientation.w = 0.512

    # ➡️ 右 (Right)
    pose_right.position.x = 0.34
    pose_right.position.y = -0.25
    pose_right.position.z = 1.15
    pose_right.orientation.x = 0.149
    pose_right.orientation.y = -0.584
    pose_right.orientation.z = -0.217
    pose_right.orientation.w = 0.768
# 0.289, 0.089, 1.291


#0.114, -0.738, -0.424, 0.512
# 0.353, -0.111, 1.276

#0.149, -0.584, -0.217, 0.768
#-0.128, -0.732, 0.200, 0.639
    # 执行全局规划
    if not await arm_ctrl.send_dual_arm_goal(pose_left, pose_right):
        arm_ctrl.get_logger().error('预抓取定位失败！')
        return
    arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    if not await arm_ctrl.execute_dual_arm_straight_line((0, -0.12, 0), (0, 0.13, 0)):
        arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
        return
    pose_left = Pose()
    pose_right = Pose()

    # # ⬅️ 左 (Left)
    pose_left.position.x = 0.3
    pose_left.position.y = 0.13
    pose_left.position.z = 1.3
    pose_left.orientation.x = 0.114
    pose_left.orientation.y = -0.738
    pose_left.orientation.z = -0.424
    pose_left.orientation.w = 0.512

    # ➡️ 右 (Right)
    pose_right.position.x = 0.34
    pose_right.position.y = -0.13
    pose_right.position.z = 1.15
    pose_right.orientation.x = 0.149
    pose_right.orientation.y = -0.584
    pose_right.orientation.z = -0.217
    pose_right.orientation.w = 0.768
    if not await arm_ctrl.send_dual_arm_goal(pose_left, pose_right):
        arm_ctrl.get_logger().error('预抓取定位失败！')
        return
    # pose_left = Pose()
    # pose_right = Pose()
    # pose_left.position.x = 0.3
    # pose_left.position.y = 0.3
    # pose_left.position.z = 1.2
    # pose_left.orientation.x = 0.114
    # pose_left.orientation.y = -0.738
    # pose_left.orientation.z = -0.424
    # pose_left.orientation.w = 0.512

    # # # ➡️ 右 (Right)
    # pose_right.position.x = 0.353
    # pose_right.position.y = 0.05
    # pose_right.position.z = 1.36
    # pose_right.orientation.x = 0.063
    # pose_right.orientation.y = -0.689
    # pose_right.orientation.z = 0.005
    # pose_right.orientation.w = 0.772
    # if not await arm_ctrl.send_dual_arm_goal(pose_left, pose_right):
    #     arm_ctrl.get_logger().error('预抓取定位失败！')
    #     return
    # pose_left = Pose()
    # pose_right = Pose()
    # pose_left.position.x = 0.3
    # pose_left.position.y = 0.3
    # pose_left.position.z = 1.2
    # pose_left.orientation.x = 0.114
    # pose_left.orientation.y = -0.738
    # pose_left.orientation.z = -0.424
    # pose_left.orientation.w = 0.512

    # # # ➡️ 右 (Right)
    # pose_right.position.x = 0.353
    # pose_right.position.y = 0.05
    # pose_right.position.z = 1.36
    # pose_right.orientation.x = 0.063
    # pose_right.orientation.y = -0.689
    # pose_right.orientation.z = 0.005
    # pose_right.orientation.w = 0.772
    # if not await arm_ctrl.send_dual_arm_goal(pose_left, pose_right):
    #     arm_ctrl.get_logger().error('预抓取定位失败！')
    #     return
    # pose_left = Pose()
    # pose_right = Pose()
    # pose_left.position.x = 0.3
    # pose_left.position.y = 0.3
    # pose_left.position.z = 1.4
    # pose_left.orientation.x = 0.114
    # pose_left.orientation.y = -0.738
    # pose_left.orientation.z = -0.424
    # pose_left.orientation.w = 0.512

    # # # ➡️ 右 (Right)
    # pose_right.position.x = 0.353
    # pose_right.position.y = 0.05
    # pose_right.position.z = 1.36
    # pose_right.orientation.x = 0.063
    # pose_right.orientation.y = -0.689
    # pose_right.orientation.z = 0.005
    # pose_right.orientation.w = 0.772
    # if not await arm_ctrl.send_dual_arm_goal(pose_left, pose_right):
    #     arm_ctrl.get_logger().error('预抓取定位失败！')
    #     return
    # # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((0, 0.05, -0.05), (0, 0.05, 0.05)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return
    # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((0.11, -0.05, 0.1), (0.11, 0.05, 0.1)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return

    # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((0.03, 0, -0.05), (0.03, 0, -0.05)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return

    # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((0.0, 0.05, 0), (0., -0.05, 0)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return

    # arm_ctrl.get_logger().info('=== 阶段2：笛卡尔直线接近 ===')
    # if not await arm_ctrl.execute_dual_arm_straight_line((-0.02, -0.0, -0.0), (-0.02, 0.0, -0.00)):
    #     arm_ctrl.get_logger().error('笛卡尔直线接近失败！')
    #     return

  
    # ppa = PickParams(
    #      search_step=0.005,
    #         contact_force_threshold=5.0,
    #         target_grasp_force=20.0,
    #         max_grasp_force=40.0,
    #         buildup_step=0.001,
    #         lift_height=0.10,
    # )
    # pph = PickPhase( arm=arm_ctrl,
    #                 fts = fts_prc,
    #                 params=ppa,
    # )

    # success = await pph.run(pose_left,pose_right )
 
    # if success:
    #     arm_ctrl.get_logger().info('箱子已搬起，可以进入下一阶段')
    # else:
    #     arm_ctrl.get_logger().error('夹取失败，检查箱子位置和传感器')

    # # ── 阶段2：切换至 Servo 进行力控夹取 ──────────────────────────────────
    # arm_ctrl.get_logger().info('=== 阶段2：激活 Servo，开始力控夹取 ===')
    # if not await servo_ctrl.activate_both_servo():
    #     return
    # cpa = ContactParams()
    # cp = ContactPhase(ctrl = servo_ctrl,params=cpa)
    
    # success = await cp.run()
    # if not success:
    #     return 


def main(args=None):
    rclpy.init(args=args)
    
    # 实例化节点
    arm_ctrl = AdamuDualArmController()
    fts_prc = FTSProcessor(mass=0.54)
    servo_ctrl = AdamuServoController(fts_processor=fts_prc) # 传入 fts 实例
    
    # 使用多线程执行器，把所有节点都塞进去
    executor = MultiThreadedExecutor()
    executor.add_node(arm_ctrl)
    executor.add_node(fts_prc)
    executor.add_node(servo_ctrl)
    
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 在独立线程中运行 executor（保障回调和话题通信顺畅）
        def run_executor():
            executor.spin()
        
        executor_thread = threading.Thread(target=run_executor, daemon=True)
        executor_thread.start()
        
        # 在主线程运行业务流程
        loop.run_until_complete(main_async(arm_ctrl, fts_prc, servo_ctrl))
        
    except KeyboardInterrupt:
        arm_ctrl.get_logger().info('用户手动中止执行')
    finally:
        executor.shutdown()
        loop.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()