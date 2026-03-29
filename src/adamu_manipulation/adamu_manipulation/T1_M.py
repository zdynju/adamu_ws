import rclpy
import asyncio
import threading
import math
import numpy as np
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import Pose, TransformStamped
from scipy.spatial.transform import Rotation as R

# 假设你的控制器位于此路径
from adamu_manipulation.arm_controller import AdamuDualArmController

# =============================================================================
# 核心接口：计算手在世界坐标系中的最终位姿
# =============================================================================
def get_hand_world_pose(box_pose_world: Pose, hand_offset_in_box: list, hand_rotation_in_box: np.ndarray) -> Pose:
    """
    计算手在世界坐标系中的最终位姿。
    """
    # 1. 确定箱子的位置与姿态 (T_world_box)
    T_world_box = np.eye(4)
    T_world_box[:3, 3] = [box_pose_world.position.x, 
                          box_pose_world.position.y, 
                          box_pose_world.position.z]
    
    box_quat = [box_pose_world.orientation.x, box_pose_world.orientation.y,
                box_pose_world.orientation.z, box_pose_world.orientation.w]
    T_world_box[:3, :3] = R.from_quat(box_quat).as_matrix()

    # 2. 确定手在箱子上的相对位姿 (T_box_hand)
    T_box_hand = np.eye(4)
    T_box_hand[:3, :3] = hand_rotation_in_box
    T_box_hand[:3, 3] = hand_offset_in_box

    # 3. 变换合成: T_world_hand = T_world_box * T_box_hand
    T_world_hand = np.dot(T_world_box, T_box_hand)

    # 4. 封装回 Pose 消息
    final_pose = Pose()
    final_pose.position.x, final_pose.position.y, final_pose.position.z = T_world_hand[:3, 3]
    q = R.from_matrix(T_world_hand[:3, :3]).as_quat()
    final_pose.orientation.x, final_pose.orientation.y = q[0], q[1]
    final_pose.orientation.z, final_pose.orientation.w = q[2], q[3]
    
    return final_pose

# =============================================================================
# 任务规划节点：负责 TF 监听与调试广播
# =============================================================================
class TaskPlanner(Node):
    def __init__(self):
        super().__init__('t1_m_task_planner')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_poses = {} 
        self.create_timer(0.1, self._publish_debug_tfs) 

    def get_box_pose(self):
        """从 TF 获取箱子的当前 Pose"""
        try:
            trans = self.tf_buffer.lookup_transform('world', 'target_box_fixed', rclpy.time.Time())
            p = Pose()
            p.position.x = trans.transform.translation.x
            p.position.y = trans.transform.translation.y
            p.position.z = trans.transform.translation.z
            p.orientation = trans.transform.rotation
            return p
        except Exception:
            return None

    def set_debug_pose(self, frame_name: str, pose: Pose):
        self.debug_poses[frame_name] = pose

    def _publish_debug_tfs(self):
        for name, pose in self.debug_poses.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = name
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)

# =============================================================================
# 几何解算：根据硬件定义生成左右手 Pose
# =============================================================================
def calc_bimanual_poses_1(box_pose, half_width=0.1, clearance=0.05):
    # 1. 定义左手硬件基准 (X向上, Y手背向左, Z向后)
    # 对应的右手系矩阵投影：
    R_left_base = np.array([
        [ 0,  0, -1], # X(拇指) -> 世界Z
        [ 0,  1,  0], # Y(手背) -> 世界Y
        [ 1,  0,  0]  # Z(手臂) -> 世界-X
    ])

    # 2. 定义右手硬件基准 (X向下, Y手背向右, Z向后)
    R_right_base = np.array([
        [ 0,  0, -1], 
        [ 0, 1,  0], 
        [1,  0,  0]
    ])

    adjust_x = -5.0 
    
    # 构造绕 X 轴旋转的局部矩阵
    R_L_adjust = R.from_euler('x', adjust_x, degrees=True).as_matrix()
    
    # 右手为了保持镜像对称，“向内”通常需要取反符号
    R_R_adjust = R.from_euler('x', -adjust_x, degrees=True).as_matrix()

    # ==========================================
    # 3. 将旋转叠加到基准姿态上
    # ==========================================
    R_left_final = np.dot(R_left_base, R_L_adjust)
    R_right_final = np.dot(R_right_base, R_R_adjust)

    # ---------------- 后续代码保持不变，只需将传入的矩阵改为 final ----------------
    left_pre = get_hand_world_pose(box_pose, [-0.08, half_width + clearance, 0.06], R_left_final)
    right_pre = get_hand_world_pose(box_pose, [-0.08, -half_width - clearance, 0.06], R_right_final)

    left_grasp = get_hand_world_pose(box_pose, [-0.08, half_width-0.01 , 0.06], R_left_final)
    right_grasp = get_hand_world_pose(box_pose, [-0.08, -half_width+0.01 , 0.06], R_right_final)
    # 计算位移矢量用于直线运动
    l_delta = (left_grasp.position.x - left_pre.position.x,
               left_grasp.position.y - left_pre.position.y,
               left_grasp.position.z - left_pre.position.z)
    
    r_delta = (right_grasp.position.x - right_pre.position.x,
               right_grasp.position.y - right_pre.position.y,
               right_grasp.position.z - right_pre.position.z)

    return (left_pre, right_pre), (left_grasp, right_grasp), (l_delta, r_delta)

def calc_bimanual_poses_2(box_pose, half_width=0.13, half_height=0.08, clearance=0.05):
    """
    根据硬件基准 R_base，计算绕 Z 轴旋转后的棱线抓取位姿
    """
    # 1. 硬件基准矩阵 (X向上, Y手背向左, Z手臂向后)
    R_base = np.array([
        [ 0,  0, -1], 
        [ 0,  1,  0], 
        [ 1,  0,  0]
    ])

    # 2. 计算掌心指向中心所需的倾斜角 theta = arctan(h/w)
    theta = np.arctan2(half_height, half_width)

    # --- 左手计算 ---
    # 绕局部 Z 轴顺时针转 theta，使掌心向下斜扣
    R_L_local = R.from_euler('z',  -np.pi/2+theta).as_matrix()
    R_L_final = np.dot(R_base, R_L_local)
    
    # 位置：左上棱点 [0, w, h]
    l_pre = get_hand_world_pose(box_pose, [-0.1, half_width + clearance, half_height + clearance], R_L_final)
    l_grasp = get_hand_world_pose(box_pose, [0.1, half_width, half_height], R_L_final)

    # --- 右手计算 ---
    # 绕局部 Z 轴转 (PI + theta)，实现镜像掌心相对
    R_R_local = R.from_euler('z', theta).as_matrix()
    R_R_final = np.dot(R_base, R_R_local)
    
    # 位置：右上棱点 [0, -w, h]
    r_pre = get_hand_world_pose(box_pose, [0.1, -half_width - clearance-0.01, half_height + clearance+0.01], R_R_final)
    r_grasp = get_hand_world_pose(box_pose, [0.1, -half_width, half_height], R_R_final)

    # 计算直线位移矢量
    l_delta = (l_grasp.position.x - l_pre.position.x,
               l_grasp.position.y - l_pre.position.y,
               l_grasp.position.z - l_pre.position.z)
    
    r_delta = (r_grasp.position.x - r_pre.position.x,
               r_grasp.position.y - r_pre.position.y,
               r_grasp.position.z - r_pre.position.z)

    return (l_pre, r_pre), (l_grasp, r_grasp), (l_delta, r_delta)


# =============================================================================
# 主任务流
# =============================================================================
async def main_task(controller, planner):
    controller.get_logger().info("🚀 开始执行 T1_M 双臂独立 Pose 规划任务...")
    
    if not await controller.wait_for_services(30.0): return

    # 1. 获取箱子位姿
    box_pose = None
    while box_pose is None:         
        box_pose = planner.get_box_pose()
        if box_pose is None: await asyncio.sleep(0.5)

    # 2. 计算位姿
    pres, grasps, deltas = calc_bimanual_poses_1(box_pose)
    l_pre, r_pre = pres
    l_delta, r_delta = deltas

    # 3. 调试显示
    planner.set_debug_pose('l_target_pre', l_pre)
    planner.set_debug_pose('r_target_pre', r_pre)

    # controller.get_logger().info("⏸️ 请在 RViz2 检查 TF 轴向（左手红轴应向上，蓝轴应向后）...")
    # input("👉 检查无误后按回车执行...")

    # 4. 运动执行
    controller.get_logger().info("▶️ 阶段 1: 关节空间移动至预抓取位")
    if await controller.send_dual_arm_goal(l_pre, r_pre):
        await asyncio.sleep(1.0)
        # controller.get_logger().info("▶️ 阶段 2: 笛卡尔直线压紧")
        # await controller.execute_dual_arm_straight_line(l_delta, r_delta)

    # controller.get_logger().info("▶️ 阶段 3: 执行斜向上 45° 同步提拉...")
        
    # pull_dist = 0.15 # 5cm
    # angle_rad = math.radians(45)
    
    # # 计算世界坐标系下的分量
    # dz = pull_dist * math.sin(angle_rad)  # 向上
    # dx = pull_dist * math.cos(angle_rad) # 向后回拉 (假设 -X 是身体方向)
    # dy = 0.01                         # 左右不偏移
    
    # # 【极其关键】双臂必须共享同一个位移矢量！
    # left_lift_delta = (dx, dy, dz)
    # right_lift_delta = (dx,-dy,dz)
    
    # # 调用直线插补运动，保持手部姿态绝对锁定
    # await controller.execute_dual_arm_straight_line(left_lift_delta, right_lift_delta)
    
    # controller.get_logger().info("✅ 提拉 5cm 动作完成！")

def main(args=None):
    rclpy.init(args=args)
    con = AdamuDualArmController()
    pla = TaskPlanner()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(con); executor.add_node(pla)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        asyncio.get_event_loop().run_until_complete(main_task(con, pla))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()