import rclpy
import asyncio
import threading
import math
import numpy as np
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import Pose, TransformStamped
from scipy.spatial.transform import Rotation as R
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents, AllowedCollisionEntry
from shape_msgs.msg import SolidPrimitive
from adamu_manipulation.arm_controller import AdamuDualArmController
from moveit_msgs.srv import GetPlanningScene
import copy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, DurabilityPolicy

def wrap_future(rclpy_future):
    """桥接 rclpy Future 到 asyncio Future"""
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()
    def rclpy_done_callback(f):
        loop.call_soon_threadsafe(aio_future.set_result, f.result())
    rclpy_future.add_done_callback(rclpy_done_callback)
    return aio_future


def get_hand_world_pose(box_pose_world: Pose, hand_offset_in_box: list, hand_rotation_in_box: np.ndarray) -> Pose:
    T_world_box = np.eye(4)
    T_world_box[:3, 3] = [box_pose_world.position.x, box_pose_world.position.y, box_pose_world.position.z]
    
    box_quat = [box_pose_world.orientation.x, box_pose_world.orientation.y,
                box_pose_world.orientation.z, box_pose_world.orientation.w]
    T_world_box[:3, :3] = R.from_quat(box_quat).as_matrix()

    T_box_hand = np.eye(4)
    T_box_hand[:3, :3] = hand_rotation_in_box
    T_box_hand[:3, 3] = hand_offset_in_box

    T_world_hand = np.dot(T_world_box, T_box_hand)

    final_pose = Pose()
    final_pose.position.x, final_pose.position.y, final_pose.position.z = T_world_hand[:3, 3]
    q = R.from_matrix(T_world_hand[:3, :3]).as_quat()
    final_pose.orientation.x, final_pose.orientation.y = q[0], q[1]
    final_pose.orientation.z, final_pose.orientation.w = q[2], q[3]
    return final_pose


class TaskPlanner(Node):
    def __init__(self):
        super().__init__('t1_m_task_planner')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_poses = {} 
        self.create_timer(0.1, self._publish_debug_tfs) 
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        # 初始化带瞬态本地 QoS 的发布器
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(Marker, '/push_vectors_viz', qos_profile)
        # 碰撞箱高频同步控制 (20Hz)
        self.sync_active = True
        self.create_timer(0.05, self._sync_box_to_moveit) 

    def pause_collision_sync(self):
        """暂停环境更新，供路径规划期间使用，防止 Scene updated during planning 错误"""
        self.sync_active = False

    def resume_collision_sync(self):
        """恢复环境高频更新"""
        self.sync_active = True

    def _sync_box_to_moveit(self):
        """后台高频执行的同步任务"""
        if not self.sync_active:
            return
        raw_pose, _, _, _ = self.get_smart_box_pose()
        if raw_pose is not None:
            # 采用静默模式更新，避免日志刷屏
            self.update_box_collision(raw_pose, quiet=True)

    def publish_arrow_marker(self, marker_id: int, start_pt: np.ndarray, vector: np.ndarray, color: tuple):
        marker = Marker()
        marker.header.frame_id = 'world'
        
        # 极其关键：将时间戳强制设为 0。
        # 这样 RViz 会无视仿真时间和系统时间的差异，强制在当前帧渲染该 Marker
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        
        marker.ns = 'push_vectors'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 确保持续时间无限，不被自动清理
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        p_start = Point()
        p_start.x, p_start.y, p_start.z = float(start_pt[0]), float(start_pt[1]), float(start_pt[2])

        p_end = Point()
        p_end.x = float(start_pt[0] + vector[0])
        p_end.y = float(start_pt[1] + vector[1])
        p_end.z = float(start_pt[2] + vector[2])

        marker.points = [p_start, p_end]

        # 尺寸放大，确保在 24cm 的大箱子旁清晰可见
        marker.scale.x = 0.01  # 箭杆直径 1cm
        marker.scale.y = 0.02  # 箭头直径 2cm
        marker.scale.z = 0.02  # 箭头长度 2cm

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0   # 1.0 为完全不透明

        self.marker_pub.publish(marker)

    def get_smart_box_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('world', 'target_box_fixed', rclpy.time.Time())
            
            raw_pose = Pose()
            raw_pose.position.x = trans.transform.translation.x
            raw_pose.position.y = trans.transform.translation.y
            raw_pose.position.z = trans.transform.translation.z
            raw_pose.orientation = trans.transform.rotation
            
            q = [raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z, raw_pose.orientation.w]
            r = R.from_quat(q)
            euler = r.as_euler('xyz', degrees=False)
            raw_yaw = euler[2]
            
            # shift_count = round(raw_yaw / (math.pi / 2.0))
            # eff_yaw = raw_yaw - shift_count * (math.pi / 2.0)
            
            # r_eff = R.from_euler('xyz', [euler[0], euler[1], eff_yaw], degrees=False)
            # q_eff = r_eff.as_quat()
            
            # eff_pose = copy.deepcopy(raw_pose)
            # eff_pose.orientation.x = q_eff[0]
            # eff_pose.orientation.y = q_eff[1]
            # eff_pose.orientation.z = q_eff[2]
            # eff_pose.orientation.w = q_eff[3]

            eff_pose = copy.deepcopy(raw_pose)
            eff_yaw = raw_yaw
            
            return raw_pose, eff_pose, math.degrees(raw_yaw), math.degrees(eff_yaw)
            
        except Exception:
            return None, None, None, None
    
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

    def update_box_collision(self, box_pose: Pose, box_name: str = 'target_box', quiet: bool = False):
        coll_obj = CollisionObject()
        coll_obj.header.stamp = self.get_clock().now().to_msg()
        coll_obj.header.frame_id = 'world'
        coll_obj.id = box_name
        coll_obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.24, 0.24, 0.24] 
        coll_obj.primitives.append(primitive)
        coll_obj.primitive_poses.append(box_pose)
        
        # 使用差分更新，强制作业场景同步重绘
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.world.collision_objects.append(coll_obj)
        self.scene_pub.publish(scene_msg)
        
        if not quiet:
            self.get_logger().info(f"已将 {box_name} 障碍物更新至 MoveIt 场景")

    async def set_acm_for_grasping(self, box_name: str = 'target_box', allow: bool = True):
        if not self.get_scene_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("ACM: 无法连接到 /get_planning_scene 服务")
            return

        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        resp = await wrap_future(self.get_scene_client.call_async(req))
        acm = resp.scene.allowed_collision_matrix

        if not acm.entry_names:
            self.get_logger().warn("当前 ACM 为空，无法进行豁免操作。")
            return

        if box_name not in acm.entry_names:
            acm.entry_names.append(box_name)
            for entry in acm.entry_values:
                entry.enabled.append(False)
            new_entry = AllowedCollisionEntry()
            new_entry.enabled = [False] * len(acm.entry_names)
            acm.entry_values.append(new_entry)

        box_idx = acm.entry_names.index(box_name)
        target_keywords = ['thumb', 'index', 'middle', 'ring', 'pinky', 'wrist', 'hand']
        
        mod_count = 0
        for i, name in enumerate(acm.entry_names):
            if any(kw in name.lower() for kw in target_keywords):
                acm.entry_values[box_idx].enabled[i] = allow
                acm.entry_values[i].enabled[box_idx] = allow
                mod_count += 1

        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.allowed_collision_matrix = acm
        self.scene_pub.publish(scene_msg)
        
        state = "豁免 (允许物理接触)" if allow else "恢复 (严格避障拦截)"
        self.get_logger().info(f"ACM更新: 已{state} [{box_name}] 与 {mod_count} 个手部结构的接触权限")


# =============================================================================
# 位姿解算库
# =============================================================================

def calc_bimanual_poses_1(box_pose, half_width=0.08, clearance=0.05):
    """双臂平行夹取两侧 (捧箱子)"""
    R_left_base = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R_right_base = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    adjust_x = -5.0 
    R_L_adjust = R.from_euler('x', adjust_x, degrees=True).as_matrix()
    R_R_adjust = R.from_euler('x', -adjust_x, degrees=True).as_matrix()

    R_left_final = np.dot(R_left_base, R_L_adjust)
    R_right_final = np.dot(R_right_base, R_R_adjust)

    l_hover = get_hand_world_pose(box_pose, [-0.07, half_width + clearance, 0.0], R_left_final)
    l_engage = get_hand_world_pose(box_pose, [-0.07, half_width - 0.015, 0.0], R_left_final)
    r_hover = get_hand_world_pose(box_pose, [-0.07, -half_width - clearance, 0.0], R_right_final)
    r_engage = get_hand_world_pose(box_pose, [-0.07, -half_width + 0.01, 0.0], R_right_final)

    l_delta = (l_engage.position.x - l_hover.position.x, l_engage.position.y - l_hover.position.y, l_engage.position.z - l_hover.position.z)
    r_delta = (r_engage.position.x - r_hover.position.x, r_engage.position.y - r_hover.position.y, r_engage.position.z - r_hover.position.z)
    return (l_hover, r_hover), (l_engage, r_engage), (l_delta, r_delta)


def calc_left_rear_corner_pose1(box_pose, half_length=0.12, half_width=0.12, half_height=0.12, clearance=0.02):
    """左后角：侧向接近直角抓取"""
    R_base = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]])
    R_roll = R.from_euler('x', np.radians(-5.0)).as_matrix()
    R_L_final = np.dot(R_base, R_roll)

    corner_x, corner_y, corner_z = -half_length, half_width, half_height
    hover_offset = [corner_x - clearance, corner_y + clearance , corner_z - 0.075]
    engage_offset = [corner_x, corner_y, corner_z - 0.075]

    l_hover = get_hand_world_pose(box_pose, hover_offset, R_L_final)
    l_engage = get_hand_world_pose(box_pose, engage_offset, R_L_final)
    l_delta = (l_engage.position.x - l_hover.position.x, l_engage.position.y - l_hover.position.y, l_engage.position.z - l_hover.position.z)
    return l_hover, l_engage, l_delta

def calc_left_rear_corner_pose2(box_pose, half_length=0.12, half_width=0.12, half_height=0.12, clearance=0.05):
    """左后角：侧向接近直角抓取"""
    R_base = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]])
    R_roll = R.from_euler('x', np.radians(70.0)).as_matrix()
    R_L_final = np.dot(R_base, R_roll)

    corner_x, corner_y, corner_z = -half_length, half_width, half_height
    hover_offset = [corner_x - clearance, corner_y-0.2 + clearance , corner_z - 0.075]
    engage_offset = [corner_x, corner_y, corner_z - 0.075]

    l_hover = get_hand_world_pose(box_pose, hover_offset, R_L_final)
    l_engage = get_hand_world_pose(box_pose, engage_offset, R_L_final)
    l_delta = (l_engage.position.x - l_hover.position.x, l_engage.position.y - l_hover.position.y, l_engage.position.z - l_hover.position.z)
    return l_hover, l_engage, l_delta

def calc_right_front_corner_pose1(box_pose, half_length=0.12, half_width=0.12, half_height=0.12, clearance=0.02):
    """右前角：平行贴合于前侧平面 (Front Face)"""
    # 保持绝对的侧面正交姿态，此时手掌法向量完全垂直于前侧平面
    R_base = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]])

    corner_x, corner_y, corner_z = half_length, -half_width, half_height
    R_roll = R.from_euler('x', np.radians(90.0)).as_matrix()
    R_R_final = np.dot(R_base, R_roll)
    # 沿 Y 轴（宽）向内收缩 2 厘米，确保手掌按压面完全落在前侧平面上，不滑出右边缘
    contact_y = corner_y + 0.02
    
    # 将安全余量 (clearance) 放在 X 轴方向，保证沿 X 轴垂直向后切入前侧平面
    hover_offset = [corner_x + clearance, contact_y-clearance, corner_z - 0.075]
    engage_offset = [corner_x, contact_y, corner_z - 0.075]

    r_hover = get_hand_world_pose(box_pose, hover_offset, R_R_final)
    r_engage = get_hand_world_pose(box_pose, engage_offset, R_R_final)
    
    r_delta = (r_engage.position.x - r_hover.position.x, 
               r_engage.position.y - r_hover.position.y, 
               r_engage.position.z - r_hover.position.z)
               
    return r_hover, r_engage, r_delta
def calc_right_front_corner_pose2(box_pose, half_length=0.12, half_width=0.12, half_height=0.12, clearance=0.02):
    """右前角：侧向接近直角抓取"""
    R_base = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]])
    R_roll = R.from_euler('x', np.radians(20.0)).as_matrix()
    R_R_final = np.dot(R_base, R_roll)

    corner_x, corner_y, corner_z = half_length, -half_width, half_height
    hover_offset = [corner_x + clearance-0.01, corner_y -clearance, corner_z - 0.04]
    engage_offset = [corner_x, corner_y, corner_z-0.04 ]

    r_hover = get_hand_world_pose(box_pose, hover_offset, R_R_final)
    r_engage = get_hand_world_pose(box_pose, engage_offset, R_R_final)
    r_delta = (r_engage.position.x - r_hover.position.x, r_engage.position.y - r_hover.position.y, r_engage.position.z - r_hover.position.z)
    return r_hover, r_engage, r_delta

# =============================================================================
# 第一阶段专用动作库 (<= 45度)
# =============================================================================
async def execute_stage1_push(controller, planner, eff_pose, raw_yaw, eff_yaw, push_distance, step):
    """第一阶段：基于最外侧棱角的切向推拨"""
    r_corner_local = np.array([0.12, -0.12, 0.0])
    l_corner_local = np.array([-0.12, 0.12, 0.0])
    
    r_push_local = np.cross(np.array([0, 0, -1]), r_corner_local) 
    r_push_local = r_push_local / np.linalg.norm(r_push_local) * push_distance
    
    l_push_local = np.cross(np.array([0, 0, -1]), l_corner_local)
    l_push_local = l_push_local / np.linalg.norm(l_push_local) * push_distance
    
    q_eff = [eff_pose.orientation.x, eff_pose.orientation.y, eff_pose.orientation.z, eff_pose.orientation.w]
    R_box_eff = R.from_quat(q_eff)
    r_push_world = R_box_eff.apply(r_push_local)
    l_push_world = R_box_eff.apply(l_push_local)
    
    box_center_world = np.array([eff_pose.position.x, eff_pose.position.y, eff_pose.position.z])
    r_start_world = box_center_world + R_box_eff.apply(r_corner_local)
    l_start_world = box_center_world + R_box_eff.apply(l_corner_local)
    
    planner.publish_arrow_marker(marker_id=1, start_pt=r_start_world, vector=r_push_world, color=(1.0, 0.0, 0.0))
    planner.publish_arrow_marker(marker_id=2, start_pt=l_start_world, vector=l_push_world, color=(0.0, 1.0, 0.0))
    
    controller.get_logger().info("\n" + "="*50)
    controller.get_logger().info(f"执行第 {step} 次动作 | 第一阶段: 卡角切向推拨")
    controller.get_logger().info(f"右手推力矢量: dx={r_push_world[0]:.4f}, dy={r_push_world[1]:.4f}")
    controller.get_logger().info(f"左手推力矢量: dx={l_push_world[0]:.4f}, dy={l_push_world[1]:.4f}")
    controller.get_logger().info("="*50)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input, "检查第一阶段 RViz 箭头方向无误后，按下回车键执行...")
    
    return await controller.execute_dual_arm_straight_line(
        (l_push_world[0], l_push_world[1], 0.0), 
        (r_push_world[0], r_push_world[1], 0.0)
    )

async def execute_stage1_retreat(controller, planner, retreat_distance=0.04):
    """第一阶段退让：沿对角线向外侧斜向抽离"""
    await asyncio.sleep(0.5)
    new_raw, new_eff, new_raw_yaw, new_eff_yaw = planner.get_smart_box_pose()
    if new_raw is None: return None, None, None, None

    controller.get_logger().info(f"第一阶段推拨完成，向对角线外侧抽离 {retreat_distance*100} 厘米...")
    q_new = [new_eff.orientation.x, new_eff.orientation.y, new_eff.orientation.z, new_eff.orientation.w]
    R_new = R.from_quat(q_new)
    
    r_outward = np.array([1.0, -1.0, 0.0])
    l_outward = np.array([-1.0, 1.0, 0.0])
    r_outward = r_outward / np.linalg.norm(r_outward) * retreat_distance
    l_outward = l_outward / np.linalg.norm(l_outward) * retreat_distance
    
    r_retreat_world = R_new.apply(r_outward)
    l_retreat_world = R_new.apply(l_outward)
    
    await controller.execute_dual_arm_straight_line(
        (l_retreat_world[0], l_retreat_world[1], 0.0), 
        (r_retreat_world[0], r_retreat_world[1], 0.0)
    )
    await asyncio.sleep(0.5)
    return new_raw, new_eff, new_raw_yaw, new_eff_yaw

# =============================================================================
# 第二阶段专用动作库 (> 45度)
# =============================================================================
async def execute_stage2_push(controller, planner, eff_pose, raw_yaw, eff_yaw, push_distance, step):
    """第二阶段：基于世界坐标系绝对 X 轴的双臂搓转 (推拉)"""
    # 接触点的起点仍然需要基于箱子姿态计算，以确保精准贴在物理表面上
    r_corner_local = np.array([0.12, -0.10, 0.0]) 
    l_corner_local = np.array([-0.12, 0.10, 0.0]) 
    
    q_eff = [eff_pose.orientation.x, eff_pose.orientation.y, eff_pose.orientation.z, eff_pose.orientation.w]
    R_box_eff = R.from_quat(q_eff)
    
    # [核心修正]：跳过局部坐标系转换！直接在世界坐标系下定义绝对直线轨迹
    # 左手笔直向前推 (世界 +X)，右手笔直往回拉 (世界 -X)
    r_push_world = np.array([-1.0, 0.0, 0.0]) * push_distance
    l_push_world = np.array([1.0, 0.0, 0.0]) * push_distance
    
    box_center_world = np.array([eff_pose.position.x, eff_pose.position.y, eff_pose.position.z])
    r_start_world = box_center_world + R_box_eff.apply(r_corner_local)
    l_start_world = box_center_world + R_box_eff.apply(l_corner_local)
    
    planner.publish_arrow_marker(marker_id=1, start_pt=r_start_world, vector=r_push_world, color=(1.0, 0.0, 0.0))
    planner.publish_arrow_marker(marker_id=2, start_pt=l_start_world, vector=l_push_world, color=(0.0, 1.0, 0.0))
    
    controller.get_logger().info("\n" + "="*50)
    controller.get_logger().info(f"执行第 {step} 次动作 | 第二阶段: 世界坐标系绝对 X 轴搓转")
    controller.get_logger().info(f"右手推力矢量: dx={r_push_world[0]:.4f}, dy={r_push_world[1]:.4f}")
    controller.get_logger().info(f"左手推力矢量: dx={l_push_world[0]:.4f}, dy={l_push_world[1]:.4f}")
    controller.get_logger().info("="*50)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input, "检查第二阶段 RViz 箭头方向是否严格平行于世界 X 轴，按下回车执行...")
    
    return await controller.execute_dual_arm_straight_line(
        (l_push_world[0], l_push_world[1], 0.0), 
        (r_push_world[0], r_push_world[1], 0.0)
    )

async def execute_stage2_retreat(controller, planner, retreat_distance=0.04):
    """第二阶段退让：垂直于前后侧面向外侧平移退出"""
    await asyncio.sleep(0.5)
    new_raw, new_eff, new_raw_yaw, new_eff_yaw = planner.get_smart_box_pose()
    if new_raw is None: return None, None, None, None

    controller.get_logger().info(f"第二阶段推拉完成，沿 X 轴向外垂直抽离 {retreat_distance*100} 厘米...")
    q_new = [new_eff.orientation.x, new_eff.orientation.y, new_eff.orientation.z, new_eff.orientation.w]
    R_new = R.from_quat(q_new)
    
    # 纯垂直方向外脱离：右手在前方继续向前(+X)离开，左手在后方继续向后(-X)离开
    r_outward = np.array([1.0, 0.0, 0.0]) * retreat_distance
    l_outward = np.array([-1.0, 0.0, 0.0]) * retreat_distance
    
    r_retreat_world = R_new.apply(r_outward)
    l_retreat_world = R_new.apply(l_outward)
    
    await controller.execute_dual_arm_straight_line(
        (l_retreat_world[0], l_retreat_world[1], 0.0), 
        (r_retreat_world[0], r_retreat_world[1], 0.0)
    )
    await asyncio.sleep(0.5)
    return new_raw, new_eff, new_raw_yaw, new_eff_yaw

# =============================================================================
# 任务流集合
# =============================================================================

async def main_parallel_task(controller, planner):
    """双臂平行捧取侧面任务"""
    controller.get_logger().info("开始执行：双臂平行侧面捧取任务...")
    if not await controller.wait_for_services(30.0): return

    box_pose = None 
    while box_pose is None:         
        box_pose = planner.get_box_pose()
        if box_pose is None: await asyncio.sleep(0.5)

    planner.update_box_collision(box_pose)
    await asyncio.sleep(0.5)

    hovers, engages, deltas = calc_bimanual_poses_1(box_pose)
    l_hover, r_hover = hovers
    l_delta, r_delta = deltas
    planner.set_debug_pose('l_target_pre', l_hover)
    planner.set_debug_pose('r_target_pre', r_hover)

    controller.get_logger().info("阶段 0: 双手预成型 (微扣掌心)")
    await asyncio.gather(
        controller.set_bionic_hand('left', thumb_opp=0.0, thumb_flex=0.2, index=0.2, middle=0.2, ring=0.2, pinky=0.2),
        controller.set_bionic_hand('right', thumb_opp=0.0, thumb_flex=0.0, index=0.2, middle=0.2, ring=0.2, pinky=0.2)
    )

    controller.get_logger().info("阶段 1: 双臂同步移动至侧方 Hover 位点")
    planner.pause_collision_sync()
    await controller.send_dual_arm_goal(l_hover, r_hover)
    planner.resume_collision_sync()

    controller.get_logger().info("阶段 2: 双臂同步笛卡尔直线侧面切入")
    await controller.execute_dual_arm_straight_line(l_delta, r_delta)


async def main_corner_task_v2 (controller, planner):
    """双臂棱角侧滑夹紧任务 (分阶段姿态切换)"""
    controller.get_logger().info("开始执行：双臂侧向切入棱角包络测试...")
    if not await controller.wait_for_services(30.0): return

    raw_pose, eff_pose, raw_yaw, eff_yaw = None, None, None, None
    while raw_pose is None:         
        raw_pose, eff_pose, raw_yaw, eff_yaw = planner.get_smart_box_pose()
        if raw_pose is None: await asyncio.sleep(0.5)
        else:
            # [新增]：将虚拟基准位姿广播到 RViz
            planner.set_debug_pose('eff_box_pose', eff_pose)

    # 初始手动同步一次（
    planner.update_box_collision(raw_pose)
    await asyncio.sleep(0.5)
    
    # ==========================================================
    # 1. 预定位与初始阶段判定 (兼容任意起始角度)
    # ==========================================================
    initial_phase_yaw = abs(raw_yaw) % 90.0
    
    if initial_phase_yaw <= 45.0:
        controller.get_logger().info(f"初始偏角 {initial_phase_yaw:.2f} <= 45度，进入第一阶段 (卡角策略)")
        # 第一阶段：获取卡角姿态
        r_hover, r_engage, r_delta = calc_right_front_corner_pose2(eff_pose)
        l_hover, l_engage, l_delta = calc_left_rear_corner_pose1(eff_pose) # 请确认这是你第一阶段左手的函数
        
        planner.set_debug_pose('r_hover', r_hover)
        planner.set_debug_pose('l_hover', l_hover)

        planner.pause_collision_sync()
        await controller.send_dual_arm_goal(l_hover, r_hover)
        planner.resume_collision_sync()
        
        # 第一阶段手型：右手半握卡角，左手大拇指张开卡角
        await asyncio.gather(
            controller.set_bionic_hand('right', thumb_opp=0.0, thumb_flex=0.0, index=0.5, middle=0.5, ring=0.5, pinky=0.5),
            controller.set_bionic_hand('left', thumb_opp=0.9, thumb_flex=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)
        )
    else:
        controller.get_logger().info(f"初始偏角 {initial_phase_yaw:.2f} > 45度，进入第二阶段 (平贴策略)")
        # 第二阶段：获取平贴姿态
        r_hover, r_engage, r_delta = calc_right_front_corner_pose1(eff_pose)
        l_hover, l_engage, l_delta = calc_left_rear_corner_pose2(eff_pose) # 请确认这是你第二阶段左手的函数
        
        planner.set_debug_pose('r_hover', r_hover)
        planner.set_debug_pose('l_hover', l_hover)

        planner.pause_collision_sync()
        await controller.send_dual_arm_goal(l_hover, r_hover)
        planner.resume_collision_sync()
        
        # 第二阶段手型：双手微屈，准备平贴平面
        await asyncio.gather(
            controller.set_bionic_hand('right', thumb_opp=0.0, thumb_flex=0.0, index=0.1, middle=0.1, ring=0.1, pinky=0.1),
            controller.set_bionic_hand('left', thumb_opp=0.0, thumb_flex=0.0, index=0.1, middle=0.1, ring=0.1, pinky=0.1)
        )



    # ==========================================================
    # 3. 动态分部旋转与重抓阶段 (Dynamic Nudging & Regrasping)
    # ==========================================================
    controller.get_logger().info("旋转阶段: 开始动态计算最大力矩方向并分步推拨")
    push_distance = 0.1 

    for step in range(1, 10):
        await asyncio.sleep(0.5)
        current_raw, current_eff, current_raw_yaw, current_eff_yaw = planner.get_smart_box_pose()
        if current_raw is None:
            controller.get_logger().error("丢失箱子 TF 坐标，无法继续！")
            break
        planner.set_debug_pose('eff_box_pose', current_eff)
        phase_yaw = abs(current_raw_yaw) % 90.0
        
        # 1. 开放接触权限
        controller.get_logger().info("正在修改 ACM 矩阵，开放手部接触权限...")
        await planner.set_acm_for_grasping('target_box', allow=True)
        await asyncio.sleep(0.5)

        # 2. 根据当前阶段，执行截然不同的专属推拨与退让逻辑
        if phase_yaw <= 45.0:
            success = await execute_stage1_push(controller, planner, current_eff, current_raw_yaw, current_eff_yaw, push_distance, step)
            if not success: break
            new_raw, new_eff, new_raw_yaw, new_eff_yaw = await execute_stage1_retreat(controller, planner)
        else:
            success = await execute_stage2_push(controller, planner, current_eff, current_raw_yaw, current_eff_yaw, push_distance, step)
            if not success: break
            new_raw, new_eff, new_raw_yaw, new_eff_yaw = await execute_stage2_retreat(controller, planner)

        if new_raw is None: break
        
        # 3. 关闭接触权限，准备下一轮 IK 规划
        controller.get_logger().info("正在修改 ACM 矩阵，恢复严格避障拦截...")
        await planner.set_acm_for_grasping('target_box', allow=False)
        
        # 4. 重新评估阶段，更新手型与位姿重定位
        new_phase_yaw = abs(new_raw_yaw) % 90.0
        
        if new_phase_yaw <= 45.0:
            new_r_hover, new_r_engage, new_r_delta = calc_right_front_corner_pose2(new_eff)
            new_l_hover, new_l_engage, new_l_delta = calc_left_rear_corner_pose1(new_eff)
            
            # 从第二阶段切回第一阶段时，恢复卡角手型
            if phase_yaw > 45.0:
                controller.get_logger().info("完成周期旋转，重置为第一阶段卡角手型！")
                await asyncio.gather(
                    controller.set_bionic_hand('right', thumb_opp=0.0, thumb_flex=0.0, index=0.5, middle=0.5, ring=0.5, pinky=0.5),
                    controller.set_bionic_hand('left', thumb_opp=0.9, thumb_flex=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)
                )
        else:
            new_r_hover, new_r_engage, new_r_delta = calc_right_front_corner_pose1(new_eff)
            new_l_hover, new_l_engage, new_l_delta = calc_left_rear_corner_pose2(new_eff)
            
            # 从第一阶段切入第二阶段时，改变为平贴手型
            if phase_yaw <= 45.0:
                controller.get_logger().info("跨越 45 度阈值：切换为第二阶段平贴手型！")
                await asyncio.gather(
                    controller.set_bionic_hand('right', thumb_opp=0.0, thumb_flex=0.0, index=0.1, middle=0.1, ring=0.1, pinky=0.1),
                    controller.set_bionic_hand('left', thumb_opp=0.0, thumb_flex=0.0, index=0.1, middle=0.1, ring=0.1, pinky=0.1)
                )
                
        # 无论哪个阶段，重算位姿并显示
        planner.set_debug_pose('r_hover', new_r_hover)
        planner.set_debug_pose('l_hover', new_l_hover)
        
        controller.get_logger().info("关节空间精确定位对齐到新的 Hover 悬停点")
        planner.pause_collision_sync()
        await controller.send_dual_arm_goal(new_l_hover, new_r_hover)
        planner.resume_collision_sync()

async def hand_test_task(controller, side='right'):
    """手指纯净测试体操"""
    controller.get_logger().info(f"执行 [{side}] 手指预演...")
    await controller.set_bionic_hand(side, thumb_opp=0.8, thumb_flex=0.0, index=0.5, middle=0.5, ring=0.5, pinky=0.5)
    await asyncio.sleep(2.0)
    await controller.set_bionic_hand(side, thumb_opp=0.0, thumb_flex=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)


# =============================================================================
# 节点启动
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    con = AdamuDualArmController()
    pla = TaskPlanner()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(con); executor.add_node(pla)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        asyncio.get_event_loop().run_until_complete(main_corner_task_v2(controller=con, planner=pla))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()