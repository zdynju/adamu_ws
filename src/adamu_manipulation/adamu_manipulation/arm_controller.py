import asyncio
import copy
import threading
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK, GetCartesianPath, GetPositionFK
from moveit_msgs.msg import (Constraints, JointConstraint, RobotState,
                              OrientationConstraint)
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PositionConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive

def wrap_future(rclpy_future):
    """
    桥接工具：将 rclpy 的 Future 转换为 asyncio 的 Future，
    解决 "Task got bad yield" 报错。
    """
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()

    def rclpy_done_callback(f):
        loop.call_soon_threadsafe(aio_future.set_result, f.result())

    rclpy_future.add_done_callback(rclpy_done_callback)
    return aio_future


class AdamuDualArmController(Node):
    def __init__(self):
        super().__init__('adamu_dual_arm_controller')
        self._current_joint_state = None
        self._js_lock = threading.Lock()          # 保护 _current_joint_state

        self.create_subscription(JointState, '/joint_states', self._js_callback, 10)

        self._move_group_client = ActionClient(self, MoveGroup,        '/move_action')
        self._ik_client         = self.create_client(GetPositionIK,    '/compute_ik')
        self._cartesian_client  = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self._fk_client         = self.create_client(GetPositionFK,    '/compute_fk')
        self._left_arm_client   = ActionClient(
            self, FollowJointTrajectory,
            '/left_arm_controller/follow_joint_trajectory')
        self._right_arm_client  = ActionClient(
            self, FollowJointTrajectory,
            '/right_arm_controller/follow_joint_trajectory')


    async def wait_for_services(self, total_timeout_sec: float = 30.0) -> bool:
        self.get_logger().info(f'正在等待规划与控制服务上线... (超时{total_timeout_sec}秒)')
        deadline = asyncio.get_event_loop().time() + total_timeout_sec

        check_items = [
            (self._move_group_client, '/move_action', True),
            (self._ik_client,         '/compute_ik', False),
            (self._cartesian_client,  '/compute_cartesian_path', False),
            (self._fk_client,         '/compute_fk', False),
            (self._left_arm_client,   '/left_arm_controller/follow_joint_trajectory', True),
            (self._right_arm_client,  '/right_arm_controller/follow_joint_trajectory', True),
        ]

        for client, name, is_action in check_items:
            self.get_logger().info(f'🔍 检查链路: {name}')
            while True:
                is_ready = client.server_is_ready() if is_action else client.service_is_ready()
                if is_ready:
                    break
                
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    self.get_logger().fatal(f'💀 等待 [{name}] 超时！')
                    return False
                
                await asyncio.sleep(0.5)

            self.get_logger().info(f'✅ 链路 [{name}] 已就绪')

        self.get_logger().info(' 所有基础运动链路已打通！')
        return True
    
    # =========================================================================
    # 回调：只存数据，不触发任何动作
    # =========================================================================

    def _js_callback(self, msg: JointState):
        with self._js_lock:
            self._current_joint_state = msg

    async def _wait_joint_state_stable(
        self, timeout_sec: float = 2.0, tol: float = 0.01
    ) -> bool:
        """
        等待关节状态稳定（连续两帧最大偏差 < tol rad 即认为稳定）。
        用于接触物体后关节受力偏移的场景，确保笛卡尔规划起点与实际一致。
        """
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        prev_positions = None
        while self.get_clock().now().nanoseconds < deadline:
            await asyncio.sleep(0.05)
            with self._js_lock:
                curr = self._current_joint_state
            if curr is None:
                continue
            curr_positions = list(curr.position)
            if prev_positions is not None:
                diff = max(abs(a - b) for a, b in zip(prev_positions, curr_positions))
                if diff < tol:
                    self.get_logger().info(
                        f'✅ 关节状态已稳定（最大偏差 {diff:.4f} rad）'
                    )
                    return True
            prev_positions = curr_positions
        self.get_logger().warn('⚠️ 等待关节稳定超时，使用当前状态继续')
        return False


    # =========================================================================
    # FK
    # =========================================================================

    async def get_current_eef_pose(self, group_name: str, tip_link: str) -> Pose | None:
        with self._js_lock:
            js = self._current_joint_state
        if js is None:
            self.get_logger().error('尚未收到 /joint_states，无法执行 FK！')
            return None

        req = GetPositionFK.Request()
        req.header.frame_id         = 'world'
        req.fk_link_names           = [tip_link]
        req.robot_state.joint_state = js

        resp = await wrap_future(self._fk_client.call_async(req))
        if resp.error_code.val != 1:
            self.get_logger().error(f'[{group_name}] FK 失败，错误码: {resp.error_code.val}')
            return None

        pose = resp.pose_stamped[0].pose
        self.get_logger().info(
            f'[{group_name}] FK → '
            f'xyz=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        )
        return pose

    # =========================================================================
    # IK
    # =========================================================================

    async def compute_ik(self, group_name: str, tip_link: str, target_pose: Pose):
        req = GetPositionIK.Request()
        req.ik_request.group_name                   = group_name
        req.ik_request.ik_link_name                 = tip_link
        req.ik_request.pose_stamped.header.frame_id = 'world'
        req.ik_request.pose_stamped.pose            = target_pose
        req.ik_request.timeout.sec                  = 5
        req.ik_request.avoid_collisions             = True

        if self._current_joint_state is not None:
            with self._js_lock:
                js = self._current_joint_state
            seed = RobotState()
            seed.joint_state           = js
            req.ik_request.robot_state = seed

        response = await wrap_future(self._ik_client.call_async(req))
        if response.error_code.val != 1:
            self.get_logger().error(f'[{group_name}] IK 失败，错误码: {response.error_code.val}')
            return None

        self.get_logger().info(f'[{group_name}] IK 求解成功')
        return response.solution.joint_state

    # =========================================================================
    # 关节空间双臂规划
    # =========================================================================

    def _build_joint_goal(self, joint_names, joint_positions, name_suffix=None):
        constraint = Constraints()
        for name, position in zip(joint_names, joint_positions):
            if name_suffix and not name.endswith(name_suffix):
                continue
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(position)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraint.joint_constraints.append(jc)
        return constraint


    async def send_single_arm_goal(self, side: str, target_pose: Pose) -> bool:
        """
        单臂大范围关节空间规划接口 (IK + RRT避障)
        
        Args:
            side: 'left' 或 'right'
            target_pose: 目标位姿
        """
        if side not in ['left', 'right']:
            self.get_logger().error(f'❌ 参数错误: side 必须是 left 或 right')
            return False

        # 动态映射参数
        group_name = 'left_arm' if side == 'left' else 'right_arm'
        tip_link   = 'left_hand_tcp' if side == 'left' else 'right_hand_tcp'
        suffix     = 'Left' if side == 'left' else 'Right'

        self.get_logger().info(f'=== 求解单臂 IK [{group_name}] ===')
        target_js = await self.compute_ik(group_name, tip_link, target_pose)
        if target_js is None:
            return False

        self.get_logger().info(f'=== 单臂关节空间规划 [{group_name}] ===')
        # 构建约束（复用完美的 _build_joint_goal）
        goal_constraints = self._build_joint_goal(target_js.name, target_js.position, suffix)

        goal_msg = MoveGroup.Goal()
        goal_msg.request.planner_id                      = 'RRTConnectkConfigDefault'
        goal_msg.request.group_name                      = group_name 
        goal_msg.request.num_planning_attempts           = 5
        goal_msg.request.allowed_planning_time           = 3.0
        goal_msg.request.max_velocity_scaling_factor     = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5
        goal_msg.request.goal_constraints                = [goal_constraints]

        # 发送给 MoveIt
        goal_handle = await wrap_future(self._move_group_client.send_goal_async(goal_msg))
        if not goal_handle.accepted:
            self.get_logger().error(f'[{group_name}] 规划请求被拒绝！')
            return False

        result = await wrap_future(goal_handle.get_result_async())
        ok = result.result.error_code.val == 1
        self.get_logger().info(
            f'✅ [{group_name}] 移动成功' if ok else
            f'❌ [{group_name}] 移动失败，错误码: {result.result.error_code.val}'
        )
        return ok
    



    async def send_dual_arm_goal(self, left_pose: Pose, right_pose: Pose) -> bool:
        self.get_logger().info('=== 求解双臂 IK ===')
        left_js = await self.compute_ik('left_arm', 'left_hand_tcp', left_pose)
        if left_js is None:
            return False
        right_js = await self.compute_ik('right_arm', 'right_hand_tcp', right_pose)
        if right_js is None:
            return False

        self.get_logger().info('=== 双臂关节空间规划 ===')
        combined = Constraints()
        combined.joint_constraints = (
            self._build_joint_goal(left_js.name,  left_js.position,  'Left').joint_constraints +
            self._build_joint_goal(right_js.name, right_js.position, 'Right').joint_constraints
        )

        goal_msg = MoveGroup.Goal()
        goal_msg.request.planner_id                      = 'RRTConnectkConfigDefault'
        goal_msg.request.group_name                      = 'dual_arm'
        goal_msg.request.num_planning_attempts           = 5
        goal_msg.request.allowed_planning_time           = 3.0
        goal_msg.request.max_velocity_scaling_factor     = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2
        goal_msg.request.goal_constraints                = [combined]

        goal_handle = await wrap_future(self._move_group_client.send_goal_async(goal_msg))
        if not goal_handle.accepted:
            self.get_logger().error('规划请求被拒绝！')
            return False

        result = await wrap_future(goal_handle.get_result_async())
        ok = result.result.error_code.val == 1
        self.get_logger().info(
            '✅ 双臂移动成功' if ok else
            f'❌ 双臂移动失败，错误码: {result.result.error_code.val}'
        )
        return ok
        
    async def _send_to_hw(self, hw_client: ActionClient, group_name: str, traj) -> bool:
        """将生成的轨迹发送给底层的 FollowJointTrajectory 硬件控制器"""
        self.get_logger().info(f'[{group_name}] 正在下发轨迹给硬件控制器...')
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        goal_handle = await wrap_future(hw_client.send_goal_async(goal_msg))
        if not goal_handle.accepted:
            self.get_logger().error(f'❌ [{group_name}] 硬件控制器拒绝执行！')
            return False

        result = await wrap_future(goal_handle.get_result_async())
        # FollowJointTrajectory 成功错误码通常为 0
        ok = (result.result.error_code == 0)
        if ok:
            self.get_logger().info(f'✅ [{group_name}] 轨迹执行完毕')
        else:
            self.get_logger().error(f'❌ [{group_name}] 执行失败，错误码: {result.result.error_code}')
        return ok
    # =========================================================================
    # Pilz 工业规划核心接口 (LIN 直线 / PTP 点到点)
    # =========================================================================

    def _build_pose_goal(self, link_name: str, target_pose: Pose) -> Constraints:
        """为 Pilz 规划器构建严格的 6DoF 笛卡尔目标约束"""
        # 1. 位置约束 (容忍度 1mm 的球形包围盒)
        pc = PositionConstraint()
        pc.header.frame_id = 'world'
        pc.link_name       = link_name
        
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.001]  # 1mm 误差允许
        bv.primitives.append(sphere)
        
        sphere_pose = copy.deepcopy(target_pose)
        bv.primitive_poses.append(sphere_pose)
        
        pc.constraint_region = bv
        pc.weight = 1.0

        # 2. 姿态约束 (严格保持姿态，容忍度极小)
        oc = OrientationConstraint()
        oc.header.frame_id = 'world'
        oc.link_name       = link_name
        oc.orientation     = target_pose.orientation
        oc.absolute_x_axis_tolerance = 0.005
        oc.absolute_y_axis_tolerance = 0.005
        oc.absolute_z_axis_tolerance = 0.005
        oc.weight = 1.0

        constraint = Constraints()
        constraint.position_constraints.append(pc)
        constraint.orientation_constraints.append(oc)
        return constraint

    async def _plan_with_pilz(
        self,
        group_name: str,
        tip_link:   str,
        target_pose: Pose,
        planner_id: str = 'LIN',  # 可选 'LIN' (直线) 或 'PTP' (点到点)
        vel_scale:  float = 0.1,
        acc_scale:  float = 0.1
    ):
        """调用 Pilz 管道生成工业级平滑轨迹"""
        self.get_logger().info(f'[{group_name}] 启动 Pilz [{planner_id}] 规划...')
        
        goal_msg = MoveGroup.Goal()
        # 【核心切换】强行指定使用 Pilz 管道
        goal_msg.request.pipeline_id                     = 'pilz_industrial_motion_planner'
        goal_msg.request.planner_id                      = planner_id
        goal_msg.request.group_name                      = group_name
        goal_msg.request.max_velocity_scaling_factor     = vel_scale
        goal_msg.request.max_acceleration_scaling_factor = acc_scale
        
        goal_constraints = self._build_pose_goal(tip_link, target_pose)
        goal_msg.request.goal_constraints = [goal_constraints]
        goal_msg.request.plan_only = True  # 只规划不执行，用于提取轨迹

        goal_handle = await wrap_future(self._move_group_client.send_goal_async(goal_msg))
        if not goal_handle.accepted:
            self.get_logger().error(f'[{group_name}] Pilz {planner_id} 规划被拒绝！(目标可能超出工作空间)')
            return None

        result = await wrap_future(goal_handle.get_result_async())
        error_code = result.result.error_code.val
        
        if error_code == 1:
            traj = result.result.planned_trajectory.joint_trajectory
            self.get_logger().info(f'✅ [{group_name}] Pilz {planner_id} 规划成功 ({len(traj.points)} 轨迹点)')
            return traj
        else:
            self.get_logger().error(f'❌ [{group_name}] Pilz 失败，错误码: {error_code}')
            return None

    # =========================================================================
    # 单/双臂直线运动高级封装 (直接调用这个)
    # =========================================================================

    async def execute_pilz_straight_line(
        self, side: str, delta: tuple[float, float, float], vel: float = 0.1
    ) -> bool:
        """
        单臂极其丝滑的笛卡尔直线移动
        """
        if side not in ['left', 'right']: return False

        group_name = 'left_arm' if side == 'left' else 'right_arm'
        tip_link   = 'left_hand_tcp' if side == 'left' else 'right_hand_tcp'
        hw_client  = self._left_arm_client if side == 'left' else self._right_arm_client

        await self._wait_joint_state_stable()
        start_pose = await self.get_current_eef_pose(group_name, tip_link)
        if start_pose is None: return False

        # 计算增量目标
        target_pose = copy.deepcopy(start_pose)
        target_pose.position.x += delta[0]
        target_pose.position.y += delta[1]
        target_pose.position.z += delta[2]

        # 强制使用 LIN 直线规划器
        traj = await self._plan_with_pilz(
            group_name, tip_link, target_pose, planner_id='LIN', vel_scale=vel, acc_scale=vel
        )
        if traj is None: return False

        return await self._send_to_hw(hw_client, group_name, traj)

    async def execute_dual_arm_straight_line(self, left_delta, right_delta):

        # 1. 获取当前位姿
        await self._wait_joint_state_stable()

        left_start  = await self.get_current_eef_pose('left_arm', 'left_hand_tcp')
        right_start = await self.get_current_eef_pose('right_arm', 'right_hand_tcp')

        # 2. 构造目标
        left_target = copy.deepcopy(left_start)
        left_target.position.x += left_delta[0]
        left_target.position.y += left_delta[1]
        left_target.position.z += left_delta[2]

        right_target = copy.deepcopy(right_start)
        right_target.position.x += right_delta[0]
        right_target.position.y += right_delta[1]
        right_target.position.z += right_delta[2]

        # 3. 分别规划（关键：plan_only）
        left_traj = await self._plan_with_pilz('left_arm', 'left_hand_tcp', left_target)
        right_traj = await self._plan_with_pilz('right_arm', 'right_hand_tcp', right_target)

        if left_traj is None or right_traj is None:
            return False

        # 4. 同步执行（核心）
        results = await asyncio.gather(
            self._send_to_hw(self._left_arm_client, 'left_arm', left_traj),
            self._send_to_hw(self._right_arm_client, 'right_arm', right_traj),
        )

        return all(results)