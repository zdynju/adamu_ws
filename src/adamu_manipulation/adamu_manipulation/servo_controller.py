"""
AdamuServoController — 双臂 MoveIt Servo 控制器（重构版）

重构核心思路
------------
原版 servo_cartesian 的 stop_condition 回调同时承担了两件事：
  - 返回 True       → 停止运动
  - 返回 list[6]    → 更新速度

这让调用方很难理解该返回什么，也让导纳、轨迹跟踪等逻辑都挤进同一个回调。

重构后拆成两个职责清晰的参数：
  - velocity_fn(t: float) -> list[float]   # 给定已运行时间，输出6维速度
  - termination_fn()      -> bool           # 判断是否应该停止

调用方只需关心"我想动多快"和"我想什么时候停"，两者互相独立。
导纳叠加也只是 velocity_fn 内部的一个加法，不需要单独的方法。
"""

import asyncio
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped
from adamu_manipulation.fts_processor import FTSProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  asyncio / rclpy 桥接工具
# ─────────────────────────────────────────────────────────────────────────────

def wrap_future(rclpy_future):
    """
    将 rclpy.Future 转换为 asyncio.Future。

    rclpy 的 Future 不能直接被 await，需要通过此桥接函数转换。
    解决在 asyncio 事件循环中调用 ROS2 服务时出现的 "Task got bad yield" 报错。
    """
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()

    def _done_cb(f):
        # rclpy 的回调在 ROS 线程触发，需切回 asyncio 线程安全地设置结果
        loop.call_soon_threadsafe(aio_future.set_result, f.result())

    rclpy_future.add_done_callback(_done_cb)
    return aio_future


# ─────────────────────────────────────────────────────────────────────────────
#  控制器主体
# ─────────────────────────────────────────────────────────────────────────────

class AdamuServoController(Node):
    """
    双臂 MoveIt Servo 控制器。

    负责：
      1. Servo 节点的启停（start_servo / stop_servo 服务）
      2. 笛卡尔速度指令的发布（TwistStamped）
      3. 执行循环：按 velocity_fn 生成速度，按 termination_fn 决定何时停止

    不负责：
      - 轨迹的几何计算（由调用方的 velocity_fn 提供）
      - 导纳的参数整定（由调用方包装进 velocity_fn）
      - 任务级状态机（由上层节点管理）
    """

    def __init__(self, fts_processor: FTSProcessor):
        super().__init__('adamu_servo_controller_node')
        self.get_logger().info('初始化 Servo 控制器...')

        # use_sim_time 在仿真环境下必须开启，否则时间戳对不上
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # ── Servo 启停服务客户端 ───────────────────────────────────────────
        self._left_start_client  = self.create_client(Trigger, '/left_arm/servo_node/start_servo')
        self._left_stop_client   = self.create_client(Trigger, '/left_arm/servo_node/stop_servo')
        self._right_start_client = self.create_client(Trigger, '/right_arm/servo_node/start_servo')
        self._right_stop_client  = self.create_client(Trigger, '/right_arm/servo_node/stop_servo')

        # ── 笛卡尔速度发布器 ──────────────────────────────────────────────
        self._left_twist_pub  = self.create_publisher(
            TwistStamped, '/left_arm/servo_node/delta_twist_cmds', 10)
        self._right_twist_pub = self.create_publisher(
            TwistStamped, '/right_arm/servo_node/delta_twist_cmds', 10)

        # ── 力传感器处理器（外部注入，供 velocity_fn 使用） ───────────────
        self.fts_processor = fts_processor

        # ── Servo 激活状态 ────────────────────────────────────────────────
        self._left_servo_active:  bool = False
        self._right_servo_active: bool = False

        self.get_logger().info('Servo 控制器初始化完成')

    # ─────────────────────────────────────────────────────────────────────────
    #  服务等待与调用
    # ─────────────────────────────────────────────────────────────────────────

    async def wait_for_services(self, total_timeout_sec: float = 30.0) -> bool:
        """
        等待所有 Servo 服务上线。

        在节点启动后、第一次激活 Servo 之前调用。
        使用 asyncio.sleep 轮询，不阻塞事件循环。
        """
        self.get_logger().info(f'等待 Servo 服务上线（超时 {total_timeout_sec}s）...')
        deadline = asyncio.get_event_loop().time() + total_timeout_sec

        clients = [
            (self._left_start_client,  '/left_arm/servo_node/start_servo'),
            (self._left_stop_client,   '/left_arm/servo_node/stop_servo'),
            (self._right_start_client, '/right_arm/servo_node/start_servo'),
            (self._right_stop_client,  '/right_arm/servo_node/stop_servo'),
        ]

        for client, name in clients:
            while not client.service_is_ready():
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    self.get_logger().fatal(f'等待 [{name}] 超时')
                    return False
                self.get_logger().warn(f'未检测到 [{name}]，剩余 {remaining:.1f}s')
                await asyncio.sleep(0.5)
            self.get_logger().info(f'服务 [{name}] 就绪')

        self.get_logger().info('所有 Servo 服务就绪')
        return True

    async def _call_service(self, client, label: str, timeout_sec: float = 5.0) -> bool:
        """
        调用单个 Trigger 服务并等待响应。

        先检查服务是否可用再调用，避免挂死。
        调用失败或超时均记录 error 日志并返回 False。
        """
        if not client.service_is_ready():
            self.get_logger().error(f'[{label}] 服务不可用')
            return False

        try:
            rclpy_future = client.call_async(Trigger.Request())
            result = await asyncio.wait_for(wrap_future(rclpy_future), timeout=timeout_sec)
        except asyncio.TimeoutError:
            self.get_logger().error(f'[{label}] 调用超时（{timeout_sec}s）')
            return False
        except Exception as e:
            self.get_logger().error(f'[{label}] 调用异常: {e}')
            return False

        if result.success:
            self.get_logger().info(f'[{label}] 成功: {result.message}')
        else:
            self.get_logger().error(f'[{label}] 失败: {result.message}')
        return result.success

    # ─────────────────────────────────────────────────────────────────────────
    #  Servo 启停
    # ─────────────────────────────────────────────────────────────────────────

    async def activate_left_servo(self) -> bool:
        """激活左臂 Servo。已激活时直接返回 True，不重复调用服务。"""
        if self._left_servo_active:
            return True
        success = await self._call_service(self._left_start_client, '左臂 start_servo')
        self._left_servo_active = success
        return success

    async def deactivate_left_servo(self) -> bool:
        """停用左臂 Servo。先发零速刹车，再调用停止服务。"""
        if not self._left_servo_active:
            return True
        self._publish_twist(self._left_twist_pub, 'world')  # 零速刹车
        await asyncio.sleep(0.1)
        success = await self._call_service(self._left_stop_client, '左臂 stop_servo')
        if success:
            self._left_servo_active = False
        return success

    async def activate_right_servo(self) -> bool:
        """激活右臂 Servo。已激活时直接返回 True，不重复调用服务。"""
        if self._right_servo_active:
            return True
        success = await self._call_service(self._right_start_client, '右臂 start_servo')
        self._right_servo_active = success
        return success

    async def deactivate_right_servo(self) -> bool:
        """停用右臂 Servo。先发零速刹车，再调用停止服务。"""
        if not self._right_servo_active:
            return True
        self._publish_twist(self._right_twist_pub, 'world')  # 零速刹车
        await asyncio.sleep(0.1)
        success = await self._call_service(self._right_stop_client, '右臂 stop_servo')
        if success:
            self._right_servo_active = False
        return success

    async def activate_both_servo(self) -> bool:
        """并发激活双臂 Servo。任一侧失败则整体返回 False。"""
        left_ok, right_ok = await asyncio.gather(
            self.activate_left_servo(),
            self.activate_right_servo(),
        )
        return left_ok and right_ok

    async def deactivate_both_servo(self) -> bool:
        """并发停用双臂 Servo。任一侧失败则整体返回 False。"""
        left_ok, right_ok = await asyncio.gather(
            self.deactivate_left_servo(),
            self.deactivate_right_servo(),
        )
        return left_ok and right_ok

    # ─────────────────────────────────────────────────────────────────────────
    #  底层发布
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_twist(self, pub, frame_id: str,
                       vx=0.0, vy=0.0, vz=0.0,
                       wx=0.0, wy=0.0, wz=0.0):
        """
        构造并发布一帧 TwistStamped。

        所有速度默认为 0，调用时只需传入非零分量。
        frame_id 决定速度的参考坐标系，通常用 'world'。
        """
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz
        pub.publish(msg)

    @staticmethod
    def _apply_deadband(value: float, deadband: float) -> float:
        """小于 deadband 的值截断为 0，防止传感器噪声引起微小抖动。"""
        return 0.0 if abs(value) < deadband else value

    # ─────────────────────────────────────────────────────────────────────────
    #  核心执行循环
    # ─────────────────────────────────────────────────────────────────────────

    async def servo_cartesian(
        self,
        side: str,
        velocity_fn: callable,
        termination_fn: callable,
        frame_id: str = 'world',
        rate_hz: float = 50.0,
    ) -> bool:
        """
        单臂笛卡尔速度执行循环。

        参数
        ----
        side : 'left' 或 'right'

        velocity_fn : (t: float) -> list[float, 6]
            给定已运行时间 t（秒），返回 [vx, vy, vz, wx, wy, wz]。
            这是唯一的速度来源，轨迹前馈、导纳修正都在这里叠加。

            示例——常速运动：
                velocity_fn = lambda t: [0.0, 0.05, 0.0, 0.0, 0.0, 0.0]

            示例——轨迹查表：
                velocity_fn = lambda t: traj.interpolate(t)

            示例——轨迹 + 导纳叠加：
                def velocity_fn(t):
                    v_ff  = traj.interpolate(t)
                    v_adj = admittance.compute(fts.get_force(side), target_force)
                    return [a + b for a, b in zip(v_ff, v_adj)]

        termination_fn : () -> bool
            返回 True 时执行循环立即停止。
            终止逻辑（超时、目标到达、力阈值等）全部在这里判断。

            示例——固定时长：
                t0 = node.get_clock().now()
                termination_fn = lambda: (node.get_clock().now() - t0).nanoseconds > 3e9

            示例——力阈值保护：
                termination_fn = lambda: fts.get_force('left')[2] > 50.0

        frame_id : 速度指令的参考坐标系，通常为 'world'

        rate_hz : 控制循环频率，建议 50–500 Hz

        返回
        ----
        True  : 正常终止（termination_fn 返回 True）
        False : 异常中止（Servo 未激活、发布异常、任务被取消）
        """
        # 检查 Servo 是否已激活
        is_active = self._left_servo_active if side == 'left' else self._right_servo_active
        if not is_active:
            self.get_logger().error(f'{side} 臂 Servo 未激活，无法执行')
            return False

        pub = self._left_twist_pub if side == 'left' else self._right_twist_pub
        start_time = self.get_clock().now()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _publish_callback():
            """
            定时器回调，每个控制周期执行一次。

            执行顺序：
              1. 先判断是否终止 —— 避免在应该停下的时刻还发出运动指令
              2. 调用 velocity_fn 获取本周期速度
              3. 发布速度
            """
            # 1. 终止判断
            try:
                should_stop = termination_fn()
            except Exception as e:
                self.get_logger().error(f'termination_fn 异常: {e}')
                loop.call_soon_threadsafe(stop_event.set)
                return

            if should_stop:
                loop.call_soon_threadsafe(stop_event.set)
                return

            # 2. 速度计算（t 为已运行时间，单位秒）
            t = (self.get_clock().now() - start_time).nanoseconds / 1e9
            try:
                vel = velocity_fn(t)
            except Exception as e:
                self.get_logger().error(f'velocity_fn 异常: {e}')
                loop.call_soon_threadsafe(stop_event.set)
                return

            # 3. 发布
            try:
                self._publish_twist(pub, frame_id, *vel)
            except Exception as e:
                self.get_logger().error(f'Twist 发布失败: {e}')
                loop.call_soon_threadsafe(stop_event.set)

        timer = None
        try:
            timer = self.create_timer(1.0 / rate_hz, _publish_callback)
            await stop_event.wait()
            return True

        except asyncio.CancelledError:
            self.get_logger().warn(f'{side} 臂任务被取消')
            return False
        except Exception as e:
            self.get_logger().error(f'{side} 臂执行异常: {e}')
            return False
        finally:
            # 销毁定时器，防止回调在刹车期间继续触发
            if timer is not None:
                self.destroy_timer(timer)

            # 发送几帧零速作为刹车，防止 Servo 因超时报警
            for _ in range(5):
                self._publish_twist(pub, frame_id)
                await asyncio.sleep(0.01)

            self.get_logger().info(f'{side} 臂运动结束')

    async def servo_both_cartesian(
        self,
        velocity_fn_left: callable,
        velocity_fn_right: callable,
        termination_fn: callable,
        frame_id: str = 'world',
        rate_hz: float = 50.0,
    ) -> bool:
        """
        双臂同步执行循环。

        两臂共享同一个 termination_fn，任意一侧触发终止条件，
        另一侧也会随之停止，保证双臂不会出现一侧还在运动、另一侧已停的情况。

        velocity_fn_left / velocity_fn_right 各自独立，但应从同一个
        轨迹对象派生，以保证几何上的间距一致性。

        典型用法（翻转任务）
        --------------------
            traj = FlipTrajectory(box_dims, pivot)
            t0   = self.get_clock().now()

            await ctrl.servo_both_cartesian(
                velocity_fn_left  = lambda t: traj.velocity_left(t),
                velocity_fn_right = lambda t: traj.velocity_right(t),
                termination_fn    = lambda: traj.is_done(
                    (self.get_clock().now() - t0).nanoseconds / 1e9
                ),
            )
        """
        # 用一个共享 Event 桥接两个子任务的终止
        shared_stop = asyncio.Event()

        def termination_with_broadcast():
            """
            任意一侧调用此函数，只要 termination_fn 返回 True，
            就设置共享 Event，让两侧都感知到终止信号。
            """
            if shared_stop.is_set():
                return True
            if termination_fn():
                shared_stop.set()
                return True
            return False

        left_ok, right_ok = await asyncio.gather(
            self.servo_cartesian(
                side='left',
                velocity_fn=velocity_fn_left,
                termination_fn=termination_with_broadcast,
                frame_id=frame_id,
                rate_hz=rate_hz,
            ),
            self.servo_cartesian(
                side='right',
                velocity_fn=velocity_fn_right,
                termination_fn=termination_with_broadcast,
                frame_id=frame_id,
                rate_hz=rate_hz,
            ),
        )
        return left_ok and right_ok