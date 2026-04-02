import asyncio
import threading
import rclpy
import math 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointJog

class ServoFunctionalTest(Node):
    def __init__(self):
        super().__init__(
            'servo_functional_test',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # Publisher & Clients
        self._start_client = self.create_client(Trigger, '/right_arm/servo_node/start_servo')
        self._stop_client = self.create_client(Trigger, '/right_arm/servo_node/stop_servo')
        self._twist_pub = self.create_publisher(TwistStamped, '/right_arm/servo_node/delta_twist_cmds', 10)
        self._joint_jog_pub = self.create_publisher(JointJog, '/right_arm/servo_node/delta_joint_cmds', 10)
        self._joint_pub = self.create_publisher(JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # 关节列表（请确保顺序与 MoveIt 组一致）
        self.joint_names = [
            'shoulderPitch_Right', 'shoulderRoll_Right', 'shoulderYaw_Right',
            'elbow_Right', 'wristYaw_Right', 'wristPitch_Right', 'wristRoll_Right'
        ]

    def _publish_twist(self, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world' 
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz
        self._twist_pub.publish(msg)

    def _publish_joint_jog(self, name, vel):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [name]
        msg.velocities = [float(vel)]
        self._joint_jog_pub.publish(msg)

    async def activate_servo(self):
        self.get_logger().info('▶️ 激活 Servo...')
        self._start_client.wait_for_service()
        res = await self._start_client.call_async(Trigger.Request())
        return res.success

    async def deactivate_servo(self):
        self._publish_twist() 
        await self._stop_client.call_async(Trigger.Request())
        self.get_logger().info('⏸️ Servo 已停止')

    # --- 🧪 功能测试项目 ---

    async def test_cartesian_circle(self, radius=0.1, speed=2.0, duration=10.0):
        """测试末端在 YZ 平面画圆"""
        self.get_logger().info(f'🧪 测试: 笛卡尔圆周运动 (R={radius})')
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            # 通过正弦/余弦控制速度分量
            vy = radius * math.cos(elapsed * speed) * speed
            vz = radius * math.sin(elapsed * speed) * speed
            self._publish_twist(vy=vy, vz=vz)
            await asyncio.sleep(0.02) # 匹配你修改后的频率

    async def test_all_joints_scan(self):
        """逐一转动每个关节，验证映射"""
        for name in self.joint_names:
            self.get_logger().info(f'🧪 测试关节: {name}')
            # 正转 1秒
            st = self.get_clock().now()
            while (self.get_clock().now() - st).nanoseconds < 1.0 * 1e9:
                self._publish_joint_jog(name, 0.2)
                await asyncio.sleep(0.02)
            # 反转 1秒
            st = self.get_clock().now()
            while (self.get_clock().now() - st).nanoseconds < 1.0 * 1e9:
                self._publish_joint_jog(name, -0.2)
                await asyncio.sleep(0.02)

    async def test_velocity_step(self, axis='y'):
        """测试速度阶跃响应 (0.1 -> -0.1 -> 0)"""
        self.get_logger().info(f'🧪 测试: {axis} 方向速度阶跃')
        speeds = [0.5, -0.5, 0.0]
        for s in speeds:
            self.get_logger().info(f'  >> 当前速度: {s}')
            st = self.get_clock().now()
            while (self.get_clock().now() - st).nanoseconds < 2.0 * 1e9:
                if axis == 'y': self._publish_twist(vy=s)
                else: self._publish_twist(vx=s)
                await asyncio.sleep(0.02)

    async def run_test_suite(self):
        if not await self.activate_servo(): return
        
        # --- 在这里选择你想跑的测试 ---
        await self.test_all_joints_scan()    # 测试 1: 关节扫描
        await asyncio.sleep(1.0)
        
        await self.test_velocity_step('y')   # 测试 2: 速度阶跃
        await asyncio.sleep(1.0)
        
        await self.test_cartesian_circle()   # 测试 3: 画圆
        
        await self.deactivate_servo()

async def main_async(node):
    await node.run_test_suite()

def main():
    rclpy.init()
    node = ServoFunctionalTest()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    asyncio.get_event_loop().run_until_complete(main_async(node))
    rclpy.shutdown()

if __name__ == '__main__':
    main()