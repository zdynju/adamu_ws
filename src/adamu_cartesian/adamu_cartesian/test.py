import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from controller_manager_msgs.srv import SwitchController
import time

class ComplianceTest(Node):
    def __init__(self):
        super().__init__('compliance_test')
        
        # 1. 订阅当前位姿（由控制器发布）
        self.current_pose = None
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/left_arm_cartesian_compliance_controller/current_pose',
            self.pose_callback,
            10)
            
        # 2. 发布目标位姿
        self.target_pub = self.create_publisher(
            PoseStamped,
            '/left_arm_cartesian_compliance_controller/target_frame',
            10)

        # 3. 切换控制器的客户端
        self.switch_cli = self.create_client(SwitchController, '/controller_manager/switch_controller')

    def pose_callback(self, msg):
        self.current_pose = msg

    def switch_to_compliance(self):
        while self.current_pose is None:
            self.get_logger().info('正在等待当前位姿数据...')
            rclpy.spin_once(self, timeout_sec=0.5)

        # 🚨 核心步骤：在切换前，先把目标点设为当前点，防止手臂“瞬移”抽风
        self.target_pub.publish(self.current_pose)
        self.get_logger().info('已同步初始位姿，准备切换...')

        req = SwitchController.Request()
        req.activate_controllers = ['left_arm_cartesian_compliance_controller', 'left_motion_control_handle']
        req.deactivate_controllers = ['left_arm_controller']
        req.strictness = 2 # STRICT
        
        self.switch_cli.call_async(req)
        self.get_logger().info('切换请求已发送！')

    def test_rotation(self):
        # 切换成功后，测试旋转角度（绕 Y 轴旋转约 45 度）
        time.sleep(2.0)
        msg = self.current_pose
        # 四元数控制角度：绕 Y 轴旋转
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.382
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.924
        
        self.get_logger().info('正在执行角度翻转测试...')
        self.target_pub.publish(msg)

def main():
    rclpy.init()
    node = ComplianceTest()
    node.switch_to_compliance()
    
    # 运行测试
    try:
        # 保持运行，让它监听并执行
        node.test_rotation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()