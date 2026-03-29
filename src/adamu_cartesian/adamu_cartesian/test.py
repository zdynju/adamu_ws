import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import time
from rclpy.parameter import Parameter

class TargetFrameTest(Node):
    def __init__(self):
        super().__init__('test_target_frame')
        
        # 1. 设置发布者 
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.target_pub = self.create_publisher(
            PoseStamped,
            '/left_arm_cartesian_compliance_controller/target_frame',
            10)
            
        # 2. 设置 TF 监听器（用于无缝接管）
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("节点已启动，正在等待 TF 树就绪...")

    def get_current_pose(self):
        # 尝试获取 torso 到 left_hand_tcp 的当前变换
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'torso', 'left_hand_tcp', now, timeout=rclpy.duration.Duration(seconds=2.0))
            
            pose = PoseStamped()
            pose.header.frame_id = 'torso'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().error(f"获取 TF 失败: {e}")
            return None

    def run_test(self):
        # 1. 获取初始位置并原位锁死
        start_pose = None
        while start_pose is None and rclpy.ok():
            start_pose = self.get_current_pose()
            time.sleep(0.5)
            
        self.get_logger().info("已获取初始位姿，原位锁死中...")
        self.target_pub.publish(start_pose)
        time.sleep(2.0) # 等待 2 秒，让你确认手臂没抽风

        # 2. 修改目标位置：Z 轴向上移动 5 厘米
        target_pose = start_pose
        target_pose.pose.position.z += 0.05
        
        self.get_logger().info("指令下发：沿 Z 轴向上移动 5cm ...")
        self.target_pub.publish(target_pose)
        time.sleep(3.0) # 等待手臂移动到位

        # 3. 回到初始位置
        self.get_logger().info("指令下发：降回原位 ...")
        start_pose.header.stamp = self.get_clock().now().to_msg() # 更新时间戳
        start_pose.pose.position.z -= 0.05 # 减回去
        self.target_pub.publish(start_pose)
        
        self.get_logger().info("轨迹测试完成！")

def main(args=None):
    rclpy.init(args=args)
    node = TargetFrameTest()
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()