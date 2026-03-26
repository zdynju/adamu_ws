import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class BoxTfBridge(Node):
    def __init__(self):
        super().__init__('box_tf_bridge')
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 订阅刚才 URDF 里配置的 Odom 话题
        self.subscription = self.create_subscription(
            Odometry,
            '/simulator/box_state',
            self.odom_callback,
            10
        )
        self.get_logger().info("箱子 TF 桥接节点已启动，正在监听 /simulator/box_state ...")

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'               # 你的全局基准坐标系
        t.child_frame_id = 'target_box_fixed'     # 生成的箱子坐标系名称
        
        # 位置提取
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # 姿态提取
        t.transform.rotation = msg.pose.pose.orientation

        # 广播 TF
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = BoxTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()