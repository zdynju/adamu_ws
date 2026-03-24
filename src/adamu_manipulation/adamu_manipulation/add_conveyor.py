import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

class ConveyorPublisher(Node):
    def __init__(self):
        super().__init__('conveyor_obstacle_publisher')
        
        # 创建发布者，目标话题是 /planning_scene
        self.publisher_ = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        self.get_logger().info('Waiting for /planning_scene subscriber...')
        # 核心改动：循环检查有没有人连接上这个话题
        import time
        while self.publisher_.get_subscription_count() == 0:
            self.get_logger().info('⏳ 还没人监听，继续等...', throttle_duration_sec=1.0)
            time.sleep(0.5)

        self.get_logger().info('✅ 发现订阅者！网络建立成功。')
        time.sleep(0.5) # 发现后稍微稳 0.5 秒，确保 DDS 连接彻底建立
        self.publish_obstacle()

    def publish_obstacle(self):
        # 1. 定义传送带的尺寸
        size_x = 0.3 * 2   # 0.6
        size_y = 10.0 * 2 # 设置为 20.0 以确保足够覆盖机器人工作空间
        size_z = 0.04 * 2  # 0.08

        # 2. 定义传送带的位姿
        pose = Pose()
        pose.position.x = 0.45
        pose.position.y = 0.0
        pose.position.z = 1.05
        pose.orientation.w = 1.0 # 无旋转

        # 3. 构建 CollisionObject 消息
        co = CollisionObject()
        co.header.frame_id = "world" 
        co.id = "conveyor_belt"
        
        # 定义形状为 BOX
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size_x, size_y, size_z]

        co.primitives.append(primitive)
        co.primitive_poses.append(pose)
        
        # 设置操作类型为 ADD 
        co.operation = CollisionObject.ADD

        # 4. 构建 PlanningScene 消息
        planning_scene_msg = PlanningScene()
        

        # 设置为 diff 模式，True 表示这是一个增量更新，不会清除场景中已有的其他物体或机器人状态
        planning_scene_msg.is_diff = True
        
        # 将碰撞对象放入 world 字段
        planning_scene_msg.world.collision_objects.append(co)

        # 5. 发布
        self.publisher_.publish(planning_scene_msg)
        self.get_logger().info(f'Published conveyor obstacle: {co.id}')

def main(args=None):
    rclpy.init(args=args)
    node = ConveyorPublisher()

    rclpy.spin_once(node, timeout_sec=1)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    
#  ros2 topic pub /conveyor_velocity_controller/commands std_msgs/Float64MultiArray "data: [0.0]"