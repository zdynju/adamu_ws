#!/usr/bin/env python3
"""
双臂控制器独立切换工具
用于在 MoveIt 2 (JointTrajectoryController) 和 FZI 笛卡尔控制器 之间进行切换。
"""

import sys
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController


class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__('dual_arm_controller_switcher')
        
        # 创建服务客户端
        self.client = self.create_client(
            SwitchController, 
            '/controller_manager/switch_controller'
        )

        # 定义控制器组
        self.moveit_controllers = [
            'left_arm_controller',
            'right_arm_controller'
        ]
        self.cartesian_controllers = [
            'left_arm_cartesian_compliance_controller',
            'right_arm_cartesian_motion_controller'
        ]

    def switch(self, mode: str) -> bool:
        """
        执行切换逻辑
        :param mode: 'cartesian' 或 'moveit'
        """
        self.get_logger().info('正在等待 /controller_manager/switch_controller 服务...')
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('连接 controller_manager 服务超时！请检查 controller_manager 是否正常运行。')
            return False

        req = SwitchController.Request()
        req.strictness = SwitchController.Request.STRICT

        # 根据用户选择配置激活/停止列表
        if mode == 'cartesian':
            req.activate_controllers = self.cartesian_controllers
            req.deactivate_controllers = self.moveit_controllers
            self.get_logger().info('请求切换至: 【FZI 笛卡尔控制模式】 (关闭 MoveIt 2)')
            
        elif mode == 'moveit':
            req.activate_controllers = self.moveit_controllers
            req.deactivate_controllers = self.cartesian_controllers
            self.get_logger().info('请求切换至: 【MoveIt 2 关节控制模式】 (关闭 FZI 笛卡尔)')
            
        else:
            self.get_logger().error('未知的切换模式！')
            return False

        # 调用服务
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        try:
            response = future.result()
            if response.ok:
                self.get_logger().info('✅ 控制器切换成功！')
                return True
            else:
                self.get_logger().error('❌ 控制器切换失败（服务返回了 False）。可能是控制器未加载或硬件接口冲突。')
                return False
        except Exception as e:
            self.get_logger().error(f'服务调用异常: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    switcher = ControllerSwitcher()
    
    print("\n" + "="*40)
    print("🤖 双臂控制器切换工具")
    print("="*40)
    print("请选择要切换的模式:")
    print("  [1] 切换到 FZI 笛卡尔控制 (用于夹取/阻抗控制)")
    print("  [2] 切换回 MoveIt 2 关节控制 (用于大范围轨迹规划)")
    print("  [q] 退出")
    
    choice = input("\n请输入选项 (1/2/q): ").strip()
    
    if choice == '1':
        switcher.switch(mode='cartesian')
    elif choice == '2':
        switcher.switch(mode='moveit')
    elif choice.lower() == 'q':
        print("已退出。")
    else:
        print("无效的输入。")
        
    switcher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()