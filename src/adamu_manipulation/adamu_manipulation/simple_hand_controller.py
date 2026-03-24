"""
simple_hand_controller.py — 简单手部控制接口 (纯净关节控制版)

只保留直接关节控制，移除所有协同导纳接口。
提供：
  - open()：张开手
  - close()：闭合手（预设抓取姿态）
  - edge_grasp()：边棱抓取姿态
  - set_joints()：直接指定 12 个关节角度（带插值平滑）
  - set_joints_immediate()：立即下发 12 个关节角度（无插值）
"""

import asyncio
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# ─────────────────────────────────────────────────────────────────────────────
#  预设姿态（12 维关节角，单位 rad）
# ─────────────────────────────────────────────────────────────────────────────

HAND_OPEN = np.array([
    0.0,  0.0,  0.0,  0.0,   # thumb
    0.0,  0.0,               # index
    0.0,  0.0,               # middle
    0.0,  0.0,               # ring
    0.0,  0.0,               # pinky
], dtype=float)

HAND_CLOSE = np.array([
    0.3,  0.5,  0.8,  0.6,   # thumb
    1.2,  0.8,               # index
    1.2,  0.8,               # middle
    1.0,  0.7,               # ring
    0.8,  0.6,               # pinky
], dtype=float)

HAND_EDGE_GRASP = np.array([
    0.2,  0.3,  0.5,  0.4,   # thumb
    0.8,  0.5,               # index
    0.8,  0.5,               # middle
    0.6,  0.4,               # ring
    0.5,  0.3,               # pinky
], dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
#  简单手部控制器 (Node 子类)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleHandController(Node):
    """简单手部控制接口 Node，纯关节位置控制。"""

    def __init__(self, side: str, node_name: str = None):
        assert side in ('left', 'right'), "side 必须是 'left' 或 'right'"
        
        if node_name is None:
            node_name = f'{side}_simple_hand_controller'
            
        super().__init__(node_name)
        
        self.side = side

        # 关节角度发布器
        joint_topic = f'/{side}_hand_controller/commands'
        self._joint_pub = self.create_publisher(Float64MultiArray, joint_topic, 10)

        self.get_logger().info(
            f'SimpleHandController [{side}] 初始化完成\n'
            f'  节点名: {self.get_name()}\n'
            f'  关节话题: {joint_topic}'
        )

    # ── 预设动作 ─────────────────────────────────────────────────────────────

    async def open(self, duration: float = 1.0):
        """张开手。"""
        await self._interpolate_to(HAND_OPEN, duration)
        self.get_logger().info(f'[{self.side}] 手已张开')

    async def close(self, duration: float = 1.5):
        """闭合手（标准抓取姿态）。"""
        await self._interpolate_to(HAND_CLOSE, duration)
        self.get_logger().info(f'[{self.side}] 手已闭合')

    async def edge_grasp(self, duration: float = 1.2):
        """边棱抓取姿态。"""
        await self._interpolate_to(HAND_EDGE_GRASP, duration)
        self.get_logger().info(f'[{self.side}] 手已到达边棱抓取姿态')

    # ── 直接控制 ─────────────────────────────────────────────────────────────

    async def set_joints(self, joint_angles: np.ndarray, duration: float = 1.0):
        """直接指定 12 维关节角度，带插值过渡。"""
        assert len(joint_angles) == 12, '必须是 12 维关节角度'
        await self._interpolate_to(np.array(joint_angles, dtype=float), duration)

    def set_joints_immediate(self, joint_angles: np.ndarray):
        """立即发布关节角度，无插值（注意：可能会引起电机突变跳跃）。"""
        assert len(joint_angles) == 12
        msg      = Float64MultiArray()
        msg.data = [float(v) for v in joint_angles]
        self._joint_pub.publish(msg)
        # 更新缓存以保证下一次插值起点正确
        self._current_joints = np.array(joint_angles, dtype=float) 

    # ── 内部插值逻辑 ─────────────────────────────────────────────────────────────

    async def _interpolate_to(self, target: np.ndarray, duration: float, freq: float = 50.0):
        """从当前位置线性插值到目标角度。"""
        steps    = max(1, int(duration * freq))
        dt       = duration / steps
        start    = self._current_joints.copy()

        for i in range(1, steps + 1):
            t   = i / steps
            cmd = start + t * (target - start)

            msg      = Float64MultiArray()
            msg.data = cmd.tolist()
            self._joint_pub.publish(msg)

            await asyncio.sleep(dt)

        self._current_joints = target.copy()

    @property
    def _current_joints(self) -> np.ndarray:
        """当前关节角度缓存。"""
        if not hasattr(self, '_joints_cache'):
            self._joints_cache = HAND_OPEN.copy()
        return self._joints_cache

    @_current_joints.setter
    def _current_joints(self, val: np.ndarray):
        self._joints_cache = val.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  双手控制器（统一管理左右手）
# ─────────────────────────────────────────────────────────────────────────────

class DualHandController:
    """管理左右手两个 Node 实例。"""

    def __init__(self):
        self.left  = SimpleHandController(side='left')
        self.right = SimpleHandController(side='right')

    async def open_both(self, duration: float = 1.0):
        await asyncio.gather(
            self.left.open(duration),
            self.right.open(duration),
        )

    async def close_both(self, duration: float = 1.5):
        await asyncio.gather(
            self.left.close(duration),
            self.right.close(duration),
        )

    async def edge_grasp_both(self, duration: float = 1.2):
        await asyncio.gather(
            self.left.edge_grasp(duration),
            self.right.edge_grasp(duration),
        )

    def get(self, side: str) -> SimpleHandController:
        return self.left if side == 'left' else self.right
    

def main(args=None):
    # 1. 初始化 ROS 2 
    rclpy.init(args=args)

    # 2. 实例化双手控制器 (内部会自动创建 left 和 right 两个 Node)
    hands = DualHandController()

    # 3. 定义异步测试序列
    async def test_sequence():
        print("\n=== 开始手部控制测试 ===")
        
        # 给一点时间让 ROS 2 底层完成话题发现 (Discovery)
        await asyncio.sleep(0.5)
        
        print("\n[动作 1] 双手同时张开 (耗时 2 秒)")
        await hands.open_both(duration=2.0)
        await asyncio.sleep(1.0)  # 动作完成后停顿 1 秒
        
        print("\n[动作 2] 双手同时闭合 (耗时 1.5 秒)")
        await hands.close_both(duration=1.5)
        await asyncio.sleep(1.0)
        
        print("\n[动作 3] 双手边棱抓取 (耗时 1.2 秒)")
        await hands.edge_grasp_both(duration=1.2)
        await asyncio.sleep(1.0)

        print("\n[动作 4] 左右手不同步动作：左手张开，右手闭合")
        # 使用 asyncio.gather 并发执行两个不同的动作
        await asyncio.gather(
            hands.left.open(duration=1.0),
            hands.right.close(duration=1.0)
        )
        await asyncio.sleep(1.0)
        
        print("\n=== 测试完成 ===")

    # 4. 运行异步事件循环
    try:
        asyncio.run(test_sequence())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        # 5. 安全清理节点和关闭 ROS 2
        hands.left.destroy_node()
        hands.right.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()