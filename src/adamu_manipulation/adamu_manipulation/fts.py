from adamu_manipulation.fts_processor import FTSProcessor
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

class DualArmFTSMonitor(Node):
    def __init__(self, fts_processor):
        super().__init__('dual_arm_fts_monitor')
        
        self.fts_prs = fts_processor
        
        self.timer = self.create_timer(0.1, self.display_dashboard)
        self.get_logger().info('🚀 双臂及灵巧手 FTS 实时监控面板已就绪')

    def display_dashboard(self):
        # 1. 获取手腕数据
        l_raw = self.fts_prs.get_original_force('left')
        r_raw = self.fts_prs.get_original_force('right')
        l_std = self.fts_prs.get_standardized_force('left')
        r_std = self.fts_prs.get_standardized_force('right')

        # 获取手指数据 (调用新版接口，返回的是包含 'force' 和 'frame' 的字典)
        l_fingers = self.fts_prs.get_all_finger_forces('left')
        r_fingers = self.fts_prs.get_all_finger_forces('right')

        # 只要手腕数据没拿到，就先跳过，防止启动瞬间没 TF 报错
        if any(x is None for x in [l_raw, r_raw, l_std, r_std]):
            return

        # 🌟 核心修改点：适配新的数据结构 v['force'][2]
        # 提取手指的 Z 轴力 (法向按压力)，如果暂无数据则默认为 0
        lf_z = {k: v['force'][2] for k, v in l_fingers.items()} if l_fingers else {k: 0.0 for k in ['thumb', 'index', 'middle', 'ring', 'pinky']}
        rf_z = {k: v['force'][2] for k, v in r_fingers.items()} if r_fingers else {k: 0.0 for k in ['thumb', 'index', 'middle', 'ring', 'pinky']}

        # 2. 打印面板 (使用 \033c 清屏实现原地刷新)
        print("\033c", end="") 
        print("======================================================")
        print("      🤖 Adam_U 双臂及灵巧手受力实时监控 (10Hz)       ")
        print("======================================================")
        print("【手腕原始受力 (Raw)】")
        print(f" 🟢 左臂 | Fx: {l_raw[0]:>7.2f} | Fy: {l_raw[1]:>7.2f} | Fz: {l_raw[2]:>7.2f}")
        print(f" 🔵 右臂 | Fx: {r_raw[0]:>7.2f} | Fy: {r_raw[1]:>7.2f} | Fz: {r_raw[2]:>7.2f}")
        print("------------------------------------------------------")
        print("【手腕重力补偿后 (Std - 需外部去皮)】")
        print(f" 🟢 左臂 | Fx: {l_std[0]:>7.2f} | Fy: {l_std[1]:>7.2f} | Fz: {l_std[2]:>7.2f}")
        print(f" 🔵 右臂 | Fx: {r_std[0]:>7.2f} | Fy: {r_std[1]:>7.2f} | Fz: {r_std[2]:>7.2f}")
        print("======================================================")
        print("【🖐️ 指腹法向接触力 (Z轴)】")
        print(f" 🟢 左手 | 拇:{lf_z['thumb']:>6.2f} 食:{lf_z['index']:>6.2f} 中:{lf_z['middle']:>6.2f} 无:{lf_z['ring']:>6.2f} 小:{lf_z['pinky']:>6.2f}")
        print(f" 🔵 右手 | 拇:{rf_z['thumb']:>6.2f} 食:{rf_z['index']:>6.2f} 中:{rf_z['middle']:>6.2f} 无:{rf_z['ring']:>6.2f} 小:{rf_z['pinky']:>6.2f}")
        print("======================================================")
        print(" 操作提示: 按 Ctrl+C 退出监控面板")
        print("======================================================")

# ==========================================
# 正确的启动方式：单线程执行器挂载两个节点
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    
    # 初始化数据处理节点 (传入质量参数 0.54kg)
    fts_processor = FTSProcessor(0.54)
    # 初始化监控面板节点，并把 processor 传进去让它拿数据
    monitor = DualArmFTSMonitor(fts_processor)
    
    # 只需要 SingleThreadedExecutor 即可，ROS 2 底层会轮询调度这两个节点的回调
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(fts_processor)
    executor.add_node(monitor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 优雅关闭
        executor.remove_node(fts_processor)
        executor.remove_node(monitor)
        fts_processor.destroy_node()
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()