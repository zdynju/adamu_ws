import os
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
import tf2_ros
import tf2_geometry_msgs
import pinocchio as pin
from adamu_manipulation.fts_processor import FTSProcessor

class SynergyAdmittanceControllerV9(Node):
    def __init__(self, fts_processor: FTSProcessor):
        super().__init__('synergy_admittance_controller_v9')
        self.get_logger().info("协同导纳控制器正在启动...")

        # 依赖注入 FTS 实例
        self.fts = fts_processor

        # --- 基础配置 ---
        model_path = self.declare_parameter('pca_model', '/home/zhoudaoyuan/adamu_ws/adam_synergy_v4_model.npz').value
        urdf_path = self.declare_parameter('urdf_path', '/home/zhoudaoyuan/adamu_ws/src/adamu_description/urdf/adam_u.urdf').value
        self.side = self.declare_parameter('hand_side', 'R').value # 'R' 或 'L'
        self.side_full = 'right' if self.side == 'R' else 'left'

        # --- Pinocchio 模型 ---
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        prefix = f"{self.side}_" # 'R_' 或 'L_'
        self.controlled_joint_names = [
            f"{prefix}thumb_MCP_joint1", f"{prefix}thumb_MCP_joint2", f"{prefix}thumb_PIP_joint", f"{prefix}thumb_DIP_joint",
            f"{prefix}index_MCP_joint",  f"{prefix}index_DIP_joint",
            f"{prefix}middle_MCP_joint", f"{prefix}middle_DIP_joint",
            f"{prefix}ring_MCP_joint",   f"{prefix}ring_DIP_joint",
            f"{prefix}pinky_MCP_joint",  f"{prefix}pinky_DIP_joint"
        ]
        self.q_indices = []
        self.v_indices = []  
        
        for name in self.controlled_joint_names:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                # 位置索引
                self.q_indices.append(self.model.joints[joint_id].idx_q)
                # 速度/雅可比索引
                self.v_indices.append(self.model.joints[joint_id].idx_v)
            else:
                self.get_logger().error(f"严重错误: 在 URDF 中找不到关节 {name}")
        self.finger_weights = {'thumb': 1.0, 'index': 1.0, 'middle': 0.8, 'ring': 0.4, 'pinky': 0.2}
        # 在你 __init__ 的最后
        print(f"✓ v_indices: {self.v_indices}")
        print(f"✓ model.nv: {self.model.nv}")
        print(f"✓ max(v_indices): {max(self.v_indices)}")

        if max(self.v_indices) >= self.model.nv:
            print(f"❌ 错误！v_indices 超出范围")
        # --- PCA 降维模型 ---
        npz = np.load(model_path)
        
        self.P = npz['pca_components']
        self.n_pc = self.P.shape[0]
        self.pca_mean = npz['pca_mean']
        self.scaler_scale = npz['scaler_scale']
        self.scaler_mean = npz['scaler_mean']
        self.joint_limits_min = npz['joint_limits_min']
        self.joint_limits_max = npz['joint_limits_max']

        # 预计算协同基底矩阵 B
        self.B = np.diag(self.scaler_scale) @ self.P.T

        # --- 导纳动力学参数 ---
        self.M = np.eye(self.n_pc) * 0.05
        self.K_base = np.eye(self.n_pc) * 15.0 
        self.D = 2.0 * np.sqrt(self.M @ self.K_base)

        self.force_threshold = 0.5
        self.alpha = 0.2
        self.contact_debounce = 0

        self.z_ref = np.zeros(self.n_pc)
        self.z_cur = np.zeros(self.n_pc)
        self.z_vel = np.zeros(self.n_pc)
        self.q_cur_rad = self.scaler_mean.copy()
        
        self.tau_z_filtered = np.zeros(self.n_pc)
        self.lpf_alpha = 0.15

        # --- 通讯接口 ---
        self.create_subscription(Float64MultiArray, f'/adam/{self.side_full}_synergy_command', self.cmd_cb, 10)
        pub_topic = '/right_hand_controller/commands' if self.side == 'R' else '/left_hand_controller/commands'
        self.cmd_pub = self.create_publisher(Float64MultiArray, pub_topic, 10)

        # 200Hz 控制循环
        self.dt = 0.005
        self.create_timer(self.dt, self.control_step)

        self.get_logger().info(
            f"Adam_U V9.0 协同导纳大脑 [{self.side_full}] 启动成功！\n"
            f"   物理关节数: 12\n"
            f"   协同维度: {self.n_pc}D\n"
            f"   基础刚度: {self.K_base[0,0]} N/m\n"
            f"   控制心跳: {int(1.0/self.dt)} Hz"
        )

    def cmd_cb(self, msg):
        """接收上层抓取规划意图"""
        if len(msg.data) == self.n_pc:
            print(msg.data)
            self.z_ref = np.clip(np.array(msg.data), -4.0, 4.0)

    def control_step(self):
        """核心 200Hz 协同导纳解算"""

        q_full = pin.neutral(self.model)
        
        # 🌟 修复点：将 12 维的受控手指角度，精准填入到全身状态的对应索引中
        for i, idx in enumerate(self.q_indices):
            q_full[idx] = self.q_cur_rad[i]

        # 🌟 修改点：把拼接好的 q_full 喂给物理引擎
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.computeJointJacobians(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        tau_z_raw = np.zeros(self.n_pc)
        max_contact_force = 0.0
        # if not hasattr(self, 'diag_count'):
        #     self.diag_count = 0

        # if self.diag_count == 0:  # 只打印一次，不要污染日志
        #     # 验证传感器数据
        #     finger_forces_dict = self.fts.get_all_finger_forces(self.side_full)
        #     for finger, data in finger_forces_dict.items():
        #         f = data['force']
        #         print(f"{finger}: force = {f}, norm = {np.linalg.norm(f):.4f}")
            
        #     # 验证雅可比
        #     sensor_frame = finger_forces_dict['thumb']['frame']
        #     print(sensor_frame)
        #     sensor_id = self.model.getFrameId(sensor_frame)
        #     import pinocchio as pin
        #     J_full = pin.getFrameJacobian(self.model, self.data, sensor_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        #     J_hand = J_full[:, self.v_indices]
        #     J_z = J_hand @ self.B
        #     print(f"J_hand shape: {J_hand.shape}, norm: {np.linalg.norm(J_hand):.4f}")
        #     print(f"J_z shape: {J_z.shape}, norm: {np.linalg.norm(J_z):.4f}")

        # self.diag_count += 1
        # 调用 FTS 实例获取数据
        finger_forces_dict = self.fts.get_all_finger_forces(self.side_full)

        for finger in self.fingers:
            data = finger_forces_dict[finger]
            f_sensor_local = data['force']
            sensor_frame_name = data['frame']

            force_mag = np.linalg.norm(f_sensor_local)
            # print(force_mag)
            if force_mag > max_contact_force:
                max_contact_force = force_mag

            # 忽略零力矩，节约计算量
            if force_mag < 0.01:
                continue

            try:
                if not self.model.existFrame(sensor_frame_name):
                    self.get_logger().error(
                        f"❌ 雅可比严重错误: URDF 中根本不存在名为 '{sensor_frame_name}' 的坐标系！请检查 FTSProcessor 输出的名字。", 
                        throttle_duration_sec=2.0
                    )
                    continue  # 找不到就跳过这根手指，绝不让系统崩溃
                # 刚体变换：将力从传感器坐标系转换至世界坐标系
                sensor_id = self.model.getFrameId(sensor_frame_name)
                R_world_sensor = self.data.oMf[sensor_id].rotation
                
                delta_F_local = - f_sensor_local
                delta_F_world = R_world_sensor @ delta_F_local
                
                # J_z = J_geo @ B (无伪逆运算的极简降维映射)
                # 获取该传感器坐标系的全尺寸雅可比矩阵 (3 x 44)
                # 获取完整雅可比（先不切片）
                J_geo_full = pin.getFrameJacobian(self.model, self.data, sensor_id, 
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

                # 检查形状
                if J_geo_full.shape[0] < 3:
                    print(f"警告: 雅可比只有 {J_geo_full.shape[0]} 行")
                    continue
                print('right')
                # 安全地切片
                J_partial = J_geo_full[:3, :]  # (3, nv)
                J_geo_hand = J_partial[:, self.v_indices]  # (3, 12)
                
                # 现在矩阵维度完美匹配！ (3 x 12) @ (12 x 4) -> 结果是一个清爽的 (3 x 4) 协同雅可比
                J_z = J_geo_hand @ self.B
                
                tau_z_raw += self.finger_weights[finger] * (J_z.T @ delta_F_world)
                
            except Exception as e:
                # print(f"[WARN] Force compute failed for {sensor_frame_name}: {e}")
                # continue
                pass
           

        # 信号滤波与状态机
        self.tau_z_filtered = (1 - self.lpf_alpha) * self.tau_z_filtered + self.lpf_alpha * tau_z_raw

        if max_contact_force > self.force_threshold:
            self.contact_debounce = min(self.contact_debounce + 1, 15)
        else:
            self.contact_debounce = max(self.contact_debounce - 1, 0)

        active_K = self.alpha * self.K_base if self.contact_debounce >= 5 else self.K_base

        # M-D-K 积分
        z_acc = np.linalg.inv(self.M) @ (self.tau_z_filtered - self.D @ self.z_vel - active_K @ (self.z_cur - self.z_ref))
        self.z_vel += z_acc * self.dt
        self.z_cur += self.z_vel * self.dt
        
        self.z_vel = np.clip(self.z_vel, -4.0, 4.0)
        self.z_cur = np.clip(self.z_cur, -6.0, 6.0)

        # 物理空间重建
        q_scaled = self.z_cur @ self.P + self.pca_mean
        q_rad_unbounded = q_scaled * self.scaler_scale + self.scaler_mean
        self.q_cur_rad = np.clip(q_rad_unbounded, self.joint_limits_min, self.joint_limits_max)

        msg = Float64MultiArray()
        msg.data = self.q_cur_rad.tolist()
        # print(msg.data)
        self.cmd_pub.publish(msg)

# ==============================================================================
# 3. 启动器：利用 MultiThreadedExecutor 并行调度
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    fts_node = FTSProcessor(mass=0.5)
    controller_node = SynergyAdmittanceControllerV9(fts_processor=fts_node)
    
    # 启用多线程，确保高速回调和稳定计算频率互不干扰
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(fts_node)
    executor.add_node(controller_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("收到退出信号，系统安全停止...")
    finally:
        executor.shutdown()
        fts_node.destroy_node()
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()