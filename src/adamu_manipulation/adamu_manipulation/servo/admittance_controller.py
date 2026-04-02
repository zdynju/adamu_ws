import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import matplotlib.pyplot as plt


@dataclass
class AdmittanceParams:
    """
    导纳控制的参数（全是物理量）
    
    标准导纳模型：
        M·dv/dt + D·v + K·e = F
    
    其中：
        M: 虚拟质量 (kg) - 越大越"重"，加速越慢
        D: 虚拟阻尼 (N·s/m) - 越大越"粘"，衰减越快
        K: 虚拟刚度 (N/m) - 越大末端越"硬"，位置误差越难存在
        dt: 控制周期 (s)
    """
    
    # ========== 虚拟动力学参数 ==========
    M: float = 5.0          # 虚拟质量 (kg) - 典型值：3-10kg
    D: float = 20.0         # 虚拟阻尼 (N·s/m) - 应该满足D = 2*sqrt(K*M)
    K: float = 100.0        # 虚拟刚度 (N/m) - 通常100-1000 N/m
    
    dt: float = 0.001       # 控制周期 (s) - 1kHz是标准频率
    
    # ========== 死区参数（非常重要！避免噪声） ==========
    f_deadband: float = 1.0   # 力死区 (N) - 小于此值忽略
    t_deadband: float = 0.05  # 力矩死区 (N·m)
    
    # ========== 速度限制 ==========
    v_max_linear: float = 0.1    # 最大线速度 (m/s) - 安全限制
    v_max_angular: float = 0.5   # 最大角速度 (rad/s)
    
    # ========== 滤波参数 ==========
    filter_alpha: float = 0.3    # 低通滤波系数 - 越小越平滑但延迟越大


class SingleArmAdmittanceController:
    """
    单臂导纳控制器（核心算法）
    
    ======================== 导纳模型 ========================
    
    连续模型：
        M·dv/dt + D·v + K·e = F         ... (1)
    
    其中：
        v: 末端执行器速度 (m/s 或 rad/s)
        e: 位置误差 (m 或 rad)
        F: 接触力 (N 或 N·m)
    
    物理意义：
        - M·dv/dt: 惯性项，质量越大加速越慢
        - D·v: 阻尼项，速度越快阻力越大
        - K·e: 刚度项，位置偏差越大恢复力越大
        - F: 外界力，驱动末端运动
    
    ======================== 离散化过程 ========================
    
    使用后向欧拉法（向后差分）离散化：
    
    Step 1: 将导数近似为差分
        dv/dt ≈ (v_k - v_{k-1})/dt        ... (2)
    
    Step 2: 代入公式(1)
        M·(v_k - v_{k-1})/dt + D·v_k + K·e_k = F_k
    
    Step 3: 求解 v_k
        M·v_k/dt - M·v_{k-1}/dt + D·v_k + K·e_k = F_k
        (M/dt + D)·v_k = M/dt·v_{k-1} + F_k - K·e_k
        v_k = [M·v_{k-1} + dt·(F_k - K·e_k)] / (M + D·dt)
    
    Step 4: 定义系数
        A = M / (M + D·dt)                  ... (3)
        B = dt / (M + D·dt)                 ... (4)
        
    最终形式：
        v_k = A·v_{k-1} + B·(F_k - K·e_k)  ... (5)
    
    这就是代码中的核心更新方程！
    
    ======================== 参数调试 ========================
    
    1. 选择虚拟质量 M：
       - M ≈ 实际物体质量
       - 例如搬一个2kg的箱子，设M=2-5kg
       - M越大，响应越慢
    
    2. 选择虚拟阻尼 D（临界阻尼）：
       - D = 2·sqrt(K·M)  （推荐）
       - 这样系统不会过度振荡
       - 超调量约4%
    
    3. 选择虚拟刚度 K：
       - K 越小末端越"柔顺"
       - K 越大位置追踪越精确
       - 典型范围：10-1000 N/m
       - 与环境刚度相匹配效果最好
    
    ======================== 公式验证 ========================
    
    检验：当F_k=0, e_k=0时（无力、无误差）
        v_k = A·v_{k-1} + 0 = (M/(M+D·dt))·v_{k-1}
        
        由于 0 < A < 1，速度指数衰减：
        v_k → 0 当 k → ∞  ✓ 物理合理
    
    """
    
    def __init__(self, params: AdmittanceParams):
        """初始化导纳控制器"""
        
        self.p = params
        
        # ========== 预计算离散化系数 ==========
        # 
        # 根据公式(3)和(4)计算：
        #     A = M / (M + D·dt)
        #     B = dt / (M + D·dt)
        #
        # 这两个系数在整个运行过程中不变，所以预先计算以提高效率
        #
        
        denominator = self.p.M + self.p.D * self.p.dt
        
        self.A = self.p.M / denominator
        """
        公式：A = M / (M + D·dt)
        含义：旧速度对新速度的影响系数
        范围：(0, 1) - 越接近0阻尼越强，越接近1阻尼越弱
        """
        
        self.B = self.p.dt / denominator
        """
        公式：B = dt / (M + D·dt)
        含义：力对速度的增益系数
        范围：(0, 1) - 越小外力的影响越弱
        """
        
        self.gain_K = self.p.K * self.p.dt / denominator
        """
        公式：K_gain = K·dt / (M + D·dt)
        含义：位置误差对速度的增益
        单位：(1/s) - 位置误差通过此系数转换为速度
        """
        
        # ========== 等价参数（用于理解） ==========
        
        self.lambda_ = self.p.D * self.p.dt / self.p.M
        """
        定义：λ = D·dt / M
        含义：相对阻尼系数（无量纲）
        通常：λ < 0.5（否则过度阻尼）
        """
        
        # ========== 状态变量初始化 ==========
        
        self.vel = np.zeros(6)
        """当前速度 [vx, vy, vz, ωx, ωy, ωz]"""
        
        self.filtered_wrench = np.zeros(6)
        """滤波后的力/力矩 [fx, fy, fz, τx, τy, τz]"""
        
        self.x0 = None
        """初始位置/姿态（参考点）"""
        
        # ========== 限制向量 ==========
        
        self.deadbands = np.array([
            self.p.f_deadband]*3 + [self.p.t_deadband]*3 )
        """
        死区向量：小于该阈值的信号被忽略
        作用：避免传感器噪声导致的控制抖动
        """
        
        self.v_max = np.array([
            self.p.v_max_linear]*3 +     # 线速度限制
            [self.p.v_max_angular]*3     # 角速度限制
        )
        """速度限制向量：超过此速度会被限幅"""
    
    def reset(self, current_pose: np.ndarray):
        """
        重置控制器状态
        
        参数：
            current_pose: (6,) 当前位姿 [x, y, z, rx, ry, rz]
        
        说明：
            初始化时必须调用此函数，以设定参考位置
        """
        self.x0 = np.array(current_pose, dtype=np.float64)
        self.vel = np.zeros(6, dtype=np.float64)
        self.filtered_wrench = np.zeros(6, dtype=np.float64)
    
    @staticmethod
    def _soft_deadband(value: float, deadband: float) -> float:
        """
        软死区处理（平滑版本）
        
        公式：
            output = value · (1 - exp(-|value|/deadband))
        
        对比硬死区：
            硬死区：if |value| < deadband: return 0; else: return value
            问题：在死区边界处不连续，容易抖动
        
            软死区：使用指数函数平滑过渡
            优势：在全范围内可导，控制更平滑
        
        物理意义：
            - value接近0时，指数项≈1，输出≈0
            - value远大于deadband时，exp项≈0，输出≈value
            - 过渡光滑无跳变
        
        参数：
            value: 输入信号
            deadband: 死区宽度（感受器灵敏度阈值）
        
        返回：
            经过死区处理后的输出
        """
        if deadband < 1e-6:
            return value
        
        # 使用指数函数实现软死区
        # 当 |value|/deadband 很小时，exp(-|value|/deadband) ≈ 1 - |value|/deadband
        # 当 |value|/deadband 很大时，exp(-|value|/deadband) ≈ 0
        return value * (1.0 - np.exp(-np.abs(value) / deadband))
    
    @staticmethod
    def _smooth_saturation(value: np.ndarray, limit: np.ndarray) -> np.ndarray:
        """
        平滑限幅（使用 tanh 函数）
        
        公式：
            output = limit · tanh(value / limit)
        
        对比硬限幅：
            硬限幅：if |value| > limit: return sign(value)·limit; else: return value
            问题：在限幅点处导数不连续
        
            软限幅（tanh）：使用双曲正切函数
            优势：在全范围内可导，更适合闭环控制
        
        tanh的性质：
            - tanh(0) = 0
            - tanh(∞) = 1
            - tanh'(x) = 1 - tanh²(x) > 0 对所有x
            - 在x=0处斜率最大 = 1
        
        所以：
            output = limit · tanh(value / limit)
            - 当 value << limit 时：output ≈ value（线性范围）
            - 当 value ≈ limit 时：平滑过渡
            - 当 value >> limit 时：output → limit（饱和）
        
        参数：
            value: 输入向量
            limit: 各轴的限制值
        
        返回：
            限幅后的向量
        """
        return limit * np.tanh(value / limit)
    
    def compute_velocity(
        self,
        current_pose: np.ndarray,
        current_wrench: np.ndarray,
        target_wrench: np.ndarray = None,
    ) -> np.ndarray:
        """
        计算末端执行器的期望速度
        
        这是导纳控制的核心函数
        
        输入参数：
            current_pose: (6,) 当前位姿 [x, y, z, rx, ry, rz]
            current_wrench: (6,) 当前力/力矩 [fx, fy, fz, τx, τy, τz]
            target_wrench: (6,) 目标力（默认为零，即自由运动）
        
        输出：
            vel: (6,) 期望速度 [vx, vy, vz, ωx, ωy, ωz]
        
        ===== 算法步骤 =====
        """
        
        current_pose = np.array(current_pose, dtype=np.float64)
        current_wrench = np.array(current_wrench, dtype=np.float64)
        
        if target_wrench is None:
            target_wrench = np.zeros(6, dtype=np.float64)
        else:
            target_wrench = np.array(target_wrench, dtype=np.float64)
        
        # ========== Step 1: 低通滤波 ==========
        # 
        # 公式：
        #     F_filtered_k = α·F_raw_k + (1-α)·F_filtered_{k-1}
        #
        # 作用：平滑传感器噪声
        # α越小越平滑，但延迟越大（通常0.1-0.5）
        #
        
        self.filtered_wrench = (
            self.p.filter_alpha * current_wrench +
            (1.0 - self.p.filter_alpha) * self.filtered_wrench
        )
        """
        一阶低通滤波器
        时间常数：τ = -dt / ln(1-α)
        """
        
        # ========== Step 2: 计算力误差 ==========
        # 
        # 公式：
        #     F_error = F_target - F_measured      ... (6)
        #
        # 含义：当前力与目标力的偏差
        # 导纳的目标：通过调节速度使力误差趋于0
        #
        
        wrench_error = target_wrench - self.filtered_wrench
        """力误差向量"""
        
        # ========== Step 3: 计算位置误差 ==========
        # 
        # 公式：
        #     e = x_target - x_current
        #       = x_0 - x_current           （因为x_target通常固定在x_0）
        #
        # 含义：当前位置与参考位置的偏差
        # 刚度项用此来产生恢复力
        #
        
        pose_error = self.x0 - current_pose
        """位置误差 - 用于虚拟刚度反馈"""
        
        # ========== Step 4: 应用死区 ==========
        # 
        # 公式：
        #     F_eff_i = deadband_func(F_error_i, deadband_i)
        #
        # 作用：
        #     - 消除传感器偏置和低幅噪声
        #     - 避免在无力/无误差时产生不必要的运动
        #     - 提高系统稳定性
        #
        # 示例：
        #     如果 |F_error| < 1N (力死区)，则该方向忽略力反馈
        #     如果 |F_error| < 0.05N·m (力矩死区)，则该方向忽略力矩反馈
        #
        
        effective_force = np.zeros(6, dtype=np.float64)
        for i in range(6):
            effective_force[i] = self._soft_deadband(
                wrench_error[i],
                self.deadbands[i]
            )
        """有效力 - 经过死区处理的力误差"""
        
        # ========== Step 5: 核心导纳动力学 ==========
        # 
        # 这是整个算法的核心！
        #
        # 从离散化的导纳方程：
        #     v_k = A·v_{k-1} + B·(F_k - K·e_k)    ... (5)
        #
        # 其中三项的含义：
        #     A·v_{k-1}:         旧速度的衰减（阻尼作用）
        #     B·F_k:             力驱动的速度变化（导纳作用）
        #     -B·K·e_k:          位置反馈（刚度作用）
        #
        # 物理直观：
        #     1. 末端受到外力 → 根据导纳参数产生速度
        #     2. 速度使末端移动 → 产生位置误差
        #     3. 位置误差通过刚度反馈 → 产生恢复力
        #     4. 同时阻尼衰减速度 → 系统最终稳定
        #
        
        # 计算力驱动项：B·F
        force_driven_term = self.B * effective_force
        """
        公式：term_F = dt/(M+D·dt) · F_error
        含义：力有多快能驱动末端运动
        """
        
        # 计算刚度反馈项：K·e
        # 只对位置应用刚度（通常不对姿态应用）
        stiffness_feedback = self.gain_K * np.array([
            self.p.K * pose_error[0],  # 位置x：有刚度反馈
            self.p.K * pose_error[1],  # 位置y：有刚度反馈
            self.p.K * pose_error[2],  # 位置z：有刚度反馈
            0.0,                       # 姿态x：暂无刚度（可选加入）
            0.0,                       # 姿态y：暂无刚度
            0.0,                       # 姿态z：暂无刚度
        ])
        """
        公式：term_K = K·dt/(M+D·dt) · e
        含义：位置误差产生的恢复力（类似弹簧）
        """
        
        # 速度更新（导纳的核心递推公式）
        # 
        # 公式：
        #     v_k = A·v_{k-1} + B·F_k - B·K·e_k
        #
        # 分项说明：
        #     第1项：A·v_{k-1}         惯性 + 阻尼，旧速度逐步衰减
        #     第2项：+B·F_k            力驱动，外力加速末端
        #     第3项：-B·K·e_k          刚度反馈，位置误差产生恢复
        #
        
        self.vel = self.A * self.vel + force_driven_term - stiffness_feedback
        """
        更新速度：v_k = A·v_{k-1} + B·F_k - B·K·e_k
        这是离散导纳模型的递推形式
        """
        
        # ========== Step 6: 平滑速度限幅 ==========
        # 
        # 公式：
        #     v_limited_i = v_max_i · tanh(v_i / v_max_i)
        #
        # 作用：
        #     - 保证末端执行器速度不超过安全限制
        #     - 使用tanh而不是硬截断，保证可导性
        #     - 避免控制系统在限幅点处的非线性跳变
        #
        # 效果：
        #     - 当 v << v_max：速度几乎不变
        #     - 当 v ≈ v_max：平滑过渡
        #     - 当 v >> v_max：速度被限制在v_max
        #
        
        self.vel = self._smooth_saturation(self.vel, self.v_max)
        """
        平滑限幅：v = v_max · tanh(v / v_max)
        保证速度不超过安全限制
        """
        
        return self.vel.copy()
    
    def get_debug_info(self) -> Dict:
        """
        获取调试信息
        
        返回：
            dict 包含当前状态和参数信息
        """
        return {
            'velocity': self.vel.copy(),
            'filtered_wrench': self.filtered_wrench.copy(),
            'position_error': (self.x0 - np.zeros(6)).copy() if self.x0 is not None else None,
            'parameters': {
                'M': self.p.M,
                'D': self.p.D,
                'K': self.p.K,
                'A': self.A,           # 旧速度系数
                'B': self.B,           # 力系数
                'lambda': self.lambda_, # 阻尼比
            }
        }


# ============================================================
# 测试和演示
# ============================================================

def test_step_response():
    """
    测试1：阶跃力响应
    
    场景：初始无力，t=0.1s施加20N力
    观察：末端如何加速、加速度如何衰减
    """
    
    print("="*70)
    print("测试1：阶跃力响应（无位置误差）")
    print("="*70)
    
    # 参数设置
    params = AdmittanceParams(
        M=5.0,      # 5kg虚拟质量
        D=20.0,     # 阻尼
        K=100.0,    # 刚度
        dt=0.001,   # 1ms控制周期
    )
    
    controller = SingleArmAdmittanceController(params)
    initial_pose = np.array([0, 0, 1.0, 0, 0, 0])
    controller.reset(initial_pose)
    
    # 模拟参数
    t_total = 2.0  # 总时间2秒
    num_steps = int(t_total / params.dt)
    
    # 存储结果
    time_history = []
    force_history = []
    velocity_history = []
    position_history = []
    acceleration_history = []
    
    current_pose = initial_pose.copy()
    prev_vel = np.zeros(6)
    
    print(f"\n参数：M={params.M}kg, D={params.D}N·s/m, K={params.K}N/m")
    print(f"离散化系数：A={controller.A:.4f}, B={controller.B:.6f}")
    print(f"\n模拟时间：{t_total}s，控制周期：{params.dt*1000}ms")
    print(f"总步数：{num_steps}\n")
    print("时间(s) | 力(N)  | 速度(m/s) | 位置(m)  | 加速度(m/s²)")
    print("-" * 70)
    
    for step in range(num_steps):
        t = step * params.dt
        
        # 模拟力：阶跃输入
        if step > 100:  # t > 0.1s
            force = np.array([0, 0, 20, 0, 0, 0])  # 20N向上
        else:
            force = np.zeros(6)
        
        # 计算速度（这是导纳控制器的输出）
        vel = controller.compute_velocity(current_pose, force)
        
        # 计算加速度（速度的变化率）
        acceleration = (vel - prev_vel) / params.dt
        
        # 更新位置（数值积分）
        current_pose += vel * params.dt
        
        # 记录数据（每10步记录一次，减少数据量）
        if step % 10 == 0:
            time_history.append(t)
            force_history.append(force[2])
            velocity_history.append(vel[2])
            position_history.append(current_pose[2])
            acceleration_history.append(acceleration[2])
            
            # 定期打印
            if step % 100 == 0:
                print(f"{t:7.3f} | {force[2]:6.1f} | {vel[2]:9.4f} | {current_pose[2]:8.4f} | {acceleration[2]:12.4f}")
        
        prev_vel = vel
    
    # 转换为numpy数组
    time_history = np.array(time_history)
    force_history = np.array(force_history)
    velocity_history = np.array(velocity_history)
    position_history = np.array(position_history)
    acceleration_history = np.array(acceleration_history)
    
    return {
        'time': time_history,
        'force': force_history,
        'velocity': velocity_history,
        'position': position_history,
        'acceleration': acceleration_history,
        'title': '阶跃力响应'
    }


def test_position_disturbance():
    """
    测试2：位置干扰响应
    
    场景：固定末端位置，但给定位置干扰
    观察：刚度项如何产生恢复力
    """
    
    print("\n" + "="*70)
    print("测试2：位置干扰响应（无外力，有位置误差）")
    print("="*70)
    
    params = AdmittanceParams(
        M=5.0,
        D=20.0,
        K=100.0,
        dt=0.001,
    )
    
    controller = SingleArmAdmittanceController(params)
    initial_pose = np.array([0, 0, 1.0, 0, 0, 0])
    controller.reset(initial_pose)
    
    t_total = 2.0
    num_steps = int(t_total / params.dt)
    
    time_history = []
    position_error_history = []
    velocity_history = []
    
    current_pose = initial_pose.copy()
    
    print(f"\n刚度项系数：K_gain = {controller.gain_K:.6f}")
    print("\n时间(s) | 位置误差(m) | 速度(m/s)  | 力反馈(N)")
    print("-" * 70)
    
    for step in range(num_steps):
        t = step * params.dt
        
        # 无外力
        force = np.zeros(6)
        
        # 人为引入位置误差（模拟干扰）
        if 0.2 < t < 0.5:
            current_pose[2] = initial_pose[2] + 0.05 * np.sin(2*np.pi*2*(t-0.2))
        
        # 计算速度
        vel = controller.compute_velocity(current_pose, force)
        
        # 记录数据
        if step % 10 == 0:
            pose_error = initial_pose[2] - current_pose[2]
            time_history.append(t)
            position_error_history.append(pose_error)
            velocity_history.append(vel[2])
            
            if step % 100 == 0:
                # 刚度反馈力 = K · 位置误差
                stiffness_force = params.K * pose_error
                print(f"{t:7.3f} | {pose_error:11.5f} | {vel[2]:10.5f} | {stiffness_force:10.2f}")
    
    return {
        'time': np.array(time_history),
        'position_error': np.array(position_error_history),
        'velocity': np.array(velocity_history),
        'title': '位置干扰响应'
    }


def plot_results(result_list):
    """绘制测试结果"""
    
    fig, axes = plt.subplots(len(result_list), 3, figsize=(15, 4*len(result_list)))
    
    if len(result_list) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(result_list):
        time = result['time']
        title = result['title']
        
        # 子图1：力响应
        if 'force' in result:
            axes[idx, 0].plot(time, result['force'], 'b-', linewidth=2)
            axes[idx, 0].set_ylabel('力 (N)', fontsize=11, fontweight='bold')
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_title(f'{title} - 输入力', fontsize=12, fontweight='bold')
        
        # 子图2：速度响应
        axes[idx, 1].plot(time, result['velocity'], 'g-', linewidth=2)
        axes[idx, 1].set_ylabel('速度 (m/s)', fontsize=11, fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_title(f'{title} - 末端速度', fontsize=12, fontweight='bold')
        
        # 子图3：位置或加速度
        if 'position' in result:
            axes[idx, 2].plot(time, result['position'], 'r-', linewidth=2)
            axes[idx, 2].set_ylabel('位置 (m)', fontsize=11, fontweight='bold')
            axes[idx, 2].set_title(f'{title} - 末端位置', fontsize=12, fontweight='bold')
        elif 'acceleration' in result:
            axes[idx, 2].plot(time, result['acceleration'], 'orange', linewidth=2)
            axes[idx, 2].set_ylabel('加速度 (m/s²)', fontsize=11, fontweight='bold')
            axes[idx, 2].set_title(f'{title} - 末端加速度', fontsize=12, fontweight='bold')
        elif 'position_error' in result:
            axes[idx, 2].plot(time, result['position_error'], 'r-', linewidth=2)
            axes[idx, 2].set_ylabel('位置误差 (m)', fontsize=11, fontweight='bold')
            axes[idx, 2].set_title(f'{title} - 位置误差', fontsize=12, fontweight='bold')
        
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 0].set_xlabel('时间 (s)', fontsize=10)
        axes[idx, 1].set_xlabel('时间 (s)', fontsize=10)
        axes[idx, 2].set_xlabel('时间 (s)', fontsize=10)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("单臂导纳控制器 - 完整演示")
    print("="*70 + "\n")
    
    # 运行测试
    result1 = test_step_response()
    result2 = test_position_disturbance()
    
    # 绘制结果
    fig = plot_results([result1, result2])
    plt.savefig('admittance_control_results.png', dpi=150, bbox_inches='tight')
    print("\n\n图表已保存为 'admittance_control_results.png'")
    plt.show()
    
    # 打印总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
    导纳控制的核心思想：
    
    1. 力驱动速度，而不是直接驱动位置
       - 末端执行器对外力的响应是速度，而非位置
       - 这使得与环境的交互更加柔和自然
    
    2. 位置反馈通过刚度项实现
       - 位置误差产生恢复力（类似弹簧）
       - 刚度可调，适应不同环境
    
    3. 阻尼保证稳定性
       - 速度逐步衰减，避免无限加速
       - 临界阻尼：D = 2√(KM)，超调≈4%
    
    4. 参数物理意义清晰
       - M: 虚拟质量（惯性）
       - D: 虚拟阻尼（衰减）
       - K: 虚拟刚度（位置反馈强度）
    
    对比混合力/位置控制：
    - 导纳：力→速度（自然、一体化）
    - 混合：需要显式选择模式（复杂、易出错）
    """)