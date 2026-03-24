import mujoco
import mujoco.viewer
import numpy as np
import time

print("=" * 70)
print("🔬 Adam_U 特征向量 (PCA Components) 独立效果剖析 V4.0 (改进版)")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════
#  1. 加载 MuJoCo 模型
# ════════════════════════════════════════════════════════════════════
xml_path = '/home/zhoudaoyuan/adamu_ws/src/adamu_description/mujoco/adam_u.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ════════════════════════════════════════════════════════════════════
#  2. 加载 V4 协同模型（包含关节约束）
# ════════════════════════════════════════════════════════════════════
try:
    npz = np.load('adam_synergy_v4_model.npz')
    print("✅ 加载 V4.0 模型: adam_synergy_v4_model.npz")
except FileNotFoundError:
    print("⚠️  V4.0 模型未找到，尝试加载 V3.0 模型...")
    npz = np.load('adam_synergy_model_v3.npz')
    print("✅ 加载 V3.0 模型: adam_synergy_model_v3.npz")

PCA_MEAN       = npz['pca_mean']        # (12,)
PCA_COMPONENTS = npz['pca_components']  # (n_pc, 12)
SCALER_MEAN    = npz['scaler_mean']     # (12,)
SCALER_SCALE   = npz['scaler_scale']    # (12,)
N_PC           = PCA_COMPONENTS.shape[0]

# ✓ 新增：从V4.0模型加载关节限度
try:
    JOINT_LIMITS_MIN = npz['joint_limits_min']
    JOINT_LIMITS_MAX = npz['joint_limits_max']
    print(f"✅ 加载关节约束: min={JOINT_LIMITS_MIN}, max={JOINT_LIMITS_MAX}")
except KeyError:
    # 如果是V3.0模型，使用默认值
    JOINT_LIMITS_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    JOINT_LIMITS_MAX = np.array([1.1, 0.5, 1.0, 1.2, 1.7, 1.6, 1.7, 1.6, 1.7, 1.6, 1.7, 1.6])
    print(f"⚠️  使用默认关节约束 (V3.0兼容模式)")

# ════════════════════════════════════════════════════════════════════
#  3. 关节顺序 & 10种典型手部姿态
# ════════════════════════════════════════════════════════════════════
#
# [0]  R_thumb_MCP_joint1   拇指 MCP 轴1 (内收/外展)   [0.0, 1.1]
# [1]  R_thumb_MCP_joint2   拇指 MCP 轴2 (屈曲/伸展)   [0.0, 0.5]  ← 最受限！
# [2]  R_thumb_PIP_joint    拇指 PIP                 [0.0, 1.0]
# [3]  R_thumb_DIP_joint    拇指 DIP                 [0.0, 1.2]
# [4]  R_index_MCP_joint    食指 MCP                 [0.0, 1.7]
# [5]  R_index_DIP_joint    食指 DIP                 [0.0, 1.6]
# [6]  R_middle_MCP_joint   中指 MCP                 [0.0, 1.7]
# [7]  R_middle_DIP_joint   中指 DIP                 [0.0, 1.6]
# [8]  R_ring_MCP_joint     无名指 MCP               [0.0, 1.7]
# [9]  R_ring_DIP_joint     无名指 DIP               [0.0, 1.6]
# [10] R_pinky_MCP_joint    小指 MCP                 [0.0, 1.7]
# [11] R_pinky_DIP_joint    小指 DIP                 [0.0, 1.6]

POSES_RAW = {
    # ─── 基础姿态 ───
    "open":  np.array([0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.    ]),
    "relax": np.array([0.154,  0.3075, 0.15,   0.18,   0.255,  0.24,   0.255,  0.24,   0.255,  0.24,   0.255,  0.24  ]),
    
    # ─── 抓握姿态 ───
    "power":             np.array([1.1,    0.275,  1.0,    0.162,  1.7,    1.6,    1.7,    1.6,    1.7,    1.6,    1.7,    1.6   ]),
    "four_finger_hook":  np.array([0.,     0.,     0.,     0.,     1.088,  0.72,   1.122,  0.664,  1.071,  0.688,  0.9945, 0.672 ]),
    "five_finger_small": np.array([1.1,    0.29,   0.215,  0.144,  0.578,  0.864,  1.105,  0.328,  0.782,  0.848,  1.02,   0.944 ]),
    "five_finger_large": np.array([1.1,    0.0239, 0.0572, 0.0764, 0.0606, 0.7489, 0.3698, 0.6522, 0.3031, 0.6923, 0.2982, 0.5249]),
    
    # ─── 精准姿态 ───
    "two_finger_pinch":  np.array([1.1,    0.5,    0.,     0.,     0.561,  0.696,  1.7,    1.6,    1.7,    1.6,    1.7,    1.6   ]),
    "tripod":            np.array([1.1,    0.4,    0.,     0.,     0.7395, 0.112,  0.782,  0.088,  1.7,    1.6,    1.7,    1.6   ]),
    "lateral_press":     np.array([0.,     0.,     0.,     0.,     0.,     0.72,   0.,     0.72,   0.,     0.712,  0.,     0.712 ]),
    
    # ─── 特殊姿态 ───
    "l_hook":            np.array([0.,     0.,     0.,     0.,     0.,     1.36,   0.,     1.536,  0.,     1.456,  0.,     1.256 ]),
}

# ✓ 自动约束所有姿态在有效范围内
for key in POSES_RAW:
    POSES_RAW[key] = np.clip(POSES_RAW[key], JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

print(f"\n📊 加载 {len(POSES_RAW)} 种典型姿态:")
for key in POSES_RAW.keys():
    print(f"    • {key:20}")

# ════════════════════════════════════════════════════════════════════
#  4. Encode / Decode 函数
# ════════════════════════════════════════════════════════════════════

def encode(joints: np.ndarray) -> np.ndarray:
    """12维关节角度 → PC得分"""
    scaled = (joints - SCALER_MEAN) / SCALER_SCALE
    return (scaled - PCA_MEAN) @ PCA_COMPONENTS.T

def decode(scores: np.ndarray) -> np.ndarray:
    """PC得分 → 12维关节角度（完整反变换 + 约束）"""
    scaled = scores @ PCA_COMPONENTS + PCA_MEAN
    angles = scaled * SCALER_SCALE + SCALER_MEAN
    # ✓ 强制约束在有效范围
    return np.clip(angles, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

# ════════════════════════════════════════════════════════════════════
#  5. 计算各PC轴在姿态空间的自然范围
# ════════════════════════════════════════════════════════════════════
all_scores = np.array([encode(v) for v in POSES_RAW.values()])
PC_MIN = all_scores.min(axis=0)
PC_MAX = all_scores.max(axis=0)
PC_MID = (PC_MIN + PC_MAX) / 2.0

print(f"\n📐 PCA 空间分析 (总PC数={N_PC}):")

# PC轴物理含义（基于热图分析）
PC_DESCRIPTIONS = [
    {
        "title": "四指整体屈曲轴",
        "hint": "观察四指（食中环小）是否同步张开/握紧",
        "key_joints": "[4-11] 食中环小 MCP/DIP",
    },
    {
        "title": "拇指独立性轴",
        "hint": "观察拇指MCP1内收+MCP2屈曲 vs 其他指伸展（精准捏取模式）",
        "key_joints": "[0-3] 拇指全部 vs [4-11] 四指",
    },
    {
        "title": "拇食指内部重构轴",
        "hint": "观察拇指根部与末端的反向运动，食指与中环小指的对抗",
        "key_joints": "[0,3] vs [1,2] (拇指MCP1/DIP vs MCP2/PIP)",
    },
    {
        "title": "指尖精细控制轴",
        "hint": "低频补偿分量，调整各指DIP（指尖屈曲度）的精细差异",
        "key_joints": "[2,5,7,9,11] 各指 PIP/DIP",
    },
]

# 动态扩展
while len(PC_DESCRIPTIONS) < N_PC:
    i = len(PC_DESCRIPTIONS)
    PC_DESCRIPTIONS.append({
        "title": f"PC{i+1} 高阶协同",
        "hint": "观察此分量对整体姿态的影响（需要实时反馈命名）",
        "key_joints": "全部关节",
    })

print(f"{'PC':4} {'范围':20} {'幅度':8} {'物理含义':30}")
print("─" * 75)
for i in range(N_PC):
    amplitude = PC_MAX[i] - PC_MIN[i]
    print(f"PC{i+1:2} [{PC_MIN[i]:+6.2f}, {PC_MAX[i]:+6.2f}] "
          f"{amplitude:7.2f}  {PC_DESCRIPTIONS[i]['title']}")

# ════════════════════════════════════════════════════════════════════
#  6. 关节地址映射
# ════════════════════════════════════════════════════════════════════

# 右手关节顺序（严格对齐）
jnt_order = [
    'R_thumb_MCP_joint1', 'R_thumb_MCP_joint2', 'R_thumb_PIP_joint', 'R_thumb_DIP_joint',
    'R_index_MCP_joint', 'R_index_DIP_joint',
    'R_middle_MCP_joint', 'R_middle_DIP_joint',
    'R_ring_MCP_joint', 'R_ring_DIP_joint',
    'R_pinky_MCP_joint', 'R_pinky_DIP_joint',
]

qpos_adr = []
for name in jnt_order:
    found = False
    for i in range(model.njnt):
        if model.joint(i).name == name:
            qpos_adr.append(model.joint(i).qposadr[0])
            found = True
            break
    if not found:
        print(f"⚠️  关节 {name} 未找到")

print(f"\n🦾 右手关节映射: {len(qpos_adr)} 个关节")
if len(qpos_adr) == 12:
    print(f"    ✅ 完整的12DOF手部")
else:
    print(f"    ⚠️  只找到{len(qpos_adr)}个关节，预期12个")

# ════════════════════════════════════════════════════════════════════
#  7. PC轴独立激活函数
# ════════════════════════════════════════════════════════════════════

def apply_pc_pose(pc_index: int, t_normalized: float):
    """
    只激活指定PC轴，在其自然范围内做正弦摆动。
    其他PC轴保持在各自中心值，避免出现奇怪的中间姿态。

    Args:
        pc_index:     要激活的PC索引 (0 ~ N_PC-1)
        t_normalized: 归一化时间 [0, 1]，驱动正弦波一个完整周期
    """
    amplitude = (PC_MAX[pc_index] - PC_MIN[pc_index]) / 2.0
    center    = PC_MID[pc_index]
    
    # 正弦摆动：从最小值 → 最大值 → 最小值
    signal    = center + amplitude * np.sin(t_normalized * 2 * np.pi)

    scores = PC_MID.copy()       # 其他轴保持中心
    scores[pc_index] = signal    # 只改当前轴

    target_12d = decode(scores)

    # 设置关节位置（只改右手）
    for i, adr in enumerate(qpos_adr):
        if i < len(target_12d) and adr < len(data.qpos):
            data.qpos[adr] = target_12d[i]

# ════════════════════════════════════════════════════════════════════
#  8. 展示阶段信息
# ════════════════════════════════════════════════════════════════════

STAGE_INFO = []
for i in range(N_PC):
    desc = PC_DESCRIPTIONS[i]
    STAGE_INFO.append({
        "title": f"PC{i+1} — {desc['title']}",
        "hint": desc['hint'],
        "key_joints": desc['key_joints'],
        "range": f"[{PC_MIN[i]:+.2f}, {PC_MAX[i]:+.2f}]",
        "amplitude": PC_MAX[i] - PC_MIN[i],
    })

STAGE_DURATION = 8.0  # 每个PC轴展示8秒
TOTAL_DURATION = STAGE_DURATION * N_PC

# ════════════════════════════════════════════════════════════════════
#  9. 参考姿态评估
# ════════════════════════════════════════════════════════════════════
print(f"\n🎯 关键姿态在PC空间的位置:")
print(f"{'姿态':20} {'PC1':10} {'PC2':10} {'PC3':10}")
print("─" * 50)
for name, angles in POSES_RAW.items():
    scores = encode(angles)
    print(f"{name:20} {scores[0]:+.3f}    {scores[1]:+.3f}    {scores[2]:+.3f}")

# ════════════════════════════════════════════════════════════════════
#  10. MuJoCo 可视化主循环
# ════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"🚀 启动仿真展示")
print(f"   • 每个PC轴展示 {STAGE_DURATION:.0f} 秒")
print(f"   • 总共 {N_PC} 个PC轴，总时长 {TOTAL_DURATION:.0f} 秒")
print(f"   • 按 Esc 或关闭窗口退出")
print(f"{'='*70}\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time    = time.time()
    current_stage = -1

    while viewer.is_running():
        elapsed    = time.time() - start_time
        stage      = int((elapsed % TOTAL_DURATION) / STAGE_DURATION)
        t_in_stage = (elapsed % STAGE_DURATION) / STAGE_DURATION

        # 阶段切换提示
        if stage != current_stage:
            current_stage = stage
            info = STAGE_INFO[stage]
            
            print(f"{'─'*70}")
            print(f"👉  {info['title']}")
            print(f"    📍 范围: {info['range']},  幅度: {info['amplitude']:.2f}")
            print(f"    💡 提示: {info['hint']}")
            print(f"    🔑 关键关节: {info['key_joints']}")

        # 激活当前PC轴
        apply_pc_pose(stage, t_in_stage)

        # 物理仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print(f"\n{'='*70}")
print("✅ 展示结束")
print(f"{'='*70}")