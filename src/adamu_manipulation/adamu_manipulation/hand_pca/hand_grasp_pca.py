"""
╔══════════════════════════════════════════════════════════════╗
║        Adam_U 仿生鲁棒协同矩阵生成 Pipeline (V4.0 改进)      ║
║                                                              ║
║  核心改进 (V4.0):                                             ║
║  [+] 从adam_u.xml解析真实关节限度                             ║
║  [+] 集成10种典型手部抓握姿势                                 ║
║  [+] 约束生成的轨迹在有效范围内                               ║
║  [+] 改进的马尔可夫转移矩阵（基于真实手部运动）              ║
║  [+] 关节协同约束验证与修正                                  ║
║  [+] 增强的可视化分析                                         ║
╚══════════════════════════════════════════════════════════════╝

关节索引约定 (12 DOF):
  [0]  R_thumb_MCP_joint1   拇指 MCP 轴1 (内收/外展)   范围: [0, 1.1]
  [1]  R_thumb_MCP_joint2   拇指 MCP 轴2 (屈曲/伸展)   范围: [0, 0.5]
  [2]  R_thumb_PIP_joint    拇指 PIP 屈曲             范围: [0, 1.0]
  [3]  R_thumb_DIP_joint    拇指 DIP 屈曲             范围: [0, 1.2]
  [4]  R_index_MCP_joint    食指 MCP 屈曲             范围: [0, 1.7]
  [5]  R_index_DIP_joint    食指 DIP 屈曲             范围: [0, 1.6]
  [6]  R_middle_MCP_joint   中指 MCP 屈曲             范围: [0, 1.7]
  [7]  R_middle_DIP_joint   中指 DIP 屈曲             范围: [0, 1.6]
  [8]  R_ring_MCP_joint     无名指 MCP 屈曲           范围: [0, 1.7]
  [9]  R_ring_DIP_joint     无名指 DIP 屈曲           范围: [0, 1.6]
  [10] R_pinky_MCP_joint    小指 MCP 屈曲             范围: [0, 1.7]
  [11] R_pinky_DIP_joint    小指 DIP 屈曲             范围: [0, 1.6]
"""

# ─────────────────────────────────────────────
#  依赖导入
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  全局配置
# ─────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
N_DOF = 12

# ═══════════════════════════════════════════════════════════════
#  [新增] 关节限度约束  ── 从 adam_u.xml 解析
# ═══════════════════════════════════════════════════════════════
JOINT_LIMITS = {
    "min": np.array([0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]),
    "max": np.array([1.1,   0.5,   1.0,   1.2,   1.7,   1.6,   1.7,   1.6,   1.7,   1.6,   1.7,   1.6]),
}

def clip_to_limits(angles: np.ndarray) -> np.ndarray:
    """将关节角度约束在有效范围内"""
    return np.clip(angles, JOINT_LIMITS["min"], JOINT_LIMITS["max"])


# ══════════════════════════════════════════════
#  [改进] 姿态字典  ── 集成10种典型手部抓握
# ══════════════════════════════════════════════

@dataclass
class HandPose:
    """单个手部姿态的数据容器"""
    name:        str
    label:       str
    angles:      np.ndarray
    description: str = ""
    category:    str = ""  # 新增：姿态分类

    def __post_init__(self):
        assert len(self.angles) == N_DOF, f"姿态 '{self.name}' 必须有 {N_DOF} 个关节角度"
        # 自动约束到有效范围
        self.angles = clip_to_limits(self.angles)


POSES: Dict[str, HandPose] = {
    # ─── 基础姿态 ───
    "open": HandPose(
        name="open", label="完全张开", category="基础",
        angles=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        description="全张开，所有关节伸直，最大开合状态"
    ),
    
    "relax": HandPose(
        name="relax", label="自然放松", category="基础",
        angles=np.array([0.0, 0.1, 0.13, 0.18, 0.255, 0.24, 0.255, 0.24, 0.255, 0.24, 0.255, 0.24]),
        description="人体双手真实静止默认态，拇指MCP1外展，各指轻微弯曲"
    ),
    
    "power": HandPose(
        name="power", label="力量抓握", category="抓握",
        angles=np.array([1.1, 0.0, 1.0, 1.2, 1.7, 1.49, 1.7, 1.27, 1.7, 1.42, 1.7, 1.6]),
        description="五指全力屈曲，适合抓取大型或重型物体"
    ),
    
    # ─── 精准抓握姿态 ───
    "two_finger_pinch": HandPose(
        name="two_finger_pinch", label="两指捏", category="精准",
        angles=np.array([1.1, 0.5, 0.0, 0.0, 0.561, 0.696, 1.6, 1.23, 1.65, 1.46, 1.7, 1.6]),
        description="拇指对食指捏取，其他指伸展"
    ),
    
    "tripod": HandPose(
        name="tripod", label="三指抓", category="精准",
        angles=np.array([1.1, 0.4, 0.0, 0.0, 0.7395, 0.112, 0.782, 0.088, 1.65, 1.46, 1.7, 1.6]),
        description="拇指+食指+中指协同，其他指辅助"
    ),
    
    # ─── 钩握与侧压 ───
    "four_finger_hook": HandPose(
        name="four_finger_hook", label="四指钩", category="抓握",
        angles=np.array([0.0, 0.0, 0.0, 0.0, 1.088, 0.72, 1.122, 0.664, 1.071, 0.688, 0.9945, 0.672]),
        description="四指DIP深度弯曲如钩，拇指完全伸展"
    ),
    
    "lateral_press": HandPose(
        name="lateral_press", label="指尖侧压", category="精准",
        angles=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.72, 0.0, 0.72, 0.0, 0.712, 0.0, 0.712]),
        description="拇指侧面压住其他指，DIP深度弯曲"
    ),
    
    # ─── 五指握姿态 ───
    "five_finger_small": HandPose(
        name="five_finger_small", label="五指握小", category="抓握",
        angles=np.array([1.1, 0.29, 0.215, 0.144, 0.578, 0.864, 1.105, 0.328, 0.782, 0.848, 1.02, 0.944]),
        description="五指握取小型物体，灵活适应"
    ),
    
    "five_finger_large": HandPose(
        name="five_finger_large", label="五指握大", category="抓握",
        angles=np.array([1.1, 0.0239, 0.0572, 0.0764, 0.0606, 0.7489, 0.3698, 0.6522, 0.3031, 0.6923, 0.2982, 0.5249]),
        description="五指握取大型物体，掌心展开"
    ),
    
    # ─── 特殊姿态 ───
    "l_hook": HandPose(
        name="l_hook", label="L型钩底", category="特殊",
        angles=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.36, 0.0, 1.536, 0.0, 1.456, 0.0, 1.256]),
        description="食指伸展，中无小三指DIP钩曲，用于推拉"
    ),
}


# ══════════════════════════════════════════════════════════════════
#  [改进] 马尔可夫转移矩阵  ── 基于真实手部运动学
# ══════════════════════════════════════════════════════════════════

POSE_KEYS = [
    "open", "relax", "power",
    "two_finger_pinch", "tripod", "four_finger_hook", "lateral_press",
    "five_finger_small", "five_finger_large", "l_hook"
]

# 转移概率矩阵：基于自然手部运动学构建
# 核心原则：
#   - open 自然回 relax（人手放松）
#   - relax 是中心枢纽，可过渡到大多数姿态
#   - 相似姿态间高概率直转
#   - 极端姿态（power, l_hook）多回 relax 再转
TRANSITION_MATRIX = np.array([
    # open  relax power 2pin trip 4hook later 5s  5l  lhk
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # open   → 必回 relax
    [0.05, 0.00, 0.25, 0.20, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05],  # relax  → 枢纽分发
    [0.05, 0.60, 0.00, 0.10, 0.15, 0.05, 0.00, 0.03, 0.02, 0.00],  # power  → 多回 relax
    [0.00, 0.40, 0.05, 0.00, 0.35, 0.05, 0.05, 0.05, 0.03, 0.02],  # 2pin   → 常转 tripod
    [0.00, 0.35, 0.10, 0.30, 0.00, 0.10, 0.05, 0.05, 0.03, 0.02],  # trip   → 常转 2pin
    [0.25, 0.50, 0.05, 0.00, 0.05, 0.00, 0.00, 0.05, 0.05, 0.05],  # 4hook  → 多回 open/relax
    [0.00, 0.50, 0.05, 0.15, 0.10, 0.05, 0.00, 0.05, 0.05, 0.05],  # later  → 回 relax
    [0.00, 0.30, 0.10, 0.20, 0.20, 0.05, 0.05, 0.00, 0.05, 0.05],  # 5s     → 多转 tripod
    [0.00, 0.30, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.00, 0.10],  # 5l     → 多转 power/2pin
    [0.20, 0.40, 0.00, 0.00, 0.00, 0.20, 0.00, 0.05, 0.10, 0.00],  # lhk    → 多回 open/relax
], dtype=float)

# 归一化
TRANSITION_MATRIX /= TRANSITION_MATRIX.sum(axis=1, keepdims=True)

# 显示转移矩阵统计
print("\n✓ 马尔可夫转移矩阵已加载")
print(f"  姿态数: {len(POSE_KEYS)}")
print(f"  矩阵形状: {TRANSITION_MATRIX.shape}")


# ══════════════════════════════════════════════════════════════════
#  [改进] 轨迹生成  ── 约束在有效范围内
# ══════════════════════════════════════════════════════════════════

def generate_bionic_trajectory(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int,
    gaussian_std: float = 0.008,
    impulse_prob: float = 0.02,
    impulse_scale: float = 0.05,
) -> np.ndarray:
    """
    生成仿生运动轨迹，约束在关节限度内。
    
    改进: 自动裁剪超出范围的值，确保物理可行性
    """
    t = np.linspace(0, 1, num_frames)
    s = (1.0 - np.cos(t * np.pi)) / 2.0
    
    traj = np.outer(1.0 - s, start_pose) + np.outer(s, end_pose)
    
    # 高斯噪声 + 泊松尖峰噪声
    gaussian_noise = RNG.normal(0, gaussian_std, size=traj.shape)
    impulse_mask = RNG.random(size=traj.shape) < impulse_prob
    impulse_noise = impulse_mask * RNG.choice([-1, 1], size=traj.shape) * impulse_scale
    
    traj = traj + gaussian_noise + impulse_noise
    
    # 关键改进：强制约束在有效范围内
    traj = clip_to_limits(traj)
    
    return traj


def sample_markov_sequence(
    n_transitions: int,
    start_pose: str = "relax",
    frames_per_transition: int = 120,
) -> List[Tuple[str, str, int]]:
    """基于马尔可夫转移矩阵采样动作序列"""
    sequence = []
    current = start_pose

    for _ in range(n_transitions):
        idx = POSE_KEYS.index(current)
        probs = TRANSITION_MATRIX[idx]
        next_pose = RNG.choice(POSE_KEYS, p=probs)
        
        frames = max(30, int(frames_per_transition * (0.7 + 0.6 * RNG.random())))
        sequence.append((current, next_pose, frames))
        current = next_pose

    return sequence


# ══════════════════════════════════════════════════════════════════
#  [改进] 数据集构建
# ══════════════════════════════════════════════════════════════════

def build_dataset(n_transitions: int = 150):
    """构建仿生运动数据集"""
    print("\n" + "─"*60)
    print(f"构建数据集 (n_transitions={n_transitions})...")
    print("─"*60)
    
    dataset_raw = []
    labels = []
    
    # 采样动作序列
    sequence = sample_markov_sequence(n_transitions, start_pose="relax", frames_per_transition=120)
    
    for src_name, dst_name, frames in sequence:
        src_pose = POSES[src_name].angles
        dst_pose = POSES[dst_name].angles
        
        traj = generate_bionic_trajectory(src_pose, dst_pose, frames)
        dataset_raw.append(traj)
        
        # 标签为起点和终点的平均
        labels.extend([f"{src_name}→{dst_name}"] * frames)
    
    dataset_raw = np.vstack(dataset_raw)  # (N, 12)
    
    print(f"✓ 样本总数: {len(dataset_raw)}")
    print(f"✓ 特征维度: {dataset_raw.shape[1]}")
    
    # 检查关节限度遵守情况
    violations = np.sum((dataset_raw < JOINT_LIMITS["min"]) | (dataset_raw > JOINT_LIMITS["max"]))
    print(f"✓ 关节限度违反数: {violations} (应为 0)")
    
    # 标准化
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset_raw)
    
    return dataset_scaled, dataset_raw, labels, scaler


# ══════════════════════════════════════════════════════════════════
#  [不变] PCA 分析
# ══════════════════════════════════════════════════════════════════

def run_pca(dataset_scaled: np.ndarray, variance_threshold: float = 0.95) -> Tuple[PCA, int]:
    """
    执行 PCA 分析，动态选择主成分数量。
    """
    print("\n" + "─"*60)
    print("执行 PCA 分析...")
    print("─"*60)
    
    pca = PCA()
    pca.fit(dataset_scaled)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = np.argmax(cumvar >= variance_threshold) + 1
    
    print(f"✓ PCA 完成")
    print(f"  前 {n_keep} 个 PC 解释 {cumvar[n_keep-1]*100:.2f}% 方差")
    print(f"  PC 方差分布:")
    for i in range(min(5, N_DOF)):
        bar = "█" * int(pca.explained_variance_ratio_[i] * 500)
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:5.1f}% {bar}")
    
    return pca, n_keep


# ══════════════════════════════════════════════════════════════════
#  [改进] 重建误差验证  ── 按姿态分类统计
# ══════════════════════════════════════════════════════════════════

def validate_reconstruction(pca, dataset_scaled, dataset_raw, scaler, n_keep):
    """
    验证重建误差，并按姿态分类统计。
    """
    print("\n" + "─"*60)
    print(f"重建误差验证 (PC={n_keep})...")
    print("─"*60)
    
    # 全局重建误差
    scores = pca.transform(dataset_scaled)[:, :n_keep]
    recon_scaled = scores @ pca.components_[:n_keep] + pca.mean_
    recon = scaler.inverse_transform(recon_scaled)
    
    overall_rmse = float(np.sqrt(np.mean((dataset_raw - recon) ** 2)))
    max_err = float(np.max(np.abs(dataset_raw - recon)))
    
    print(f"✓ 全局 RMSE: {overall_rmse:.5f}")
    print(f"✓ 最大误差: {max_err:.5f}")
    
    # 各关键姿态的误差
    pose_errors = {}
    print(f"\n✓ 关键姿态重建误差:")
    
    # 按分类显示
    categories = {}
    for key, pose in POSES.items():
        if pose.category not in categories:
            categories[pose.category] = []
        categories[pose.category].append(key)
    
    for category in sorted(categories.keys()):
        print(f"\n  {category}:")
        for key in categories[category]:
            pose = POSES[key]
            angle = pose.angles.reshape(1, -1)
            angle_scaled = scaler.transform(angle)
            scores_p = pca.transform(angle_scaled)[:, :n_keep]
            recon_scaled = scores_p @ pca.components_[:n_keep] + pca.mean_
            recon = scaler.inverse_transform(recon_scaled)
            err = float(np.sqrt(np.mean((angle - recon) ** 2)))
            pose_errors[key] = err
            bar = "█" * int(err * 3000)
            print(f"    {pose.label:<10}: RMSE = {err:.5f}  {bar}")
    
    print("\n" + "─"*60)
    return {"overall_rmse": overall_rmse, "max_err": max_err, "pose_rmse": pose_errors}


# ══════════════════════════════════════════════════════════════════
#  [改进] 可视化分析
# ══════════════════════════════════════════════════════════════════

def plot_results(pca, dataset_scaled, dataset_raw, scaler, labels, n_keep, error_dict):
    """增强的可视化分析"""
    plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    scores = pca.transform(dataset_scaled)[:, :n_keep]
    recon_scaled = scores @ pca.components_[:n_keep] + pca.mean_
    recon = scaler.inverse_transform(recon_scaled)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Adam_U 仿生协同矩阵分析 V4.0 (关节限度约束版)", fontsize=18, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    variance = pca.explained_variance_ratio_
    cumvar = np.cumsum(variance)
    colors_bar = ["#E84545" if i < n_keep else "#AAAAAA" for i in range(N_DOF)]
    
    # ── 图1: Scree Plot ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, N_DOF + 1), variance * 100, color=colors_bar, edgecolor="white", width=0.7)
    ax1.plot(range(1, N_DOF + 1), cumvar * 100, "o-", color="#2B5B84", lw=2, ms=5, label="累计方差")
    ax1.axvline(n_keep + 0.5, color="#E84545", linestyle=":", lw=2)
    ax1.set_xlabel("主成分序号")
    ax1.set_ylabel("方差解释率 (%)")
    ax1.set_title("Scree Plot")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(range(1, N_DOF + 1))
    ax1.grid(axis='y', alpha=0.3)
    
    # ── 图2: PC1-PC2 散点 ──
    ax2 = fig.add_subplot(gs[0, 1])
    scores = pca.transform(dataset_scaled)
    
    # 按姿态分类着色
    color_map = {}
    colors = plt.cm.tab20(np.linspace(0, 1, len(POSE_KEYS)))
    for i, key in enumerate(POSE_KEYS):
        color_map[key] = colors[i]
    
    for key in POSE_KEYS:
        pose = POSES[key]
        ang_s = scaler.transform(pose.angles.reshape(1, -1))
        sc_p = pca.transform(ang_s)
        ax2.scatter(sc_p[0, 0], sc_p[0, 1], s=150, zorder=5, marker="*",
                   color=color_map[key], edgecolors="black", linewidths=0.8)
        ax2.annotate(pose.label, (sc_p[0, 0], sc_p[0, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", fontweight='bold')
    
    ax2.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)", fontsize=11)
    ax2.set_title("PC1-PC2 姿态空间")
    ax2.grid(alpha=0.3)
    
    # ── 图3: 关节限度可视化 ──
    ax3 = fig.add_subplot(gs[0, 2])
    joint_labels = [
        "拇\nMCP1", "拇\nMCP2", "拇\nPIP",  "拇\nDIP",
        "食\nMCP",  "食\nDIP",  "中\nMCP",  "中\nDIP",
        "环\nMCP",  "环\nDIP",  "小\nMCP",  "小\nDIP",
    ]
    x = np.arange(N_DOF)
    ax3.bar(x, JOINT_LIMITS["max"] - JOINT_LIMITS["min"], 
           bottom=JOINT_LIMITS["min"], color="#4CAF50", alpha=0.6, label="有效范围")
    ax3.scatter(x, np.array([POSES["open"].angles, POSES["power"].angles]).mean(axis=0),
               color='red', s=50, marker='o', zorder=5, label="平均位置")
    ax3.set_xticks(x)
    ax3.set_xticklabels(joint_labels, fontsize=7)
    ax3.set_ylabel("关节角度范围")
    ax3.set_title("关节限度约束")
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # ── 图4: 前3个PC的权重热图 ──
    ax4 = fig.add_subplot(gs[1, 0])
    pc_matrix = pca.components_[:min(3, n_keep)]
    im = ax4.imshow(pc_matrix, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)
    ax4.set_xticks(range(N_DOF))
    ax4.set_xticklabels(joint_labels, fontsize=6.5)
    ax4.set_yticks(range(pc_matrix.shape[0]))
    ax4.set_yticklabels([f"PC{i+1}" for i in range(pc_matrix.shape[0])], fontsize=9)
    ax4.set_title("协同向量热图")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # ── 图5: 姿态误差柱状图 ──
    ax5 = fig.add_subplot(gs[1, 1])
    pose_keys_sorted = sorted(error_dict["pose_rmse"].keys(), 
                             key=lambda k: error_dict["pose_rmse"][k], reverse=True)
    pose_errs = [error_dict["pose_rmse"][k] for k in pose_keys_sorted]
    pose_names = [POSES[k].label for k in pose_keys_sorted]
    bar_colors = plt.cm.RdYlGn_r(np.array(pose_errs) / (max(pose_errs) + 1e-6))
    
    ax5.barh(pose_names, pose_errs, color=bar_colors, edgecolor="white")
    ax5.axvline(error_dict["overall_rmse"], color="navy", linestyle="--", lw=2, 
               label=f"全局 RMSE={error_dict['overall_rmse']:.4f}")
    ax5.set_xlabel("RMSE（归一化角度）")
    ax5.set_title(f"关键姿态重建误差 (PC={n_keep})")
    ax5.legend(fontsize=8)
    ax5.grid(axis='x', alpha=0.3)
    
    # ── 图6: 姿态分布（按分类） ──
    ax6 = fig.add_subplot(gs[1, 2])
    categories = {}
    for key in POSE_KEYS:
        cat = POSES[key].category
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    cats = list(categories.keys())
    counts = list(categories.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    ax6.pie(counts, labels=cats, autopct='%1.1f%%', colors=colors, startangle=90)
    ax6.set_title(f"姿态分类分布 (总数={len(POSE_KEYS)})")
    
    # ── 图7: 单姿态重建对比（tripod） ──
    ax7 = fig.add_subplot(gs[2, 0])
    demo_pose = POSES["tripod"]
    demo_scaled = scaler.transform(demo_pose.angles.reshape(1, -1))
    demo_scores = pca.transform(demo_scaled)[:, :n_keep]
    demo_recon_scaled = demo_scores @ pca.components_[:n_keep] + pca.mean_
    demo_recon = scaler.inverse_transform(demo_recon_scaled).flatten()
    
    x = np.arange(N_DOF)
    ax7.bar(x - 0.2, demo_pose.angles, 0.35, label="原始", color="#2B5B84", alpha=0.85)
    ax7.bar(x + 0.2, demo_recon, 0.35, label="重建", color="#E84545", alpha=0.85)
    ax7.set_xticks(x)
    ax7.set_xticklabels(joint_labels, fontsize=6)
    ax7.set_title(f"'{demo_pose.label}' 重建对比")
    ax7.set_ylabel("关节角度")
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # ── 图8: PC时序演化 ──
    ax8 = fig.add_subplot(gs[2, 1])
    n_show = min(500, scores.shape[0])
    ax8.plot(scores[:n_show, 0], lw=1.5, color="#2B5B84", label="PC1", alpha=0.8)
    ax8.plot(scores[:n_show, 1], lw=1.5, color="#E84545", label="PC2", alpha=0.8)
    if n_keep >= 3:
        ax8.plot(scores[:n_show, 2], lw=1, color="#FFA500", label="PC3", alpha=0.6)
    ax8.set_xlabel("帧序号")
    ax8.set_ylabel("PC 得分")
    ax8.set_title(f"前 {n_show} 帧 PC 时序演化")
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    # ── 图9: 误差分布直方图 ──
    ax9 = fig.add_subplot(gs[2, 2])
    all_errors = (dataset_raw - recon) ** 2
    per_joint_rmse = np.sqrt(np.mean(all_errors, axis=0))
    
    ax9.bar(x, per_joint_rmse, color="#FF6B6B", alpha=0.7, edgecolor="white")
    ax9.axhline(np.mean(per_joint_rmse), color="navy", linestyle="--", lw=2, label="平均")
    ax9.set_xticks(x)
    ax9.set_xticklabels(joint_labels, fontsize=6)
    ax9.set_ylabel("关节 RMSE")
    ax9.set_title("各关节重建误差")
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.savefig("adam_synergy_v4_improved.png", dpi=150, bbox_inches="tight")
    print("\n✓ 分析图表已保存: adam_synergy_v4_improved.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  模型保存
# ══════════════════════════════════════════════════════════════════

def save_model(pca, scaler, n_keep, error_dict):
    """保存完整模型和元数据"""
    pose_angles = {k: v.angles for k, v in POSES.items()}
    
    # 添加关节限度到保存
    np.savez(
        "adam_synergy_v4_model.npz",
        # PCA 参数
        pca_mean           = pca.mean_,
        pca_components     = pca.components_[:n_keep],
        pca_variance_ratio = pca.explained_variance_ratio_[:n_keep],
        # Scaler 参数
        scaler_mean        = scaler.mean_,
        scaler_scale       = scaler.scale_,
        # 关节限度约束（新增）
        joint_limits_min   = JOINT_LIMITS["min"],
        joint_limits_max   = JOINT_LIMITS["max"],
        # 元数据
        n_keep             = np.array(n_keep),
        overall_rmse       = np.array(error_dict["overall_rmse"]),
        # 关键姿态快照
        **{f"pose_{k}": v for k, v in pose_angles.items()},
    )
    
    print(f"\n✓ V4 模型已保存: adam_synergy_v4_model.npz")
    print(f"  主成分数: {n_keep}  |  RMSE: {error_dict['overall_rmse']:.5f}")
    print(f"  包含 {len(POSES)} 个关键姿态")
    print(f"  包含关节限度约束")


# ══════════════════════════════════════════════════════════════════
#  主 Pipeline
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("🧠 Adam_U 仿生协同矩阵生成 Pipeline V4.0（改进版）")
    print("   ✓ 关节限度约束 | ✓ 10种典型姿势 | ✓ 增强可视化")
    print("="*70)
    
    # Step 1: 构建数据集（约束在有效范围内）
    dataset_scaled, dataset_raw, labels, scaler = build_dataset(n_transitions=150)
    
    # Step 2: PCA 分析
    pca, n_keep = run_pca(dataset_scaled, variance_threshold=0.95)
    
    # Step 3: 重建误差验证
    error_dict = validate_reconstruction(pca, dataset_scaled, dataset_raw, scaler, n_keep)
    # Step 5: 模型保存
    save_model(pca, scaler, n_keep, error_dict)
    # Step 4: 可视化
    plot_results(pca, dataset_scaled, dataset_raw, scaler, labels, n_keep, error_dict)
    

    
    print("\n" + "="*70)
    print("✅ Pipeline 完成！")
    print("="*70)