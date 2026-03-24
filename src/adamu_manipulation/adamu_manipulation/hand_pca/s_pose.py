import mujoco
import mujoco.viewer
import numpy as np

print("="*60)
print("🤖 Adam_U 仿真原生抓取位姿生成器 (Sim-Native Pose Creator)")
print("="*60)

# 1. 加载模型
xml_path = '/home/zhoudaoyuan/adamu_ws/src/adamu_description/mujoco/scene_adam_u.xml' # 请确保路径正确
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 提取左手/右手的关节索引 (假设你的关节名包含 L_)
jnt_names = [model.joint(i).name for i in range(model.njnt) if 'L_' in model.joint(i).name]
qpos_adr = [model.joint(name).qposadr[0] for name in jnt_names]

print(f"✅ 成功锁定 {len(jnt_names)} 个手指关节。")
print("\n🎮 操作指南：")
print("  1. 稍后会弹出一个 MuJoCo 仿真窗口。")
print("  2. 在右侧控制面板 (Control) 中，展开 'Joints'。")
print("  3. 用鼠标拖动滑块，把 Adam_U 捏成你想要的抓取手型 (比如: 强力包络/捏取)。")
print("  4. 捏好之后，直接点击右上角的 'X' 关闭窗口。")
print("  5. 终端会自动打印出你刚刚捏好的那套 12 维完美参数！")

# 3. 启动交互式可视化
mujoco.viewer.launch(model, data)

# 4. 窗口关闭后，提取数据
final_qpos = np.array([data.qpos[adr] for adr in qpos_adr])

print("\n🎉 位姿捕捉成功！这是你亲手捏出的 Adam_U 原生基因：")
np.set_printoptions(precision=4, suppress=True)
print(f"np.array({repr(final_qpos)})")



# 全握np.array(array([1.1  , 0.275, 1.   , 0.162, 1.7  , 1.6  , 1.7  , 1.6  , 1.7  ,
#        1.6  , 1.7  , 1.6  ]))



# 全张 np.array(array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

#
#
# 两指捏 np.array(array([1.1  , 0.5  , 0.   , 0.   , 0.561, 0.696, 1.7  , 1.6  , 1.7  ,
#        1.6  , 1.7  , 1.6  ]))




# 三指抓 np.array(array([ 1.1   ,  0.4   , -0.    ,  0.    ,  0.7395,  0.112 ,  0.782 ,
#       0.088 ,  1.7   ,  1.6   ,  1.7   ,  1.6   ]))


# 四指钩 np.array(array([0.    , 0.    , 0.    , 0.    , 1.088 , 0.72  , 1.122 , 0.664 ,
#      1.071 , 0.688 , 0.9945, 0.672 ]))

# 放松 np.array(array([0.154 , 0.3075, 0.15  , 0.18  , 0.255 , 0.24  , 0.255 , 0.24  ,
#        0.255 , 0.24  , 0.255 , 0.24  ]))



# 五指握 小 np.array(array([1.1  , 0.29 , 0.215, 0.144, 0.578, 0.864, 1.105, 0.328, 0.782,
#        0.848, 1.02 , 0.944]))


# 五指握大 np.array(array([ 1.1009, -0.0239,  0.0572,  0.0764,  0.0606,  0.7489,  0.3698,
#         0.6522,  0.3031,  0.6923,  0.2982,  0.5249]))

# L 型钩底 np.array(array([0.   , 0.   , 0.   , 0.   , 0.   , 1.36 , 0.   , 1.536, 0.   ,
#       1.456, 0.   , 1.256]))


# 指尖侧压 np.array(array([0.   , 0.   , 0.   , 0.   , 0.   , 0.72 , 0.   , 0.72 , 0.   ,
#       0.712, 0.   , 0.712])) 
