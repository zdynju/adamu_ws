#!/bin/bash

# ADAMU 系统启动脚本
# 在多个终端标签页中自动启动所有必需的节点

WORKSPACE_DIR="$HOME/adamu_ws"

# 检查工作空间是否存在
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "❌ 错误：工作空间目录 $WORKSPACE_DIR 不存在！"
    exit 1
fi

echo "🚀 正在启动 ADAMU 操作系统..."

# 使用 gnome-terminal 打开多个标签页
gnome-terminal \
    --tab --title="2️⃣ T1_M 节点" -- bash -c "
        cd $WORKSPACE_DIR && \
        source install/setup.bash && \
        echo '🤖 正在启动 T1_M 节点...' && \
        ros2 run adamu_manipulation T1_M; \
        exec bash
    " \
    --tab --title="3️⃣ 控制器切换" -- bash -c "
        cd $WORKSPACE_DIR && \
        echo '⏳ 等待 T1_M 启动 (15秒)...' && \
        sleep 15 && \
        source install/setup.bash && \
        echo '🔄 正在启动控制器切换器...' && \
        echo '👉 请选择选项 1' && \
        ros2 run adamu_manipulation switcher; \
        exec bash
    " \
    --tab --title="4️⃣ 测试脚本" -- bash -c "
        cd $WORKSPACE_DIR && \
        echo '⏳ 等待控制器切换完成 (20秒)...' && \
        sleep 20 && \
        source install/setup.bash && \
        echo '🧪 正在运行测试脚本...' && \
        ros2 run adamu_manipulation test_static.py; \
        exec bash
    "

echo "✅ 所有终端已启动！"
echo "📌 注意：在第3个终端中记得选择选项 1"