import os
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 定义包名
    pkg_description = FindPackageShare("adamu_moveit_config")
    pkg_cartesian = FindPackageShare("adamu_cartesian")

    # 1. 文件路径：URDF 在 description 包，配置在 cartesian 包
    urdf_file = PathJoinSubstitution([pkg_description, "config" ,"adam_u.urdf.xacro"])
    controllers_file = PathJoinSubstitution([pkg_cartesian, "config", "ros2_controllers.yaml"])

    # 解析 URDF
    robot_description = {"robot_description": Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ", urdf_file
    ])}

    # 2. 核心基础节点
    rsp_node = Node(
        package="robot_state_publisher", 
        executable="robot_state_publisher", 
        parameters=[robot_description, {"use_sim_time": True}]
    )
    
    mujoco_node = Node(
        package="mujoco_ros2_control", 
        executable="ros2_control_node", 
        parameters=[robot_description, controllers_file, {"use_sim_time": True}]
    )
    
    rviz_node = Node(
        package="rviz2", 
        executable="rviz2", 
        name="rviz2"
    )

    # 3. Spawner 定义
    def spawner(name, active=True):
        cmd = [name, "--controller-manager", "/controller_manager"]
        if not active: cmd.append("--inactive")
        return Node(package="controller_manager", executable="spawner", arguments=cmd)

    jsb = spawner("joint_state_broadcaster")
    left_jtc = spawner("left_arm_controller")
    right_jtc = spawner("right_arm_controller")
    
    # 柔顺控制器设为 inactive
    left_comp = spawner("left_arm_cartesian_compliance_controller", active=False)
    left_handl = spawner("left_motion_control_handle", active=False)

    # 5. 事件顺序：JSB 退出后（即加载完成后）启动其他控制器
    delay_spawners = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb,
            on_exit=[left_jtc, right_jtc, left_comp, left_handl]
        )
    )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        rsp_node, 
        mujoco_node, 
        rviz_node, 
        jsb, 
        delay_spawners
    ])