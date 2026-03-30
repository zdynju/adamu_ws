import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

# --- 新增：导入 ParameterBuilder ---
from launch_param_builder import ParameterBuilder

def generate_launch_description():
    # 强制全局使用仿真时间 (解决之前 MoveIt 报时间错乱导致 Abort 的问题)
    use_sim_time = True

    # 1. 仅使用 Builder 提取参数配置，不使用它的黑盒启动函数
    moveit_config = (
        MoveItConfigsBuilder("adam_u", package_name="adamu_moveit_config")
        .robot_description(file_path="config/adam_u.urdf.xacro")
        .robot_description_semantic(file_path="config/adam_u.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )
    
    # 把配置转成字典，并手动注入仿真时间参数
    moveit_config_dict = moveit_config.to_dict()
    moveit_config_dict.update({"use_sim_time": use_sim_time})

    controller_config = PathJoinSubstitution(
        [FindPackageShare("adamu_moveit_config"), "config", "ros2_controllers.yaml"]   
    )
    static_cam_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_cam_tf_publisher',
        # 严格按照：x, y, z, qx, qy, qz, qw, 父坐标系, 子坐标系
        arguments=['0.55', '0', '2.0', '0', '0', '0.7071068', '0.7071068', 'world', 'overhead_cam']
    )

    # ========================================================================
    # 2. 显式定义所有核心节点 (告别黑盒)
    # ========================================================================
    
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
    )
    
    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="both",
        parameters=[
            moveit_config.robot_description,
            controller_config,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            # FTS broadcaster → fts_processor 订阅的话题
            ('/left_fts_broadcaster/wrench',  '/left_ft_sensor_wrench'),
            ('/right_fts_broadcaster/wrench', '/right_ft_sensor_wrench'),
            # cartesian_compliance_controller 内部硬编码订阅 /{controller_name}/ft_sensor_wrench
            # 所有控制器都跑在同一个 control_node 进程里，所以重映射必须加在这里
            ('/left_arm_cartesian_compliance_controller/ft_sensor_wrench',  '/left_ft_sensor_wrench'),
            ('/right_arm_cartesian_compliance_controller/ft_sensor_wrench', '/right_ft_sensor_wrench'),
        ]
    )
    # 显式定义的 MoveGroup 节点
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config_dict,
                    {"use_sim_time": use_sim_time}
        ], # 所有的规划参数都在这个字典里
        
    )

    right_servo_params = {
        "moveit_servo": ParameterBuilder("adamu_moveit_config")
        .yaml("config/right_arm_servo.yaml")
        .to_dict()
    }

    # 定义右臂 Servo 节点
    right_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        namespace="/right_arm", # 放入右臂命名空间，防止话题冲突
        output="screen",
        parameters=[
            right_servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time}, # 🚨 极其重要：强制 Servo 使用仿真时间！
        ],
    )
    left_servo_params = {
        "moveit_servo": ParameterBuilder("adamu_moveit_config")
        .yaml("config/left_arm_servo.yaml")
        .to_dict()
    }
    

    # 定义左臂 Servo 节点
    left_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        namespace="/left_arm", # 放入左臂命名空间，防止话题冲突
        output="screen",
        parameters=[
            left_servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time}, # 🚨 极其重要：强制 Servo 使用仿真时间！
        ],
    )

    # 显式定义的 RViz2 节点
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("adamu_moveit_config"), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time}
        ],
    )

    # ========================================================================
    # 3. 显式定义所有控制器 Spawners (清晰可见)
    # ========================================================================
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller", "--controller-manager", "/controller_manager"],
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller", "--controller-manager", "/controller_manager"],
    )
    
    left_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_hand_controller", "--controller-manager", "/controller_manager"],
    )
    
    right_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_hand_controller", "--controller-manager", "/controller_manager"],
    )

    # 包含传送带等其他部件
    waist_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["waist_controller", "--controller-manager", "/controller_manager"],
    )
    
    head_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["head_controller", "--controller-manager", "/controller_manager"],
    )

    left_fts_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_fts_broadcaster", "--controller-manager", "/controller_manager"],
    )
    right_fts_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_fts_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # ✅ 新增：左手 5 个指腹 FTS 广播器 Spawner
    L_thumb_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["L_thumb_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    L_index_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["L_index_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    L_middle_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["L_middle_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    L_ring_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["L_ring_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    L_pinky_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["L_pinky_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # ✅ 新增：右手 5 个指腹 FTS 广播器 Spawner
    R_thumb_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["R_thumb_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    R_index_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["R_index_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    R_middle_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["R_middle_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    R_ring_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["R_ring_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )
    R_pinky_pad_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["R_pinky_pad_broadcaster", "--controller-manager", "/controller_manager"],
    )

    right_compliance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_cartesian_compliance_controller", "--controller-manager", "/controller_manager", "--inactive"]
            # 初始设为 inactive，后续根据需要再激活
    )
    left_compliance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_cartesian_compliance_controller", "--controller-manager", "/controller_manager", "--inactive"],
         # 初始设为 inactive，后续根据需要再激活
    )
    # conveyor_velocity_controller_spawner = Node(
        # package="controller_manager",
        # executable="spawner",
        # arguments=["conveyor_velocity_controller", "--controller-manager", "/controller_manager"],
    # )

    # dual_arm_controller_spawner = Node(
        # package="controller_manager",
        # executable="spawner",
        # arguments=["dual_arm_controller", "--controller-manager", "/controller_manager"],
    # )
    # 控制器启动顺序：先启动状态广播，再启动所有关节控制
    delay_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[
                left_arm_controller_spawner,
                right_arm_controller_spawner,
                left_hand_controller_spawner,
                right_hand_controller_spawner,
                waist_controller_spawner,
                head_controller_spawner,
                left_fts_broadcaster_spawner,
                right_fts_broadcaster_spawner,
                # ✅ 将新定义的手指传感器 Spawner 加入启动队列
                L_thumb_pad_broadcaster_spawner,
                L_index_pad_broadcaster_spawner,
                L_middle_pad_broadcaster_spawner,
                L_ring_pad_broadcaster_spawner,
                L_pinky_pad_broadcaster_spawner,
                R_thumb_pad_broadcaster_spawner,
                R_index_pad_broadcaster_spawner,
                R_middle_pad_broadcaster_spawner,
                R_ring_pad_broadcaster_spawner,
                R_pinky_pad_broadcaster_spawner,
                right_compliance_controller_spawner,
                left_compliance_controller_spawner,
                # conveyor_velocity_controller_spawner,
                # dual_arm_controller_spawner
            ],
        )
    )
    box_tf_node = Node(
        package='adamu_manipulation',  # 替换为你的真实包名
        executable='box_state',
        name='box_tf_broadcaster',
        output='screen'
    )
    # yolo_vision_node = Node(
    #     package="adamu_manipulation",
    #     executable="yolo_vision_node",
    #     name="yolo_vision_node",
    #     output="screen",
    #     parameters=[{"use_sim_time": use_sim_time}],
    # )

    convenyor_node = Node(
        package="adamu_manipulation",
        executable="add_conveyor",
        name="add_conveyor_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        # 全局强制时间同步魔法
        SetParameter(name='use_sim_time', value=True),
        static_cam_tf,
        node_robot_state_publisher,
        control_node,
        move_group_node,
        right_servo_node,
        left_servo_node,
        rviz_node,
        # yolo_vision_node,
        box_tf_node,
        joint_state_broadcaster_spawner,
        delay_controllers,
        convenyor_node
    ])