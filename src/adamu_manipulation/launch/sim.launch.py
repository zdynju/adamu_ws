import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    static_sm_node = Node(
        package="adamu_manipulation",
        executable="T1",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    
    delayed_sm_node = TimerAction(
        period=5.0,
        actions=[static_sm_node]
    )

    return LaunchDescription([
        delayed_sm_node
    ])