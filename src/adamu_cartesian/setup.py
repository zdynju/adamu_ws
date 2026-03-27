import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'adamu_cartesian'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- 核心：将沙盒环境的文件安装到 share 目录下 ---
        # 这样 ros2 launch 和 ros2 run 才能找到它们
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'urdf'),   glob(os.path.join('urdf', '*.urdf'))),
        (os.path.join('share', package_name, 'mujoco'), glob(os.path.join('mujoco', '*.xml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zdy',
    maintainer_email='2939669401@qq.com',
    description='Adamu 双臂笛卡尔柔顺控制沙盒测试包',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # --- 核心：注册自动化脚本入口 ---
            # 格式：'可执行文件名 = 包名.文件名:函数名'
            'test = adamu_cartesian.test:main',
        ],
    },
)