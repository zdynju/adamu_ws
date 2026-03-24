from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'adamu_manipulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhoudaoyuan',
    maintainer_email='2939669401@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'static_pick_place = adamu_manipulation.static_pick_place:main',
            'yolo_vision_node = adamu_manipulation.yolo_vision_node:main',
            'servo_controller = adamu_manipulation.servo_controller:main',
            'test_servo = adamu_manipulation.test:main',
            'add_conveyor = adamu_manipulation.add_conveyor:main',
            'T1 = adamu_manipulation.T1:main',
            'fts_processor = adamu_manipulation.fts_processor:main',
            'fts = adamu_manipulation.fts:main',
            'hand = adamu_manipulation.hand_controller:main',
            'hand_test = adamu_manipulation.hand_test:main',
            'simple_hand = adamu_manipulation.simple_hand_controller:main',
            'T2 = adamu_manipulation.T2:main'
        ],
    },
)
