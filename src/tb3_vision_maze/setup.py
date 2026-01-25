import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_vision_maze'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moaaz',
    maintainer_email='moaazalasadi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'test_node = tb3_vision_maze.test_node:main',
        'motion_node = tb3_vision_maze.motion_node:main',
        'planner_node = tb3_vision_maze.planner_node:main',
        'vision_node = tb3_vision_maze.vision_node:main',
        'train_node = tb3_vision_maze.train_node:main',
    ],
    },
)
