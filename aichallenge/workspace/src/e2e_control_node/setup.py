from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'e2e_control_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # package.xml をインストール
        (os.path.join('share', package_name), ['package.xml']),
        # launch/config フォルダを含める
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A package for managing bag files in ROS2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # 実行可能スクリプトを定義
            'e2e_control_node = e2e_control_node.e2e_control_node:main',
        ],
    },
)