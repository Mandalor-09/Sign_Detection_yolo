from setuptools import setup, find_packages
import os

def install_required(path='./requirements.txt'):
    with open(path, 'r') as f:
        packages = f.readlines()
        packages = [i.replace('/n','') for i in packages if i != '-e .' and not i.startswith('#')]
        return packages

setup(
    name='object_detection_yolov5',
    version='1.0.0',
    author='Mandalor_09',
    author_email='oms42162@gmail.com',
    description="Hand Sign Detection Using YOLOv5",
    packages=find_packages(),
    install_requires=install_required(),  
)
