import os
from setuptools import setup, find_packages
# from knn_cuda import __version__


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='KNN_CUDA',
    version="2",
    description='pytorch version knn support cuda.',
    author='Shuaipeng Li',
    author_email='sli@mail.bnu.edu.cn',
    packages=find_packages(),
    package_data={
        'knn_cuda': ["csrc/cuda/knn.cu", "csrc/cuda/knn.cpp"]
    },
    install_requires=required,
    # --- 在这里添加 ---
    python_requires='==3.10.19', # 例子：要求 Python 3.7 或更高版本
    # ------------------
)