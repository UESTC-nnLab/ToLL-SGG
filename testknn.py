import sys
import os

# 包含正确 knn.so 文件的目录
correct_dir = "/home/honsen/conda_envs/anaconda3/envs/pdiff/lib/python3.10/site-packages/knn_cuda/csrc/_ext/knn" # <--- 注意是目录！

# 插入到搜索路径的最前面
sys.path.insert(0, correct_dir)
# 如果需要确保 site-packages 也在前面
site_packages_dir = "/home/honsen/conda_envs/anaconda3/envs/pdiff/lib/python3.10/site-packages"
if site_packages_dir not in sys.path:
     sys.path.insert(0, site_packages_dir)
elif sys.path.index(site_packages_dir) > 0:
     sys.path.remove(site_packages_dir)
     sys.path.insert(0, site_packages_dir)


print("--- 修改后的 sys.path ---")
import pprint
pprint.pprint(sys.path)
print("------------------------")

try:
    # 现在尝试导入
    from knn_cuda import KNN
    print("导入成功！")
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")