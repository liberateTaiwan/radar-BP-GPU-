# 在 Python 中运行以下命令来查找路径
import sys
print(sys.version)  # 查看 Python 版本
print(sys.prefix)   # 查看 Python 安装路径

import numpy
print(numpy.get_include())  # Numpy 头文件路径