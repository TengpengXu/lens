# -*- coding: utf-8 -*-
# @Time : 2021/10/7 18:40
# @Author : Tengpeng Xu
# @File : main

from Lens import ConvexLens
import numpy as np
import matplotlib.pyplot as plt


w, h, n = 0.5, 1, 1.2 # 透镜半宽度，半高度，折射率

lens = ConvexLens(width=w, height=h, refractive=n, x_range=(-10, 5)) # 设置透镜系统

# 光线起点坐标，斜率， 方向（往前为1， 往后为-1）
x0, k0, v0 = -10, 0, 1
y0s = np.linspace(-0.5, 0.5, 10)

for y0 in y0s:
    lens.get_path(x0, y0, k0, v0, x_end=5)

lens.plot_path()
plt.grid()
plt.show()

