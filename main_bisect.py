# -*- coding: utf-8 -*-
# @Time : 2021/10/7 18:40
# @Author : Tengpeng Xu
# @File : main

import numpy as np
from scipy.optimize import bisect
from scipy.misc import derivative as deriv
import matplotlib.pyplot as plt

class Lens(object):
    def __init__(self, left_func, right_func, width=0.1, height=1, refractive=0.8, x_range=(-10, 10)):
        self.left_func = left_func
        self.right_func = right_func
        self.width = width
        self.height = height
        self.refractive = refractive
        self.hl = self.height / left_func(self.width)
        self.hr = self.height / right_func(self.width)
        self.xrange = x_range

    def left_side(self, x):
        return self.hl * self.left_func(x + self.width)

    def right_side(self, x):
        return self.hr * self.right_func(- x + self.width)

    def refraction(self, side, x, y, k_in):
        if side == 'left':
            kn = -1 / deriv(self.left_side, x, dx=1e-4)
            refractive = self.refractive
            m = 1
        elif side == 'right':
            kn = -1 / deriv(self.right_side, x, dx=1e-4)
            refractive = 1 / self.refractive
            m = -1
        else:
            raise ValueError
        tan_alpha = (kn - k_in) / (1 + kn * k_in)
        sin_alpha = np.sqrt(tan_alpha ** 2 / (1 + tan_alpha ** 2))
        sin_beta = sin_alpha * refractive
        tan_beta = sin_beta / np.sqrt(1 - sin_beta ** 2)
        k_out = (kn + m * tan_beta) / (1 - m * kn * tan_beta)
        return k_out

    def get_path(self, x0, y0, k0, x_end=10):
        path0 = lambda x: k0 * (x - x0) + y0

        func1 = lambda x: path0(x) - self.left_side(x)
        x1 = bisect(func1, -self.width, 0)
        y1 = path0(x1)
        k1 = self.refraction('left', x1, y1, k0)
        path1 = lambda x: k1 * (x - x1) + y1

        func2 = lambda x: path1(x) - self.right_side(x)
        x2 = bisect(func2, 0, self.width)
        y2 = path1(x2)
        k2 = self.refraction('right', x2, y2, k1)
        path2 = lambda x: k2 * (x - x2) + y2
        y_end = path2(x_end)

        self.paths = [path0, path1, path2]
        self.xs = [x0, x1, x2, x_end]
        self.ys = [y0, y1, y2, y_end]

    def plot_lens(self):
        xl = np.linspace(-self.width, 0, 100)
        xr = np.linspace(0, self.width, 100)
        yl = self.left_side(xl)
        yr = self.right_side(xr)
        xs = np.append(xl, xr)
        ys = np.append(yl, yr)
        plt.plot(xs, ys, 'k')

    def plot_axis(self):
        xs = np.linspace(self.xrange[0], self.xrange[1])
        ys = np.zeros_like(xs)
        plt.plot(xs, ys, 'k')

    def plot_path(self):
        plt.plot(self.xs, self.ys)

def left_func(x):
    # 透镜左侧函数形式
    return np.sqrt(1-(1-x)**2)

def right_func(x):
    # 透镜右侧函数形式
    return np.sqrt(1-(1-x)**2)

w, h, n = 0.5, 1, 0.8 # 透镜半宽度，半高度，折射率

lens = Lens(left_func, right_func, width=w, height=h, refractive=n, x_range=(-10, 4)) # 设置透镜系统
lens.plot_axis() # 画出主光轴
lens.plot_lens() # 画出透镜

x0, y0, k0 = -10, 0, 0 # 光线起点坐标，斜率
# lens.get_path(x0, y0, k0, x_end=4)
# lens.plot_path()

y0s = np.linspace(0.1, 0.2, 10)
for y0 in y0s:
    lens.get_path(x0, y0, k0, x_end=4)
    lens.plot_path()

plt.grid()
plt.show()
