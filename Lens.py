# -*- coding: utf-8 -*-
# @Time : 2021/10/8 22:00
# @Author : Tengpeng Xu
# @File : Lens

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

xx = sp.symbols('x')


class ConvexLens(object):
    def __init__(self,
                 lens_type='spheric',
                 left_func=None,
                 right_func=None,
                 width=0.1,
                 height=1,
                 refractive=0.8,
                 x_range=(-10,
                          10)):
        self.width = width
        self.height = height
        self.refractive = refractive
        self.xrange = x_range
        self.paths = []
        if lens_type == 'spheric':
            R = (height**2 + width**2) / width / 2
            self.ul_side = sp.sqrt(R**2 - (xx - R + width)**2)
            self.dl_side = -sp.sqrt(R ** 2 - (xx - R + width) ** 2)
            self.ur_side = sp.sqrt(R**2 - (xx + R - width)**2)
            self.dr_side = -sp.sqrt(R ** 2 - (xx + R - width) ** 2)
        else:
            if left_func is None or right_func is None:
                raise ValueError
            hl = height / left_func.evalf(subs={xx: width})
            hr = height / right_func.evalf(subs={xx: width})
            self.ul_side = hl * left_func.subs(xx, xx + width)
            self.ur_side = hr * right_func.subs(xx, -xx + width)
            self.dl_side = -hl * left_func.subs(xx, xx + width)
            self.dr_side = -hr * right_func.subs(xx, -xx + width)

    def refraction(self, x, y, ki, vi, side):
        lens_side = getattr(self, side + '_side')
        kn = float(-1 / lens_side.diff(xx).evalf(subs={xx: x}))
        vn = int('r' in side) - int('l' in side)
        ni = vn * vi * (1 + kn * ki)
        refractive = self.refractive ** (int(ni < 0) - int(ni > 0))
        tan_alpha = (ki - kn) / (1 + kn * ki)
        sin_alpha = np.sqrt(tan_alpha ** 2 / (1 + tan_alpha ** 2))
        sin_beta = sin_alpha / refractive
        tan_beta = sin_beta / np.sqrt(1 - sin_beta ** 2)
        tan_beta *= tan_alpha / abs(tan_alpha)
        kj = (kn + tan_beta) / (1 - kn * tan_beta)
        vj = vi * (1 + kn * ki) / (1 + kn * kj)
        vj /= abs(vj)
        return kj, int(vj)

    def get_path(self, x0, y0, k0, v0, x_end=10):
        if np.isclose(y0, 0) and np.isclose(k0, 0):
            xs = np.array([x0, -self.width, self.width, x_end])
            ys = np.zeros(4)
            vs = np.ones(4) * v0
            self.paths.append([xs, ys])
            return
        path0 = k0 * (xx - x0) + y0
        try:
            res = np.array(sp.solve(path0 - self.ul_side, xx))
            x1 = res[(res <= 0) & (res >= -self.width)][0]
            side = 'ul'
        except BaseException:
            res = np.array(sp.solve(path0 - self.dl_side, xx))
            x1 = res[(res <= 0) & (res >= -self.width)][0]
            side = 'dl'
        y1 = path0.evalf(subs={xx: x1})
        k1, v1 = self.refraction(x1, y1, k0, v0, side)

        path1 = k1 * (xx - x1) + y1
        try:
            res = np.array(sp.solve(path1 - self.ur_side, xx))
            x2 = res[(res <= self.width) & (res >= 0)][0]
            side = 'ur'
        except BaseException:
            res = np.array(sp.solve(path1 - self.dr_side, xx))
            x2 = res[(res <= self.width) & (res >= 0)][0]
            side = 'dr'
        y2 = path1.evalf(subs={xx: x2})
        k2, v2 = self.refraction(x2, y2, k1, v1, side)

        path2 = k2 * (xx - x2) + y2
        y_end = path2.evalf(subs={xx: x_end})

        xs = np.array([x0, x1, x2, x_end])
        ys = np.array([y0, y1, y2, y_end])
        self.paths.append([xs, ys])

    def plot_lens(self):
        ul = sp.lambdify(xx, self.ul_side, 'numpy')
        ur = sp.lambdify(xx, self.ur_side, 'numpy')
        dl = sp.lambdify(xx, self.dl_side, 'numpy')
        dr = sp.lambdify(xx, self.dr_side, 'numpy')
        xl = np.linspace(-self.width, 0, 100)
        xr = np.linspace(0, self.width, 100)
        yul = ul(xl)
        ydl = dl(xl)
        yur = ur(xr)
        ydr = dr(xr)
        xs = np.append(xl, xr)
        yu = np.append(yul, yur)
        yd = np.append(ydl, ydr)
        plt.plot(xs, yu, 'k')
        plt.plot(xs, yd, 'k')

    def plot_axis(self):
        xs = np.linspace(self.xrange[0], self.xrange[1])
        ys = np.zeros_like(xs)
        plt.plot(xs, ys, 'k')

    def plot_path(self, axis=True, lens=True):
        if axis:
            self.plot_axis()
        if lens:
            self.plot_lens()
        for xs, ys in self.paths:
            plt.plot(xs, ys)

    def reset(self):
        self.paths = []
