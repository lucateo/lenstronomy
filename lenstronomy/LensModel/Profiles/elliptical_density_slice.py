__author__ = "lynevdv"

import numpy as np
import cmath as c
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['ElliSLICE']

class ElliSLICE (LensProfileBase):
    """
    This class computes the lensing quantities for an elliptical slice of constant density.
    Based on Schramm 1994 https://ui.adsabs.harvard.edu/abs/1994A%26A...284...44S/abstract

    Computes the lensing quantities of an elliptical slice with semi major axis 'a' and
    semi minor axis 'b', centered on 'center_x' and 'center_y', oriented with an angle 'psi'
    in radian, and with constant surface mass density 'sigma_0'

    """
    param_names = ['a','b','psi','sigma_0','center_x','center_y']
    lower_limit_default = {'a': 0., 'b': 0., 'psi': -90./180.*np.pi,'center_x':-100.,'center_y':-100.}
    upper_limit_default = {'a': 100., 'b': 100., 'psi': 90. / 180. * np.pi, 'center_x': 100., 'center_y': 100.}

    def function(self,x, y, a, b, psi, sigma_0, center_x=0., center_y=0.):
        """
        lensing potential

        :param a: float, semi-major axis, must be positive
        :param b: float, semi-minor axis, must be positive
        :param psi: float, orientation in radian
        :param sigma_0: float, surface mass density, must be positive
        :param center_x: float, center on the x axis
        :param center_y: float, center on the y axis

        """
        kwargs_slice = {'center_x': center_x, 'center_y': center_y, 'a': a, 'b': b, 'psi': psi, 'sigma_0': sigma_0}
        x_ = x - center_x
        y_ = y - center_y
        x_rot = x_*np.cos(psi) + y_*np.sin(psi)
        y_rot = -x_*np.sin(psi) + y_*np.cos(psi)
        try:
            len(x_)
        except:
            if (x_rot ** 2 / a ** 2) + (y_rot ** 2 / b ** 2) <= 1:
                return self.pot_in(x_, y_, kwargs_slice)
            else:
                return self.pot_ext(x_, y_, kwargs_slice)
        else:
            f = np.array([self.pot_in(x_[i], y_[i], kwargs_slice) if (x_rot[i] ** 2 / a ** 2) + (y_rot[i] ** 2 / b ** 2) <= 1
                          else self.pot_ext(x_[i], y_[i], kwargs_slice) for i in range(len(x_))])
            return f

    def derivatives(self,x, y, a, b, psi, sigma_0, center_x=0., center_y=0.):
        """
        lensing deflection angle

        :param a: float, semi-major axis, must be positive
        :param b: float, semi-minor axis, must be positive
        :param psi: float, orientation in radian
        :param sigma_0: float, surface mass density, must be positive
        :param center_x: float, center on the x axis
        :param center_y: float, center on the y axis

        """
        kwargs_slice = {'center_x': center_x, 'center_y': center_y, 'a': a, 'b': b, 'psi': psi, 'sigma_0': sigma_0}
        x_ = x - center_x
        y_ = y - center_y
        x_rot = x_ * np.cos(psi) + y_ * np.sin(psi)
        y_rot = -x_ * np.sin(psi) + y_ * np.cos(psi)
        try:
            len(x_)
        except:
            if (x_rot ** 2 / a ** 2) + (y_rot ** 2 / b ** 2) <= 1:
                return self.alpha_in(x_, y_, kwargs_slice)
            else:
                return self.alpha_ext(x_, y_, kwargs_slice)
        else:
            defl = np.array([self.alpha_in(x_[i], y_[i], kwargs_slice) if (x_rot[i] ** 2 / a ** 2) + (y_rot[i] ** 2 / b ** 2) <= 1
                             else self.alpha_ext(x_[i], y_[i], kwargs_slice) for i in range(len(x_))])
            return defl[:, 0], defl[:, 1]


    def hessian(self,x, y, a, b, psi, sigma_0, center_x=0., center_y=0.):
        """
        lensing second derivatives

        :param a: float, semi-major axis, must be positive
        :param b: float, semi-minor axis, must be positive
        :param psi: float, orientation in radian
        :param sigma_0: float, surface mass density, must be positive
        :param center_x: float, center on the x axis
        :param center_y: float, center on the y axis

        """

        x_ = x - center_x
        y_ = y - center_y
        diff = 0.000000001
        alpha_ra, alpha_dec = self.derivatives(x_, y_, a, b, psi, sigma_0, center_x, center_y)
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x_ + diff, y_, a, b, psi, sigma_0, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x_, y_ + diff, a, b, psi, sigma_0, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_yy, f_xy

    def density_2d(self,x,y,a,b,psi,sigma_0,center_x=0.,center_y=0.):
        """
        surface mass density : sigma_0 inside the slice and 0. otherwise

        :param a: float, semi-major axis, must be positive
        :param b: float, semi-minor axis, must be positive
        :param psi: float, orientation in radian
        :param sigma_0: float, surface mass density, must be positive
        :param center_x: float, center on the x axis
        :param center_y: float, center on the y axis

        """
        x_ = x - center_x
        y_ = y - center_y
        if (x_ ** 2 / a ** 2) + (y_ ** 2 / b ** 2) <= 1:
            return sigma_0
        else:
            return 0.

    def sign(self,z):
        """
        sign function

        :param z: complex

        """
        x = z.real
        y = z.imag
        if (x > 0 or (x == 0 and y >= 0)):
            return 1
        else:
            return -1

    def alpha_in(self,x, y, kwargs_slice):
        """
        deflection angle for (x,y) inside the elliptical slice

        :param kwargs_slice: dict, dictionary with  the slice definition (a,b,psi,sigma_0)

        """
        z = complex(x, y)
        zb = z.conjugate()
        psi = kwargs_slice['psi']
        e = (kwargs_slice['a'] - kwargs_slice['b']) / (kwargs_slice['a'] + kwargs_slice['b'])
        sig_0 = kwargs_slice['sigma_0']
        e2ipsi = c.exp(2j * psi)
        I_in = (z - e * zb * e2ipsi) * sig_0
        return I_in.real, I_in.imag

    def alpha_ext(self,x, y, kwargs_slice):
        """
        deflection angle for (x,y) outside the elliptical slice

        :param kwargs_slice: dict, dictionary with  the slice definition (a,b,psi,sigma_0)

        """
        z = complex(x, y)
        zb = z.conjugate()
        psi = kwargs_slice['psi']
        a = kwargs_slice['a']
        b = kwargs_slice['b']
        f2 = a ** 2 - b ** 2
        sig_0 = kwargs_slice['sigma_0']
        e2ipsi = c.exp(2j * psi)
        eipsi = c.exp(1j * psi)
        buf = zb ** 2 * e2ipsi - f2  ##problem with 0. and -0. giving different answers
        if buf.real == 0:
            buf = complex(0, buf.imag)
        if buf.imag == 0:
            buf = complex(buf.real, 0)
        I_out = 2 * a * b / f2 * (zb * e2ipsi - eipsi * self.sign(zb * eipsi) * c.sqrt(buf)) * sig_0
        return I_out.real, I_out.imag

    def pot_in(self,x, y, kwargs_slice):
        """
        lensing potential for (x,y) inside the elliptical slice

        :param kwargs_slice: dict, dictionary with  the slice definition (a,b,psi,sigma_0)

        """
        psi = kwargs_slice['psi']
        a = kwargs_slice['a']
        b = kwargs_slice['b']
        sig_0 = kwargs_slice['sigma_0']
        e = (a - b) / (a + b)
        rE = (a + b) / 2.
        pot_in = 0.5 * ((1 - e) * (x * np.cos(psi) + y * np.sin(psi)) ** 2 + (1 + e) * (
                    y * np.cos(psi) - x * np.sin(psi)) ** 2) * sig_0
        cst = sig_0 * rE ** 2 * (1 - e ** 2) * np.log(rE)
        return pot_in + cst

    def pot_ext(self,x, y, kwargs_slice):
        """
        lensing potential for (x,y) outside the elliptical slice

        :param kwargs_slice: dict, dictionary with  the slice definition (a,b,psi,sigma_0)

        """
        z = complex(x, y)
        zb = z.conjugate()
        psi = kwargs_slice['psi']
        a = kwargs_slice['a']
        b = kwargs_slice['b']
        sig_0 = kwargs_slice['sigma_0']
        e = (a - b) / (a + b)
        f2 = a ** 2 - b ** 2
        emipsi = c.exp(-1j * psi)
        em2ipsi = c.exp(-2j * psi)
        buf = z ** 2 * em2ipsi - f2  ##problem with 0. and -0. giving different answers
        if buf.real == 0:
            buf = complex(0, buf.imag)
        if buf.imag == 0:
            buf = complex(buf.real, 0)
        pot_ext = (1 - e ** 2) / (4 * e) * (f2 * c.log((self.sign(z * emipsi) * z * emipsi + c.sqrt(buf)) / 2.)
                                            - self.sign(z * emipsi) * z * emipsi * c.sqrt(buf) + z ** 2 * em2ipsi) * sig_0
        return pot_ext.real
