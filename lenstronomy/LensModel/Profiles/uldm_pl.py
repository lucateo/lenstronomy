__author__ = 'lucateo'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.Profiles.pemd import PEMD
from scipy.special import gamma as gamma_func
import numpy as np

__all__ = ['Uldm_PL']


class Uldm_PL(LensProfileBase):
    """
    Personal class
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'kappa_tilde', 'sampled_theta_c', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0., 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'kappa_tilde': 0., 'sampled_theta_c': 0., 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'kappa_tilde': 10, 'sampled_theta_c': 100.,'center_x': 100, 'center_y': 100}

    def __init__(self):
        self._uldm = Uldm()
        self._pl = PEMD()
        super(Uldm_PL, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, kappa_tilde, sampled_theta_c, center_x=0, center_y=0):
        """
        lensing potential of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: lensing potential correction
        """
        kappa_0 = self._kappa_0_real(theta_E, kappa_tilde, sampled_theta_c)
        theta_c = self._half_density_thetac(theta_E, kappa_tilde, sampled_theta_c)
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        #  kappa_E = self._kappa_E(theta_E, kappa_0, sampled_theta_c)
        #  lambda_approx = 1 - kappa_E

        f_uldm_density = self._uldm.function(x, y, kappa_0, theta_c, slope, center_x, center_y)
        #  f_pl = lambda_approx * self._pl.function(x, y, theta_E, gamma, e1, e2, kappa_ext, center_x, center_y)
        f_pl = self._pl.function(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        return f_uldm_density + f_pl

    def derivatives(self, x, y, theta_E, gamma, e1, e2, kappa_tilde, sampled_theta_c, center_x=0, center_y=0):
        """
        deflection angles of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param lambda_approx: approximate mass sheet transform
        :param r_core: core radius of the cored density profile
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: alpha_x, alpha_y
        """
        kappa_0 = self._kappa_0_real(theta_E, kappa_tilde, sampled_theta_c)
        theta_c = self._half_density_thetac(theta_E, kappa_tilde, sampled_theta_c)
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        #  kappa_E = self._kappa_E(theta_E, kappa_0, sampled_theta_c)
        #  lambda_approx = 1 - kappa_E

        f_x_uldm, f_y_uldm = self._uldm.derivatives(x, y, kappa_0, theta_c, slope, center_x, center_y)
        #  f_x_pl, f_y_pl = lambda_approx * self._pl.derivatives(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        f_x_pl, f_y_pl = self._pl.derivatives(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        return f_x_uldm + f_x_pl, f_y_uldm + f_y_pl

    def hessian(self, x, y, theta_E, gamma, e1, e2, kappa_tilde, sampled_theta_c, center_x=0, center_y=0):
        """
        Hessian terms of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param lambda_approx: approximate mass sheet transform
        :param r_core: core radius of the cored density profile
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: df/dxx, df/dyy, df/dxy
        """
        kappa_0 = self._kappa_0_real(theta_E, kappa_tilde, sampled_theta_c)
        theta_c = self._half_density_thetac(theta_E, kappa_tilde, sampled_theta_c)
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        #  kappa_E = self._kappa_E(theta_E, kappa_0, sampled_theta_c)
        #  lambda_approx = 1 - kappa_E

        f_xx_uldm, f_yy_uldm, f_xy_uldm = self._uldm.hessian(x, y, kappa_0, theta_c, slope, center_x, center_y)
        #  f_xx_pl, f_yy_pl, f_xy_pl = lambda_approx * self._convergence.hessian(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        f_xx_pl, f_yy_pl, f_xy_pl = self._pl.hessian(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        return f_xx_uldm + f_xx_pl, f_yy_uldm + f_yy_pl, f_xy_uldm + f_xy_pl

    def _theta_c_true(self, sampled_theta_c):
        """
        Recover the real theta_c from the sampling option
        """
        #  theta_c = 10*(np.log(sampled_theta_c) - np.log(1 - sampled_theta_c))
        theta_c = sampled_theta_c
        return theta_c

    def _z_factor(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        z factor, z = A/\lambda
        """
        theta_c = self._theta_c_true(sampled_theta_c)
        #if kappa_tilde != 0:
        angular_factor = theta_E / (kappa_tilde * theta_c)
        #else:
        #    angular_factor = 0
        return angular_factor / (2* np.sqrt(np.pi)  )

    def _a_fit(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        """
        z_factor = np.abs(self._z_factor(theta_E, kappa_tilde, sampled_theta_c))
        # print(theta_E, kappa_tilde, sampled_theta_c, z_factor)
        a_fit = 0.23 * np.sqrt(1 + 7.5 * z_factor * np.tanh( 1.5 * z_factor**(0.24)) )
        return a_fit

    def _slope(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        """
        z_factor = np.abs(self._z_factor(theta_E, kappa_tilde, sampled_theta_c))
        b_fit = 1.69 + 2.23/(1 + 2.2 * z_factor)**(2.47)
        return 2 * b_fit

    def _half_density_thetac(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        """
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        core_half_factor = np.sqrt(0.5**(-1/slope) -1)
        return self._theta_c_true(sampled_theta_c) * core_half_factor

    def _kappa_0_real(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        """
        a_fit = self._a_fit(theta_E, kappa_tilde, sampled_theta_c)
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        return gamma_func(slope - 0.5) * kappa_tilde / (gamma_func(slope) * a_fit**2)

    def _kappa_E(self, theta_E, kappa_tilde, sampled_theta_c):
        """
        """
        kappa_0 = self._kappa_0_real(theta_E, kappa_tilde, sampled_theta_c)
        theta_c = self._half_density_thetac(theta_E, kappa_tilde, sampled_theta_c)
        slope = self._slope(theta_E, kappa_tilde, sampled_theta_c)
        return self._uldm.kappa_r(theta_E, kappa_0, theta_c, slope)
