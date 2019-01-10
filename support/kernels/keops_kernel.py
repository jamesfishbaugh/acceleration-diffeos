from support.kernels import AbstractKernel
from pykeops.torch import generic_sum
# from pykeops.torch.generic.generic_red import Genred as generic_sum
from core import default


import logging
logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None, device=default.deformation_kernel_device, **kwargs):

        if device.lower() == 'cuda':
            device = 'GPU'

        super().__init__('keops', kernel_width, device)

        self.gamma = 1. / default.tensor_scalar_type([self.kernel_width ** 2])

        self.gaussian_convolve = []
        self.point_cloud_convolve = []
        self.varifold_convolve = []
        self.gaussian_convolve_gradient_x = []

        for dimension in [2, 3]:
            self.gaussian_convolve.append(generic_sum(
                "Exp(-G*SqDist(X,Y)) * P",
                "O = Vx(" + str(dimension) + ")",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "P = Vy(" + str(dimension) + ")"))

            self.point_cloud_convolve.append(generic_sum(
                "Exp(-G*SqDist(X,Y)) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "P = Vy(1)"))

            self.varifold_convolve.append(generic_sum(
                "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Nx = Vx(" + str(dimension) + ")",
                "Ny = Vy(" + str(dimension) + ")",
                "P = Vy(1)"))

            self.gaussian_convolve_gradient_x.append(generic_sum(
                "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                "O = Vx(" + str(dimension) + ")",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Px = Vx(" + str(dimension) + ")",
                "Py = Vy(" + str(dimension) + ")"))

            # #   Note: the following syntax corresponds to the new upcoming Keops syntax (>v0.0.89)
            # self.gaussian_convolve.append(generic_sum(
            #     "Exp(-G*SqDist(X,Y)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "P = Vy(" + str(dimension) + ")"],
            #     axis=1))
            #
            # self.point_cloud_convolve.append(generic_sum(
            #     "Exp(-G*SqDist(X,Y)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "P = Vy(1)"],
            #     axis=1))
            #
            # self.varifold_convolve.append(generic_sum(
            #     "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "Nx = Vx(" + str(dimension) + ")",
            #      "Ny = Vy(" + str(dimension) + ")",
            #      "P = Vy(1)"],
            #     axis=1))
            #
            # self.gaussian_convolve_gradient_x.append(generic_sum(
            #     "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "Px = Vx(" + str(dimension) + ")",
            #      "Py = Vy(" + str(dimension) + ")"],
            #     axis=1))

    def convolve(self, x, y, p, mode='gaussian'):
        if mode == 'gaussian':
            d = x.size(1)
            return self.gaussian_convolve[d - 2](self.gamma.type(x.type()), x, y, p, backend=self.device)

        elif mode == 'pointcloud':
            d = x.size(1)
            return self.point_cloud_convolve[d - 2](self.gamma.type(x.type()), x, y, p, backend=self.device)

        elif mode == 'varifold':
            x, nx = x
            y, ny = y
            d = x.size(1)
            return self.varifold_convolve[d - 2](self.gamma.type(x.type()), x, y, nx, ny, p, backend=self.device)

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        if y is None:
            y = x
        if py is None:
            py = px

        d = x.size(1)
        return -2 * self.gamma.type(x.type()) * self.gaussian_convolve_gradient_x[d - 2](
            self.gamma.type(x.type()), x, y, px, py, backend=self.device)
