import logging

import torch

from core import default
from support.kernels.abstract_kernel import AbstractKernel

logger = logging.getLogger(__name__)


def gaussian(r2, s):
    return torch.exp(-r2 / (s * s))


def binet(prs):
    return prs ** 2


class TorchKernel(AbstractKernel):
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel_width=None, device=default.deformation_kernel_device, **kwargs):
        if device.lower() == 'auto':
            device = self.get_auto_device()

        elif device.lower() == 'gpu':
            device = 'cuda'

        super().__init__('torch', kernel_width, device.lower())

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode='gaussian'):
        # move tensors to device if needed

        res = None

        if mode in ['gaussian', 'pointcloud']:
            previous_device = x.device.type
            (x, y, p) = map(self.__move_tensor_to_device_if_needed, [x, y, p])

            sq = self._squared_distances(x, y)

            # if x.type() == 'torch.cuda.FloatTensor'
            res = torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
            if previous_device == 'cpu':
                res = res.cpu()

        elif mode == 'varifold':
            assert isinstance(x, tuple), 'x must be a tuple'
            assert len(x) == 2, 'tuple length must be 2'
            assert isinstance(y, tuple), 'y must be a tuple'
            assert len(y) == 2, 'tuple length must be 2'

            previous_device = x[0].device.type
            (x, y, p) = map(self.__move_tensor_to_device_if_needed, [x, y, p])

            sq = self._squared_distances(x[0], y[0])
            res = torch.mm(gaussian(sq, self.kernel_width) * binet(torch.mm(x[1], torch.t(y[1]))), p)
            if previous_device == 'cpu':
                res = res.cpu()
        else:
            raise RuntimeError('Unknown kernel mode.')

        assert res is not None
        return res

    def convolve_gradient(self, px, x, y=None, py=None):
        if y is None:
            y = x
        if py is None:
            py = px

        # A=exp(-(x_i - y_j)^2/(ker^2)).
        sq = self._squared_distances(x, y)
        A = torch.exp(-sq / (self.kernel_width ** 2))

        # B=(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        return (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    @staticmethod
    def _differences(x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin

    @staticmethod
    def get_auto_device():
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        return device

    def __move_tensor_to_device_if_needed(self, t):
        """
        Move tensor t to self.device
        :param t:   Can either be a torch.Tensor object or a tuple of torch.Tensor
        :return:    torch.Tensor object on the defined device or tuple of torch.Tensor
        """

        def move(t, device):
            res = t.to(device=device)
            assert res.device.type == self.device, 'error moving tensor to device'
            return res

        res = None

        if isinstance(t, tuple):
            res = ()
            for tt in t:
                res = res + (move(tt, self.device),)   # append to tuple

        elif t is not None and t.device is not self.device:
            res = move(t, self.device)

        return res
