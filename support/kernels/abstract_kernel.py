import logging
from abc import ABC, abstractmethod
import torch
from core import default

logger = logging.getLogger(__name__)


class AbstractKernel(ABC):
    def __init__(self, kernel_type='undefined', kernel_width=None, device=default.deformation_kernel_device):
        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        self.device = device
        logger.debug('instantiating kernel %s with kernel_width %s on device %s. addr: %s', self.kernel_type, self.kernel_width, self.device, hex(id(self)))

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        raise NotImplementedError

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        raise NotImplementedError

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None:
            y = x
        assert (x.size(0) == y.size(0))
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)
