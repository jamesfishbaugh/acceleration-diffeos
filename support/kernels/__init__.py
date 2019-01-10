from enum import Enum

from support.kernels.abstract_kernel import AbstractKernel


class Type(Enum):
    from support.kernels.torch_kernel import TorchKernel
    from support.kernels.keops_kernel import KeopsKernel

    UNDEFINED = None
    NO_KERNEL = None
    TORCH = TorchKernel
    KEOPS = KeopsKernel


def factory(kernel_type, *args, **kwargs):
    """Return an instance of a kernel corresponding to the requested kernel_type"""

    # turn enum string to enum object
    if isinstance(kernel_type, str):
        try:
            for c in [' ', '-']:    # chars to be replaced for normalization
                kernel_type = kernel_type.replace(c, '_')
            kernel_type = Type[kernel_type.upper()]
        except:
            raise TypeError('kernel_type ' + kernel_type + ' could not be found')

    if not isinstance(kernel_type, Type):
        raise TypeError('kernel_type must be an instance of KernelType Enum')

    if kernel_type in [Type.UNDEFINED, Type.NO_KERNEL]:
        return None

    return kernel_type.value(*args, **kwargs)
