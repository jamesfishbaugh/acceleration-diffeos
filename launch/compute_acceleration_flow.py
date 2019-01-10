import logging

import torch

from core import default
from core.model_tools.deformations.acceleration_path import AccelerationPath
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
import support.kernels as kernel_factory

logger = logging.getLogger(__name__)


def compute_acceleration_flow(template_specifications,
                     dimension=default.dimension,
                     tensor_scalar_type=default.tensor_scalar_type,
                     tensor_integer_type=default.tensor_integer_type,

                     deformation_kernel_type=default.deformation_kernel_type,
                     deformation_kernel_width=default.deformation_kernel_width,
                     deformation_kernel_device=default.deformation_kernel_device,

                     shoot_kernel_type=None,
                     initial_control_points=default.initial_control_points,
                     initial_velocity=default.initial_velocity,
                     impulse_t=None,
                     tmin=default.tmin, tmax=default.tmax,
                     number_of_time_points=default.number_of_time_points,
                     output_dir=default.output_dir, **kwargs
                     ):
    print('[ compute_acceleration_flow function ]')

    """
    Create the template object
    """

    deformation_kernel = kernel_factory.factory(deformation_kernel_type, deformation_kernel_width, device=deformation_kernel_device)

    (object_list, t_name, t_name_extension,
     t_noise_variance, multi_object_attachment) = create_template_metadata(template_specifications, dimension)
    template = DeformableMultiObject(object_list)

    """
    Reading Control points and impulse
    """

    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
    else:
        raise RuntimeError('Please specify a path to control points to perform an acceleration flow')

    # Determine the number of control points
    [number_of_control_points, d] = control_points.shape
    
    if initial_velocity is not None:
        v0 = read_3D_array(initial_velocity)
    else:
        raise RuntimeError('Please specify an path to initial velocity to perform an acceleration flow')
       
    impulse = np.zeros((number_of_time_points, number_of_control_points, dimension)) 
        
    if impulse_t is not None:
        
        for i in range(0, number_of_time_points):
        
          cur_impulse_path = '%s%0.3d.txt' %(impulse_t, i)
          impulse[i,:,:] = read_3D_array(cur_impulse_path)
        
    else:
        raise RuntimeError('Please specify an path to impulse prefix to perform an acceleration flow')
    
    impulse_torch = torch.from_numpy(impulse).type(tensor_scalar_type)
    v0_torch = torch.from_numpy(v0).type(tensor_scalar_type)
    control_points_torch = torch.from_numpy(control_points).type(tensor_scalar_type)

    template_points = {key: torch.from_numpy(value).type(tensor_scalar_type)
                       for key, value in template.get_points().items()}
    template_data = {key: torch.from_numpy(value).type(tensor_scalar_type)
                     for key, value in template.get_data().items()}

    acceleration_path = AccelerationPath(kernel=kernel_factory.factory(deformation_kernel_type, deformation_kernel_width, device=deformation_kernel_device),
                                                  shoot_kernel_type=shoot_kernel_type, number_of_time_points=number_of_time_points)

    acceleration_path.set_tmin(tmin)
    acceleration_path.set_tmax(tmax)
    acceleration_path.set_template_points_tmin(template_points)
    acceleration_path.set_control_points_tmin(control_points_torch)
    acceleration_path.set_impulse_t(impulse_torch)
    acceleration_path.set_initial_velocity(v0_torch)
    acceleration_path.update()
    
    names = [elt for elt in t_name]
    acceleration_path.write2('AccelerationFlow', names, t_name_extension, template, template_data, None, output_dir)