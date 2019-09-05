import logging
import math

import torch
from torch.autograd import Variable

import support.kernels as kernel_factory
from core import default
from core.model_tools.deformations.acceleration_path import AccelerationPath
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import initialize_control_points, initialize_impulse, initialize_initial_velocity
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata

from numpy import linalg as LA
logger = logging.getLogger(__name__)


class AccelerationRegression(AbstractStatisticalModel):
    """
    Acceleration regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 number_of_threads=default.number_of_threads,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,
                 estimate_initial_velocity=default.estimate_initial_velocity,
                 initial_velocity_weight=default.initial_velocity_weight,
                 regularity_weight = default.regularity_weight,
                 data_weight = default.data_weight,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_impulse_t=None, initial_velocity=None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='AccelerationRegression')

        # Global-like attributes
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.number_of_threads = number_of_threads

        # Declare model structure
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points

        # Deformation
        self.acceleration_path = AccelerationPath(kernel=kernel_factory.factory(deformation_kernel_type, deformation_kernel_width,
                                                  device=deformation_kernel_device), shoot_kernel_type=shoot_kernel_type, number_of_time_points=number_of_time_points)

        # Template
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,  self.dimension)

        self.template = DeformableMultiObject(object_list)
        self.template.update()

        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type, smoothing_kernel_width,  device=deformation_kernel_device)

        # Template data
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points
        self.fixed_effects['control_points'] = initialize_control_points(initial_control_points, self.template,
                                                                         initial_cp_spacing, deformation_kernel_width,
                                                                         self.dimension, False)

        self.estimate_initial_velocity = estimate_initial_velocity
        self.initial_velocity_weight = initial_velocity_weight
        self.regularity_weight = regularity_weight
        self.data_weight = data_weight

        self.number_of_control_points = len(self.fixed_effects['control_points'])
        self.number_of_time_points = number_of_time_points

        # Impulse
        self.fixed_effects['impulse_t'] = initialize_impulse(initial_impulse_t, self.number_of_time_points,  self.number_of_control_points, self.dimension)
        if (self.estimate_initial_velocity):
            self.fixed_effects['initial_velocity'] = initialize_initial_velocity(initial_velocity, self.number_of_control_points, self.dimension)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp

    # Impulse ----------------------------------------------------------------------------------------------------------
    def get_impulse_t(self):
        return self.fixed_effects['impulse_t']

    def set_impulse_t(self, impulse_t):
        self.fixed_effects['impulse_t'] = impulse_t

    def get_initial_velocity(self):
        if (self.estimate_initial_velocity):
            return self.fixed_effects['initial_velocity']
        else:
            return np.zeros((self.number_of_control_points, self.dimension))

    def set_initial_velocity(self, initial_velocity):
        self.fixed_effects['initial_velocity'] = initial_velocity

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        out['impulse_t'] = self.fixed_effects['impulse_t']
        if self.estimate_initial_velocity:
            out['initial_velocity'] = self.fixed_effects['initial_velocity']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        self.set_impulse_t(fixed_effects['impulse_t'])
        if self.estimate_initial_velocity:
            self.set_initial_velocity(fixed_effects['initial_velocity'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.
        :param dataset: LongitudinalDataset instance
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, impulse_t, initial_velocity = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        data_attachment, regularity, velocity_regularity = self._compute_attachment_and_regularity(dataset, template_data, template_points,
                                                                                                   control_points, impulse_t, initial_velocity)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:

            total = self.initial_velocity_weight*velocity_regularity + self.regularity_weight*regularity + self.data_weight*data_attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(
                        template_data['landmark_points'].detach(),
                        template_data['landmark_points'].detach(),
                        gradient['landmark_points'].detach())

            # Control points
            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad

            # Initial velocity
            if self.estimate_initial_velocity:
                gradient['initial_velocity'] = initial_velocity.grad

            # Impulse t
            gradient['impulse_t'] = impulse_t.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return self.data_weight*data_attachment.detach().cpu().numpy(), \
                   self.regularity_weight*regularity.detach().cpu().numpy() + self.initial_velocity_weight*velocity_regularity.detach().cpu().numpy(), gradient

        else:

            return self.data_weight * data_attachment.detach().cpu().numpy(), \
                   self.regularity_weight * regularity.detach().cpu().numpy() + self.initial_velocity_weight * velocity_regularity.detach().cpu().numpy()

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, impulse_t, initial_velocity):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.acceleration_path.set_tmin(min(target_times))
        self.acceleration_path.set_tmax(max(target_times))
        self.acceleration_path.set_template_points_tmin(template_points)
        self.acceleration_path.set_control_points_tmin(control_points)
        self.acceleration_path.set_impulse_t(impulse_t)
        self.acceleration_path.set_initial_velocity(initial_velocity)
        self.acceleration_path.update()

        deformation_noise_variance = np.zeros(len(self.objects_noise_variance))
        for i in range(0, len(deformation_noise_variance)):
            self.objects_noise_variance[i] = 1.0

        data_attachment = 0.0
        for j, (time, obj) in enumerate(zip(target_times, target_objects)):
            deformed_points = self.acceleration_path.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            data_attachment += self.multi_object_attachment.compute_weighted_distance(deformed_data, self.template, obj, self.objects_noise_variance)

        regularity = self.acceleration_path.get_norm_squared()
        velocity_regularity = self.acceleration_path.get_velocity_norm()

        return data_attachment, regularity, velocity_regularity

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
                                       requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
                                         requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}

        control_points = self.fixed_effects['control_points']
        control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
                                  requires_grad=(not self.freeze_control_points and with_grad))

        # Impulse.
        impulse_t = self.fixed_effects['impulse_t']
        impulse_t = Variable(torch.from_numpy(impulse_t).type(self.tensor_scalar_type), requires_grad=with_grad)

        if (self.estimate_initial_velocity):
            initial_velocity = self.fixed_effects['initial_velocity']
            # Scale to unit norm
            norms = LA.norm(initial_velocity, axis=1) + 1e-6
            initial_velocity = initial_velocity / norms.reshape(-1, 1)
            # Now scale to the number of timesteps
            initial_velocity = initial_velocity / self.number_of_time_points
            self.fixed_effects['initial_velocity'] = initial_velocity
            initial_velocity = Variable(torch.from_numpy(initial_velocity).type(self.tensor_scalar_type),
                                        requires_grad=with_grad)
        else:
            initial_velocity_np = np.zeros((self.number_of_control_points, self.dimension))
            initial_velocity = Variable(torch.from_numpy(initial_velocity_np).type(self.tensor_scalar_type),
                                        requires_grad=False)

        return template_data, template_points, control_points, impulse_t, initial_velocity
    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_adjoint_parameters=False):
        self._write_model_predictions(output_dir, dataset, write_adjoint_parameters)
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, output_dir, dataset=None, write_adjoint_parameters=False):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, template_points, control_points, impulse_t, initial_velocity = self._fixed_effects_to_torch_tensors(False)
        target_times = dataset.times[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.acceleration_path.set_tmin(min(target_times))
        self.acceleration_path.set_tmax(max(target_times))
        self.acceleration_path.set_template_points_tmin(template_points)
        self.acceleration_path.set_control_points_tmin(control_points)
        self.acceleration_path.set_impulse_t(impulse_t)
        self.acceleration_path.set_initial_velocity(initial_velocity)
        self.acceleration_path.update()

        # Write --------------------------------------------------------------------------------------------------------
        self.acceleration_path.write2(self.name, self.objects_name, self.objects_name_extension, self.template,
                                     template_data, output_dir, write_adjoint_parameters)

        # Model predictions.
        if dataset is not None:
            for j, time in enumerate(target_times):
                names = []
                for k, (object_name, object_extension) in enumerate(
                        zip(self.objects_name, self.objects_name_extension)):
                    name = '%s__Reconstruction__%s__%0.03f%s' % (self.name, object_name, j, object_extension)
                    print(name)
                    names.append(name)
                deformed_points = self.acceleration_path.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def _write_model_parameters(self, output_dir):
        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Initial velocity
        write_3D_array(self.get_initial_velocity(), output_dir,
                       self.name + "__EstimatedParameters__InitialVelocity.txt")

        # Write impulse
        impulse_t = self.acceleration_path.get_impulse_t()
        [T, number_of_control_points, dimension] = impulse_t.shape
        for i in range(0, T):
            out_name = '%s__EstimatedParameters__Impulse_t_%0.3d.txt' % (self.name, i)
            cur_impulse = impulse_t[i, :, :].data.cpu().numpy()
            write_3D_array(cur_impulse, output_dir, out_name)