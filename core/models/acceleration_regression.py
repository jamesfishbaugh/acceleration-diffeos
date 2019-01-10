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
from core.observations.deformable_objects import image

from numpy import linalg as LA
from scipy.ndimage.filters import gaussian_filter

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
                 use_intensity_model=False,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,
                 estimate_initial_velocity=default.estimate_initial_velocity,
                 initial_velocity_weight=default.initial_velocity_weight,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_impulse_t = None, initial_velocity=None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='AccelerationRegression')

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.number_of_threads = number_of_threads

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points

        self.use_intensity_model = use_intensity_model


        # Deformation.
        self.acceleration_path = AccelerationPath(kernel=kernel_factory.factory(deformation_kernel_type, deformation_kernel_width, device=deformation_kernel_device),
                                                  shoot_kernel_type=shoot_kernel_type, number_of_time_points=number_of_time_points)
        
        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications, self.dimension)

        self.template = DeformableMultiObject(object_list)
        self.template.update()

        template_data = self.template.get_data()

        if ('image_intensities' in template_data.keys()) and (self.use_intensity_model):
            intensities = template_data['image_intensities']
            #if (self.dimension == 2):
            #    [width, height] = intensities.shape
            #    self.fixed_effects['slope_images_t'] = np.zeros((number_of_time_points, width, height))
            #else:
            #    [width, height, length] = intensities.shape
            #    self.fixed_effects['slope_images_t'] = np.zeros((number_of_time_points, width, height, length))
            #self.slope_image = np.zeros(intensities.shape)
            self.fixed_effects['slope_image'] = np.zeros(intensities.shape)

        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type, smoothing_kernel_width, device=deformation_kernel_device)

        self.deformation_attachment_weight = 10
        self.intensity_attachment_weight = 1
        self.total_variation_weight = 1
        self.deformation_regularity_weight = 1

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width, self.dimension, False)

        self.estimate_initial_velocity = estimate_initial_velocity
        self.initial_velocity_weight = initial_velocity_weight

        self.number_of_control_points = len(self.fixed_effects['control_points'])
        self.number_of_time_points = number_of_time_points

        # Impulse
        self.fixed_effects['impulse_t'] = initialize_impulse(initial_impulse_t, self.number_of_time_points, self.number_of_control_points, self.dimension)
        if (self.estimate_initial_velocity):
            self.fixed_effects['initial_velocity'] =  initialize_initial_velocity(initial_velocity, self.number_of_control_points, self.dimension)
        
                
    def initialize_noise_variance(self, dataset):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, impulse_t, initial_velocity = self._fixed_effects_to_torch_tensors(False)
            target_times = dataset.times[0]
            target_objects = dataset.deformable_objects[0]

            self.acceleration_path.set_tmin(min(target_times))
            self.acceleration_path.set_tmax(max(target_times))
            self.acceleration_path.set_template_points_tmin(template_points)
            self.acceleration_path.set_control_points_tmin(control_points)
            self.acceleration_path.set_impulse_t(impulse_t)
            self.acceleration_path.set_initial_velocity(initial_velocity)
            self.acceleration_path.update()

            residuals = np.zeros((self.number_of_objects,))
            for (time, target) in zip(target_times, target_objects):
                deformed_points = self.acceleration_path.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals += self.multi_object_attachment.compute_distances(deformed_data, self.template, target).data.numpy()

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(len(target_times))
                    self.objects_noise_variance[k] = nv
                    print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

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
        # self.number_of_control_points = len(cp)

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

    def get_slope_image(self):
        if (self.use_intensity_model):
            return self.fixed_effects['slope_image']
        else:
            return None

    def set_slope_image(self, slope_image):
        self.fixed_effects['slope_image'] = slope_image

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
        if self.use_intensity_model:
            out['slope_image'] = self.fixed_effects['slope_image']
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
        if self.use_intensity_model:
            self.set_slope_image(fixed_effects['slope_image'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False, cur_iter=None):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, impulse_t, initial_velocity, slope_image = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        deformation_attachment, intensity_attachment, regularity, velocity_regularity, total_variation = self._compute_attachment_and_regularity(dataset, template_data, template_points, control_points, impulse_t, initial_velocity, slope_image)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = self.total_variation_weight*total_variation + self.initial_velocity_weight*velocity_regularity + self.deformation_regularity_weight*regularity + self.deformation_attachment_weight*deformation_attachment + self.intensity_attachment_weight*intensity_attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(template_data['landmark_points'].detach(), 
                                                                               template_data['landmark_points'].detach(), 
                                                                               gradient['landmark_points'].detach())

            # Intensity change model
            if 'image_intensities' in template_data.keys() and self.use_intensity_model:
                gradient['slope_image'] = slope_image.grad
                #if (cur_iter is not None) and (cur_iter > 1):
                #    gradient['slope_image'] = slope_image.grad
                #else:
                #    gradient['slope_image'] = torch.zeros_like(slope_image)

            # Control points
            if not self.freeze_control_points: 
              gradient['control_points'] = control_points.grad
            
            # Initial velocity
            if self.estimate_initial_velocity: 
              gradient['initial_velocity'] = initial_velocity.grad
              #print(initial_velocity)
            
            # Impulse t
            gradient['impulse_t'] = impulse_t.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return self.deformation_attachment_weight*deformation_attachment.detach().cpu().numpy()+self.intensity_attachment_weight*intensity_attachment.detach().cpu().numpy(), self.deformation_regularity_weight*regularity.detach().cpu().numpy() + self.initial_velocity_weight*velocity_regularity.detach().cpu().numpy() + self.total_variation_weight*total_variation.detach().cpu().numpy(), gradient

        else:

            return self.deformation_attachment_weight*deformation_attachment.detach().cpu().numpy()+self.intensity_attachment_weight*intensity_attachment.detach().cpu().numpy(), self.deformation_regularity_weight*regularity.detach().cpu().numpy() + self.initial_velocity_weight*velocity_regularity.detach().cpu().numpy() + self.total_variation_weight*total_variation.detach().cpu().numpy()


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, impulse_t, initial_velocity, slope_image):
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

        # a = template_data['image_intensities']
        # height, width = a.size()
        # dy = torch.abs(a[-1:, :] - a[:-1, :])
        # error = torch.norm(dy, 1)
        # total_variation = -(error/height)
        # total_variation = torch.zeros_like(total_variation)
        total_variation = torch.zeros([1], dtype=torch.float32, requires_grad=True)

        deformation_attachment = 0.
        intensity_attachment = 0.
        for j, (time, obj) in enumerate(zip(target_times, target_objects)):
            deformed_points = self.acceleration_path.get_template_points(time)

            if self.use_intensity_model and slope_image is not None:
                linear_image_model = {}
                #zero_tensor = torch.zeros_like(slope_image)
                # The slope image is zero in areas where the baseline image is zero
                #slope_image = torch.where(template_data['image_intensities'] < 0.05, zero_tensor, slope_image )
                #self.fixed_effects['slope_image'] = slope_image
                #slope_image = zero_tensor
                linear_image_model['image_intensities'] = slope_image*time + template_data['image_intensities']

                deformed_data_withitensity = self.template.get_deformed_data(deformed_points, linear_image_model)
                deformed_data_nointensity = self.template.get_deformed_data(deformed_points, template_data)


                if (self.dimension == 2):
                    # Compute total variation norm
                    height, width = slope_image.size()
                    dy = torch.abs(slope_image[-1:, :] - slope_image[:-1, :])
                    error = torch.norm(dy, 1)
                    total_variation = -(error/height)

            else:
                deformed_data_nointensity = self.template.get_deformed_data(deformed_points, template_data)

            #attachment -= self.multi_object_attachment.compute_weighted_distance(deformed_data, self.template, obj, self.objects_noise_variance)
            deformation_attachment -= self.multi_object_attachment.compute_weighted_distance(deformed_data_nointensity, self.template, obj, self.objects_noise_variance)
            if self.use_intensity_model and slope_image is not None:
                intensity_attachment -= self.multi_object_attachment.compute_weighted_distance(deformed_data_withitensity, self.template, obj, self.objects_noise_variance)
            else:
                intensity_attachment -= (deformation_attachment-deformation_attachment)

        regularity = - self.acceleration_path.get_norm_squared()
        
        velocity_regularity = - self.acceleration_path.get_velocity_norm()

        #print(deformation_attachment)
        #print(intensity_attachment)
        #print(regularity)
        #print(velocity_regularity)
        #print(total_variation)


        return deformation_attachment, intensity_attachment, regularity, velocity_regularity, total_variation

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

        if self.use_intensity_model:
            slope_image = self.fixed_effects['slope_image']
            slope_image = gaussian_filter(slope_image, sigma=1.75)
            self.fixed_effects['slope_image'] = slope_image
            slope_image = Variable(torch.from_numpy(slope_image).type(self.tensor_scalar_type),
                                    requires_grad=(self.use_intensity_model and with_grad))
        else:
            slope_image = None

        if (self.estimate_initial_velocity):
            initial_velocity = self.fixed_effects['initial_velocity']
            # Scale to unit norm
            norms = LA.norm(initial_velocity, axis=1) + 1e-6
            initial_velocity = initial_velocity / norms.reshape(-1,1)
            # Now scale to the number of timesteps
            initial_velocity = initial_velocity / self.number_of_time_points
            self.fixed_effects['initial_velocity'] = initial_velocity
            initial_velocity = Variable(torch.from_numpy(initial_velocity).type(self.tensor_scalar_type), requires_grad=with_grad)
        else:
            initial_velocity_np = np.zeros((self.number_of_control_points, self.dimension))
            initial_velocity = Variable(torch.from_numpy(initial_velocity_np).type(self.tensor_scalar_type), requires_grad=False)

        return template_data, template_points, control_points, impulse_t, initial_velocity, slope_image

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_adjoint_parameters=False):
        self._write_model_predictions(output_dir, dataset, write_adjoint_parameters)
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, output_dir, dataset=None, write_adjoint_parameters=False):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, template_points, control_points, impulse_t, initial_velocity, slope_image = self._fixed_effects_to_torch_tensors(False)
        target_times = dataset.times[0]

        # Slope image
        if self.use_intensity_model and slope_image is not None:
            out_image = image.Image(self.dimension)
            out_image.set_intensities(slope_image.data.cpu().numpy())

            if (self.dimension == 2):
                out_image.set_dtype(np.dtype(np.uint8))
                out_image.write(output_dir, self.name + "__slope_image.png")
            else:
                out_image.set_dtype(np.dtype(np.float32))
                out_image.write(output_dir, self.name + "__slope_image.nii")

        # Deform -------------------------------------------------------------------------------------------------------
        self.acceleration_path.set_tmin(min(target_times))
        self.acceleration_path.set_tmax(max(target_times))
        self.acceleration_path.set_template_points_tmin(template_points)
        self.acceleration_path.set_control_points_tmin(control_points)
        self.acceleration_path.set_impulse_t(impulse_t)
        self.acceleration_path.set_initial_velocity(initial_velocity)
        self.acceleration_path.update()

        # Write --------------------------------------------------------------------------------------------------------
        self.acceleration_path.write(self.name, self.objects_name, self.objects_name_extension, self.template, template_data, slope_image, output_dir, write_adjoint_parameters)

        # Model predictions.
        if dataset is not None:
            for j, time in enumerate(target_times):
                names = []
                for k, (object_name, object_extension) in enumerate(zip(self.objects_name, self.objects_name_extension)):
                    name = '%s__Reconstruction__%s__%0.03f%s' %(self.name, object_name, j, object_extension)
                    print(name)
                    names.append(name)
                deformed_points = self.acceleration_path.get_template_points(time)
                if self.use_intensity_model and slope_image is not None:
                    linear_image_model = {}
                    linear_image_model['image_intensities'] = slope_image * time + template_data['image_intensities']
                    deformed_data = self.template.get_deformed_data(deformed_points, linear_image_model)
                else:
                    deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def _write_model_parameters(self, output_dir):
        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Initial velocity
        write_3D_array(self.get_initial_velocity(), output_dir, self.name + "__EstimatedParameters__InitialVelocity.txt")
        
        # Write impulse
        impulse_t = self.acceleration_path.get_impulse_t()
        [T, number_of_control_points, dimension] = impulse_t.shape
        for i in range(0, T):
          out_name = '%s__EstimatedParameters__Impulse_t_%0.3d.txt' %(self.name, i)
          cur_impulse = impulse_t[i,:,:].data.cpu().numpy()
          write_3D_array(cur_impulse, output_dir, out_name)


