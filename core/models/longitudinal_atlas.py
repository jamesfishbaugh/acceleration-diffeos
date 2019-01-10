import gc
import gc
import glob
import logging
import math
import os
import os.path
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import torch
from scipy.stats import norm
from torch.autograd import Variable

import support.kernels as kernel_factory
from core import default
from core.model_tools.deformations.spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import initialize_control_points, initialize_momenta, initialize_modulation_matrix, initialize_sources, \
    initialize_onset_ages, initialize_accelerations, initialize_covariance_momenta_inverse
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from support.probability_distributions.multi_scalar_truncated_normal_distribution import MultiScalarTruncatedNormalDistribution

logger = logging.getLogger(__name__)


def compute_exponential_and_attachment(args):
    # Read inputs and restore the general settings.
    i, j, general_settings, exponential, template_data, template, target, multi_object_attachment = args
    Settings().initialize(general_settings)

    # Deform and compute the distance.
    exponential.update()
    deformed_points = exponential.get_template_points()
    deformed_data = template.get_deformed_data(deformed_points, template_data)
    residual = multi_object_attachment.compute_distances(deformed_data, template, target)

    del template_data, deformed_points, deformed_data
    gc.collect()
    return i, j, residual


class LongitudinalAtlas(AbstractStatisticalModel):
    """
    Longitudinal atlas object class.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", Bône et al. (2018), Computer Vision and Pattern Recognition conference.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_threads=default.number_of_threads,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 concentration_of_time_points=default.concentration_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot,
                 use_rk2_for_flow=default.use_rk2_for_flow,
                 t0=default.t0,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 number_of_sources=default.number_of_sources,
                 initial_modulation_matrix=default.initial_modulation_matrix,
                 freeze_modulation_matrix=default.freeze_modulation_matrix,

                 freeze_reference_time=default.freeze_reference_time,

                 initial_time_shift_variance=default.initial_time_shift_variance,
                 freeze_time_shift_variance=default.freeze_time_shift_variance,

                 initial_acceleration_mean=default.initial_acceleration_mean,
                 initial_acceleration_variance=default.initial_acceleration_variance,
                 freeze_acceleration_variance=default.freeze_acceleration_variance,

                 freeze_noise_variance=default.freeze_noise_variance,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='LongitudinalAtlas')

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.number_of_threads = number_of_threads

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None
        self.fixed_effects['modulation_matrix'] = None
        self.fixed_effects['reference_time'] = None
        self.fixed_effects['time_shift_variance'] = None
        self.fixed_effects['acceleration_variance'] = None
        self.fixed_effects['noise_variance'] = None

        self.is_frozen = {'template_data': freeze_template, 'control_points': freeze_control_points,
                          'momenta': freeze_momenta, 'modulation_matrix': freeze_modulation_matrix,
                          'reference_time': freeze_reference_time, 'time_shift_variance': freeze_time_shift_variance,
                          'acceleration_variance': freeze_acceleration_variance,
                          'noise_variance': freeze_noise_variance}

        self.priors['template_data'] = {}
        self.priors['control_points'] = MultiScalarNormalDistribution()
        self.priors['momenta'] = MultiScalarNormalDistribution()
        self.priors['modulation_matrix'] = MultiScalarNormalDistribution()
        self.priors['reference_time'] = MultiScalarNormalDistribution()
        self.priors['time_shift_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['acceleration_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        self.individual_random_effects['acceleration'] = MultiScalarTruncatedNormalDistribution()

        # Deformation.
        self.spatiotemporal_reference_frame = SpatiotemporalReferenceFrame(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(deformation_kernel_type, deformation_kernel_width,
                                          device=deformation_kernel_device),
            shoot_kernel_type=shoot_kernel_type,
            concentration_of_time_points=concentration_of_time_points, number_of_time_points=number_of_time_points,
            t0=t0, use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)
        self.spatiotemporal_reference_frame_is_modified = True

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         objects_noise_variance, self.multi_object_attachment) = create_template_metadata(
            template_specifications, self.dimension)

        self.template = DeformableMultiObject(object_list)
        self.template.update()

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dimension, self.objects_name)
        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type, smoothing_kernel_width,
                                                         device=deformation_kernel_device)

        # Template data.
        self.set_template_data(self.template.get_data())
        self.__initialize_template_data_prior()

        # Control points.
        self.set_control_points(initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode))
        self.number_of_control_points = len(self.fixed_effects['control_points'])
        self.__initialize_control_points_prior()

        # Momenta.
        self.set_momenta(initialize_momenta(initial_momenta, self.number_of_control_points, self.dimension))
        self.__initialize_momenta_prior()

        # Modulation matrix.
        self.number_of_sources = number_of_sources
        self.fixed_effects['modulation_matrix'] = initialize_modulation_matrix(
            initial_modulation_matrix, self.number_of_control_points, self.number_of_sources, self.dimension)
        self.number_of_sources = self.get_modulation_matrix().shape[1]
        self.__initialize_modulation_matrix_prior()

        # Reference time.
        self.set_reference_time(t0)
        self.__initialize_reference_time_prior(initial_time_shift_variance)

        # Time-shift variance.
        self.set_time_shift_variance(initial_time_shift_variance)
        self.__initialize_time_shift_variance_prior()

        # Acceleration variance.
        if initial_acceleration_variance is not None:
            self.set_acceleration_variance(initial_acceleration_variance)
        else:
            acceleration_std = 1.5
            print('>> The initial acceleration std fixed effect is ARBITRARILY set to %.1f.' % acceleration_std)
            self.set_acceleration_variance(acceleration_std ** 2)
        self.__initialize_acceleration_variance_prior()

        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Source random effect.
        assert self.number_of_sources is not None, \
            'Please specify the number of sources, or provide a modulation matrix file.'
        self.individual_random_effects['sources'].set_mean(np.zeros((self.number_of_sources,)))
        self.individual_random_effects['sources'].set_variance(1.0)

        # Time-shift random effect.
        assert self.individual_random_effects['onset_age'].mean is not None
        assert self.individual_random_effects['onset_age'].variance_sqrt is not None

        # Acceleration random effect.
        acceleration_mean = self.individual_random_effects['acceleration'].get_mean()
        if initial_acceleration_mean is None:
            self.individual_random_effects['acceleration'].set_mean(np.ones((1,)))
        elif isinstance(acceleration_mean, float):
            self.individual_random_effects['acceleration'].set_mean(np.ones((1,)) * acceleration_mean)

    def initialize_random_effects_realization(
            self, number_of_subjects,
            initial_sources=default.initial_sources,
            initial_onset_ages=default.initial_onset_ages,
            initial_accelerations=default.initial_accelerations,
            **kwargs):

        # Initialize the random effects realization.
        individual_RER = {
            'sources': initialize_sources(initial_sources, number_of_subjects, self.number_of_sources),
            'onset_age': initialize_onset_ages(initial_onset_ages, number_of_subjects, self.get_reference_time()),
            'acceleration': initialize_accelerations(initial_accelerations, number_of_subjects)
        }

        return individual_RER

    def initialize_noise_variance(self, dataset, individual_RER):
        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, normalized_dof in enumerate(self.objects_noise_variance_prior_normalized_dof):
            dof = dataset.total_number_of_observations * normalized_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].degrees_of_freedom.append(dof)

        if np.min(self.fixed_effects['noise_variance']) < 0.0:
            # Prior on the noise variance (inverse Wishart: scale scalars parameters).
            (template_data, template_points, control_points,
             momenta, modulation_matrix) = self._fixed_effects_to_torch_tensors(False)
            sources, onset_ages, accelerations = self._individual_RER_to_torch_tensors(individual_RER, False)
            absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations)
            self._update_spatiotemporal_reference_frame(
                template_points, control_points, momenta, modulation_matrix, tmin, tmax)
            residuals = self._compute_residuals(dataset, template_data, absolute_times, sources)

            residuals_per_object = np.zeros((self.number_of_objects,))
            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    residuals_per_object += residuals[i][j].detach().cpu().numpy()

            for k, scale_std in enumerate(self.objects_noise_variance_prior_scale_std):
                if scale_std is None:
                    self.priors['noise_variance'].scale_scalars.append(
                        0.01 * residuals_per_object[k] / self.priors['noise_variance'].degrees_of_freedom[k])
                else:
                    self.priors['noise_variance'].scale_scalars.append(scale_std ** 2)

            # New, more informed initial value for the noise variance.
            self.fixed_effects['noise_variance'] = np.array(self.priors['noise_variance'].scale_scalars)

        else:
            for k, object_noise_variance in enumerate(self.fixed_effects['noise_variance']):
                self.priors['noise_variance'].scale_scalars.append(object_noise_variance)

    def __initialize_template_data_prior(self):
        """
        Initialize the template data prior.
        """
        # If needed (i.e. template not frozen), initialize the associated prior.
        if not self.is_frozen['template_data']:
            template_data = self.get_template_data()

            for key, value in template_data.items():
                # Initialization.
                self.priors['template_data'][key] = MultiScalarNormalDistribution()

                # Set the template data prior mean as the initial template data.
                self.priors['template_data'][key].mean = value

                if key == 'landmark_points':
                    # Set the template data prior standard deviation to the deformation kernel width.
                    self.priors['template_data'][key].set_variance_sqrt(
                        self.spatiotemporal_reference_frame.get_kernel_width())
                elif key == 'image_intensities':
                    # Arbitrary value.
                    std = 0.5
                    logger.info('Template image intensities prior std parameter is ARBITRARILY set to %.3f.' % std)
                    self.priors['template_data'][key].set_variance_sqrt(std)

    def __initialize_control_points_prior(self):
        """
        Initialize the control points prior.
        """
        # If needed (i.e. control points not frozen), initialize the associated prior.
        if not self.is_frozen['control_points']:
            # Set the control points prior mean as the initial control points.
            self.priors['control_points'].set_mean(self.get_control_points())
            # Set the control points prior standard deviation to the deformation kernel width.
            self.priors['control_points'].set_variance_sqrt(self.spatiotemporal_reference_frame.get_kernel_width())

    def __initialize_momenta_prior(self):
        """
        Initialize the momenta prior.
        """
        # If needed (i.e. momenta not frozen), initialize the associated prior.
        if not self.is_frozen['momenta']:
            # Set the momenta prior mean as the initial momenta.
            self.priors['momenta'].set_mean(self.get_momenta())
            # Set the momenta prior variance as the norm of the initial rkhs matrix.
            assert self.spatiotemporal_reference_frame.get_kernel_width() is not None
            rkhs_matrix = initialize_covariance_momenta_inverse(
                self.fixed_effects['control_points'], self.spatiotemporal_reference_frame.exponential.kernel,
                self.dimension)
            self.priors['momenta'].set_variance(1. / np.linalg.norm(rkhs_matrix))  # Frobenius norm.
            print('>> Momenta prior std set to %.3E.' % self.priors['momenta'].get_variance_sqrt())

    def __initialize_modulation_matrix_prior(self):
        """
        Initialize the modulation matrix prior.
        """
        # If needed (i.e. modulation matrix not frozen), initialize the associated prior.
        if not self.is_frozen['modulation_matrix']:
            # Set the modulation_matrix prior mean as the initial modulation_matrix.
            self.priors['modulation_matrix'].set_mean(self.get_modulation_matrix())
            # Set the modulation_matrix prior standard deviation to the deformation kernel width.
            self.priors['modulation_matrix'].set_variance_sqrt(self.spatiotemporal_reference_frame.get_kernel_width())

    def __initialize_reference_time_prior(self, initial_time_shift_variance):
        """
        Initialize the reference time prior.
        """
        # If needed (i.e. reference time not frozen), initialize the associated prior.
        if not self.is_frozen['reference_time']:
            # Set the reference_time prior mean as the initial reference_time.
            self.priors['reference_time'].set_mean(np.zeros((1,)) + self.get_reference_time())
            # Check that the reference_time prior variance has been set.
            self.priors['reference_time'].set_variance(initial_time_shift_variance)

    def __initialize_time_shift_variance_prior(self):
        """
        Initialize the time-shift variance prior.
        """
        # If needed (i.e. time-shift variance not frozen), initialize the associated prior.
        if not self.is_frozen['time_shift_variance']:
            # Set the time_shift_variance prior scale to the initial time_shift_variance fixed effect.
            self.priors['time_shift_variance'].scale_scalars.append(self.get_time_shift_variance())
            # Arbitrarily set the time_shift_variance prior dof to 1.
            print('>> The time shift variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
            self.priors['time_shift_variance'].degrees_of_freedom.append(1.0)

    def __initialize_acceleration_variance_prior(self):
        """
        Initialize the acceleration variance prior.
        """
        # If needed (i.e. acceleration variance not frozen), initialize the associated prior.
        if not self.is_frozen['acceleration_variance']:
            # Set the acceleration_variance prior scale to the initial acceleration_variance fixed effect.
            self.priors['acceleration_variance'].scale_scalars.append(self.get_acceleration_variance())
            # Arbitrarily set the acceleration_variance prior dof to 1.
            print('>> The acceleration variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
            self.priors['acceleration_variance'].degrees_of_freedom.append(1.0)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)
        self.spatiotemporal_reference_frame_is_modified = True

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp
        self.spatiotemporal_reference_frame_is_modified = True

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom):
        self.fixed_effects['momenta'] = mom
        self.spatiotemporal_reference_frame_is_modified = True

    # Modulation matrix ------------------------------------------------------------------------------------------------
    def get_modulation_matrix(self):
        return self.fixed_effects['modulation_matrix']

    def set_modulation_matrix(self, mm):
        self.fixed_effects['modulation_matrix'] = mm
        self.spatiotemporal_reference_frame_is_modified = True

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_reference_time(self):
        return self.fixed_effects['reference_time']

    def set_reference_time(self, rt):
        self.fixed_effects['reference_time'] = np.float64(rt)
        self.individual_random_effects['onset_age'].set_mean(np.zeros((1,)) + rt)
        self.spatiotemporal_reference_frame_is_modified = True

    # Time-shift variance ----------------------------------------------------------------------------------------------
    def get_time_shift_variance(self):
        return self.fixed_effects['time_shift_variance']

    def set_time_shift_variance(self, tsv):
        self.fixed_effects['time_shift_variance'] = np.float64(tsv)
        self.individual_random_effects['onset_age'].set_variance(tsv)

    # Log-acceleration variance ----------------------------------------------------------------------------------------
    def get_acceleration_variance(self):
        return self.fixed_effects['acceleration_variance']

    def set_acceleration_variance(self, lav):
        self.fixed_effects['acceleration_variance'] = np.float64(lav)
        self.individual_random_effects['acceleration'].set_variance(lav)

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Class 2 fixed effects --------------------------------------------------------------------------------------------
    def get_fixed_effects(self, mode='class2'):
        out = {}

        if mode == 'class2':
            if not self.is_frozen['template_data']:
                for key, value in self.fixed_effects['template_data'].items():
                    out[key] = value
            if not self.is_frozen['control_points']:
                out['control_points'] = self.fixed_effects['control_points']
            if not self.is_frozen['momenta']:
                out['momenta'] = self.fixed_effects['momenta']
            if not self.is_frozen['modulation_matrix']:
                out['modulation_matrix'] = self.fixed_effects['modulation_matrix']

        elif mode == 'all':
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
            out['control_points'] = self.fixed_effects['control_points']
            out['momenta'] = self.fixed_effects['momenta']
            out['modulation_matrix'] = self.fixed_effects['modulation_matrix']
            out['reference_time'] = self.fixed_effects['reference_time']
            out['time_shift_variance'] = self.fixed_effects['time_shift_variance']
            out['acceleration_variance'] = self.fixed_effects['acceleration_variance']
            out['noise_variance'] = self.fixed_effects['noise_variance']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen['template_data']:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.is_frozen['control_points']: self.set_control_points(fixed_effects['control_points'])
        if not self.is_frozen['momenta']: self.set_momenta(fixed_effects['momenta'])
        if not self.is_frozen['modulation_matrix']: self.set_modulation_matrix(fixed_effects['modulation_matrix'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False,
                               modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.
        Start by updating the class 1 fixed effects.

        :param dataset: LongitudinalDataset instance
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, momenta, modulation_matrix = self._fixed_effects_to_torch_tensors(
            with_grad)
        sources, onset_ages, accelerations = self._individual_RER_to_torch_tensors(individual_RER,
                                                                                   with_grad and mode == 'complete')

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        # Compute residuals.
        absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations)
        self._update_spatiotemporal_reference_frame(
            template_points, control_points, momenta, modulation_matrix, tmin, tmax,
            modified_individual_RER=modified_individual_RER)  # Problem if with_grad ?
        residuals = self._compute_residuals(dataset, template_data, absolute_times, sources, with_grad=with_grad)

        # Update the fixed effects only if the user asked for the complete log likelihood.
        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Compute the attachment, with the updated noise variance parameter in the 'complete' mode.
        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)

        # Compute the regularity terms according to the mode.
        regularity = 0.0
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity(sources, onset_ages, accelerations)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, control_points, momenta,
                                                                 modulation_matrix)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = attachment + regularity
            total.backward()

            gradient = {}
            # Template data.
            if not self.is_frozen['template_data']:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad
                # for key, value in template_data.items():
                #     gradient[key] = value.grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(
                        template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                        gradient['landmark_points'].detach())

            # Other gradients.
            if not self.is_frozen['control_points']: gradient['control_points'] = control_points.grad
            if not self.is_frozen['momenta']: gradient['momenta'] = momenta.grad
            if not self.is_frozen['modulation_matrix']: gradient['modulation_matrix'] = modulation_matrix.grad

            if mode == 'complete':
                gradient['sources'] = sources.grad
                gradient['onset_age'] = onset_ages.grad
                gradient['acceleration'] = accelerations.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.detach().cpu().numpy() for key, value in gradient.items()}

            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.detach().cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()
            elif mode == 'model':
                return attachments.detach().cpu().numpy()

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None, model_terms=None):
        """
        Compute the model sufficient statistics.
        """
        sufficient_statistics = {}

        # First statistical moment of the onset ages.
        if (not self.is_frozen['reference_time']) or (not self.is_frozen['time_shift_variance']):
            onset_ages = individual_RER['onset_age']
            sufficient_statistics['S1'] = np.sum(onset_ages)

        # Second statistical moment of the onset ages.
        if not self.is_frozen['time_shift_variance']:
            sufficient_statistics['S2'] = np.sum(onset_ages ** 2)

        # Second statistical moment of the accelerations.
        if not self.is_frozen['acceleration_variance']:
            accelerations = individual_RER['acceleration']
            sufficient_statistics['S3'] = np.sum((accelerations - 1.0) ** 2)

        # Second statistical moment of the residuals (most costy part).
        if not self.is_frozen['noise_variance']:
            sufficient_statistics['S4'] = np.zeros((self.number_of_objects,))

            # Trick to save useless computations. Could be extended to work in the multi-object case as well ...
            if model_terms is not None and self.number_of_objects == 1:
                sufficient_statistics['S4'][0] += - 2 * np.sum(model_terms) * self.get_noise_variance()
                return sufficient_statistics

            # Standard case.
            if residuals is None:
                template_data, template_points, control_points, momenta, modulation_matrix = self._fixed_effects_to_torch_tensors(
                    False)
                sources, onset_ages, accelerations = self._individual_RER_to_torch_tensors(individual_RER, False)
                absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations)
                self._update_spatiotemporal_reference_frame(template_points, control_points, momenta, modulation_matrix,
                                                            tmin, tmax)
                residuals = self._compute_residuals(dataset, template_data, absolute_times, sources, with_grad=False)

            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    for k in range(self.number_of_objects):
                        sufficient_statistics['S4'][k] += residuals[i][j][k].detach().cpu().numpy()

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        total_number_of_observations = dataset.total_number_of_observations

        # Intricate update of the reference time and the time-shift variance -------------------------------------------
        if (not self.is_frozen['reference_time']) and (not self.is_frozen['time_shift_variance']):
            reftime_prior_mean = self.priors['reference_time'].mean[0]
            reftime_prior_variance = self.priors['reference_time'].variance_sqrt ** 2
            tshiftvar_prior_scale = self.priors['time_shift_variance'].scale_scalars[0]
            tshiftvar_prior_dof = self.priors['time_shift_variance'].degrees_of_freedom[0]

            reftime_old, reftime_new = self.get_reference_time(), self.get_reference_time()
            tshiftvar_old, tshiftvar_new = self.get_time_shift_variance(), self.get_time_shift_variance()

            max_number_of_iterations = 100
            convergence_tolerance = 1e-5
            maximum_difference = 0.0

            for iteration in range(max_number_of_iterations):
                reftime_new = (reftime_prior_variance * sufficient_statistics['S1']
                               + tshiftvar_new * reftime_prior_mean) \
                              / (number_of_subjects * reftime_prior_variance + tshiftvar_new)
                tshiftvar_new = (sufficient_statistics['S2'] - 2 * reftime_new * sufficient_statistics['S1']
                                 + number_of_subjects * reftime_new ** 2
                                 + tshiftvar_prior_dof * tshiftvar_prior_scale) \
                                / (number_of_subjects + tshiftvar_prior_dof)

                maximum_difference = max(math.fabs(reftime_new - reftime_old), math.fabs(tshiftvar_new - tshiftvar_old))
                if maximum_difference < convergence_tolerance:
                    break
                else:
                    reftime_old = reftime_new
                    tshiftvar_old = tshiftvar_new

            if iteration == max_number_of_iterations:
                msg = 'In longitudinal_atlas.update_fixed_effects, the intricate update of the reference time and ' \
                      'time-shift variance does not satisfy the tolerance threshold. Maximum difference = ' \
                      + str(maximum_difference) + ' > tolerance = ' + str(convergence_tolerance)
                warnings.warn(msg)

            self.set_reference_time(reftime_new)
            self.set_time_shift_variance(tshiftvar_new)

        elif not self.is_frozen['reference_time']:
            reftime_prior_mean = self.priors['reference_time'].mean[0]
            reftime_prior_variance = self.priors['reference_time'].variance_sqrt ** 2
            tshiftvar = self.get_time_shift_variance()
            reference_time = (reftime_prior_variance * sufficient_statistics['S1'] + tshiftvar * reftime_prior_mean) \
                             / (number_of_subjects * reftime_prior_variance + tshiftvar)
            self.set_reference_time(reference_time)

        elif not self.is_frozen['time_shift_variance']:
            tshiftvar_prior_scale = self.priors['time_shift_variance'].scale_scalars[0]
            tshiftvar_prior_dof = self.priors['time_shift_variance'].degrees_of_freedom[0]
            reftime = self.get_reference_time()
            time_shift_variance = (sufficient_statistics['S2'] - 2 * reftime * sufficient_statistics['S1']
                                   + number_of_subjects * reftime ** 2 + tshiftvar_prior_dof * tshiftvar_prior_scale) \
                                  / (number_of_subjects + tshiftvar_prior_dof)
            self.set_time_shift_variance(time_shift_variance)

        # Update of the acceleration variance: fixed-point algorithm ---------------------------------------------------
        if not self.is_frozen['acceleration_variance']:
            prior_scale = self.priors['acceleration_variance'].scale_scalars[0]
            prior_dof = self.priors['acceleration_variance'].degrees_of_freedom[0]

            max_number_of_iterations = 100
            convergence_tolerance = 1e-5

            std_old, std_new = math.sqrt(self.get_acceleration_variance()), math.sqrt(self.get_acceleration_variance())
            for iteration in range(max_number_of_iterations):
                phi = norm.pdf(- 1.0 / std_old)
                Phi = norm.cdf(- 1.0 / std_old)
                std_new = 1.0 / math.sqrt((number_of_subjects * (1 - (phi / std_old) / (1 - Phi)) + prior_dof) /
                                          (sufficient_statistics['S3'] + prior_dof * prior_scale))
                difference = math.fabs(std_new - std_old)
                if difference < convergence_tolerance:
                    break
                else:
                    std_old = std_new
                if iteration == max_number_of_iterations:
                    msg = 'When updating the acceleration std parameter from the empirical std, the fixed-point ' \
                          'algorithm did not satisfy the tolerance threshold within the allowed ' \
                          + str(max_number_of_iterations) + 'iterations. Difference = ' \
                          + str(difference) + ' > tolerance = ' + str(convergence_tolerance)
                    warnings.warn(msg)

            self.set_acceleration_variance(std_new ** 2)

        # Update of the residual noise variance ------------------------------------------------------------------------
        if not self.is_frozen['noise_variance']:
            noise_variance = np.zeros((self.number_of_objects,))
            prior_scale_scalars = self.priors['noise_variance'].scale_scalars
            prior_dofs = self.priors['noise_variance'].degrees_of_freedom
            for k in range(self.number_of_objects):
                noise_variance[k] = \
                    (sufficient_statistics['S4'][k] + prior_scale_scalars[k] * prior_dofs[k]) \
                    / float(total_number_of_observations * self.objects_noise_dimension[k] + prior_dofs[k])
            self.set_noise_variance(noise_variance)

    def preoptimize(self, individual_RER):
        # Removes the mean of the accelerations.
        factor = 0.25
        expected_mean_acceleration = self.individual_random_effects['acceleration'].get_expected_mean()
        mean_acceleration = np.mean(individual_RER['acceleration'])
        self.set_momenta(self.get_momenta() * ((1 - factor) + factor * mean_acceleration / expected_mean_acceleration))

        # Remove the mean of the sources.
        mean_sources = torch.from_numpy(np.mean(individual_RER['sources'], axis=0)).type(self.tensor_scalar_type)
        (template_data, template_points,
         control_points, _, modulation_matrix) = self._fixed_effects_to_torch_tensors(False)
        space_shift = 0.5 * torch.mm(modulation_matrix, mean_sources.unsqueeze(1)).view(control_points.size())
        self.spatiotemporal_reference_frame.exponential.set_initial_template_points(template_points)
        self.spatiotemporal_reference_frame.exponential.set_initial_control_points(control_points)
        self.spatiotemporal_reference_frame.exponential.set_initial_momenta(space_shift)
        self.spatiotemporal_reference_frame.exponential.update()
        deformed_control_points = self.spatiotemporal_reference_frame.exponential.control_points_t[-1]
        self.set_control_points(deformed_control_points.detach().cpu().numpy())
        deformed_points = self.spatiotemporal_reference_frame.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        self.set_template_data({key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        # Remove the standard deviation of the sources.
        std_sources = np.std(individual_RER['sources'], axis=0)
        self.set_modulation_matrix(self.get_modulation_matrix() * 0.5 * (1.0 + std_sources))

    ####################################################################################################################
    ### Private key methods:
    ####################################################################################################################

    def _compute_attachment(self, residuals):
        """
        Fully torch.
        """
        return torch.sum(self._compute_individual_attachments(residuals))

    def _compute_individual_attachments(self, residuals):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        attachments = Variable(torch.zeros((number_of_subjects,)).type(self.tensor_scalar_type),
                               requires_grad=False)
        for i in range(number_of_subjects):
            attachment_i = 0.0
            for j in range(len(residuals[i])):
                attachment_i -= 0.5 * torch.sum(residuals[i][j] / Variable(
                    torch.from_numpy(self.fixed_effects['noise_variance']).type(self.tensor_scalar_type),
                    requires_grad=False))
            attachments[i] = attachment_i
        return attachments

    def _compute_random_effects_regularity(self, sources, onset_ages, accelerations):
        """
        Fully torch.
        """
        number_of_subjects = onset_ages.shape[0]
        regularity = 0.0

        # Sources random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['sources'].compute_log_likelihood_torch(
                sources[i], self.tensor_scalar_type)

        # Onset age random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['onset_age'].compute_log_likelihood_torch(
                onset_ages[i], self.tensor_scalar_type)

        # Acceleration random effect.
        for i in range(number_of_subjects):
            regularity += \
                self.individual_random_effects['acceleration'].compute_log_likelihood_torch(
                    accelerations[i], self.tensor_scalar_type)

        # Noise random effect (if not frozen).
        if not self.is_frozen['noise_variance']:
            for k in range(self.number_of_objects):
                regularity -= 0.5 * self.objects_noise_dimension[k] * number_of_subjects * math.log(
                    self.fixed_effects['noise_variance'][k])

        return regularity

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Reference time prior (if not frozen).
        if not self.is_frozen['reference_time']:
            regularity += self.priors['reference_time'].compute_log_likelihood(self.fixed_effects['reference_time'])

        # Time-shift variance prior (if not frozen).
        if not self.is_frozen['time_shift_variance']:
            regularity += \
                self.priors['time_shift_variance'].compute_log_likelihood(self.fixed_effects['time_shift_variance'])

        # Log-acceleration variance prior (if not frozen).
        if not self.is_frozen['acceleration_variance']:
            regularity += self.priors['acceleration_variance'].compute_log_likelihood(
                self.fixed_effects['acceleration_variance'])

        # Noise variance prior (if not frozen).
        if not self.is_frozen['noise_variance']:
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, template_data, control_points, momenta, modulation_matrix):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Prior on template_data fixed effects (if not frozen).
        if not self.is_frozen['template_data']:
            for key, value in template_data.items():
                regularity += self.priors['template_data'][key].compute_log_likelihood_torch(
                    value, self.tensor_scalar_type)

        # Prior on control_points fixed effects (if not frozen).
        if not self.is_frozen['control_points']:
            regularity += self.priors['control_points'].compute_log_likelihood_torch(
                control_points, self.tensor_scalar_type)

        # Prior on momenta fixed effects (if not frozen).
        if not self.is_frozen['momenta']:
            regularity += self.priors['momenta'].compute_log_likelihood_torch(momenta, self.tensor_scalar_type)

        # Prior on modulation_matrix fixed effects (if not frozen).
        if not self.is_frozen['modulation_matrix']:
            regularity += self.priors['modulation_matrix'].compute_log_likelihood_torch(
                modulation_matrix, self.tensor_scalar_type)

        return regularity

    def clear_memory(self):
        """
        Called by the srw_mhwg_sampler if a ValueError is detected. Useful if the geodesic had been extended before
        a problematic parallel transport.
        """
        self.spatiotemporal_reference_frame_is_modified = True

    def _update_spatiotemporal_reference_frame(self, template_points, control_points, momenta, modulation_matrix,
                                               tmin, tmax, modified_individual_RER='all'):
        """
        Tries to optimize the computations, by avoiding repetitions of shooting / flowing / parallel transporting.
        If modified_individual_RER is None or that self.spatiotemporal_reference_frame_is_modified is True,
        no particular optimization is carried.
        In the opposite case, the spatiotemporal reference frame will be more subtly updated.
        """

        # print('self.spatiotemporal_reference_frame_is_modified', self.spatiotemporal_reference_frame_is_modified)
        # t1 = time.time()

        if self.spatiotemporal_reference_frame_is_modified:
            t0 = self.get_reference_time()
            self.spatiotemporal_reference_frame.set_template_points_t0(template_points)
            self.spatiotemporal_reference_frame.set_control_points_t0(control_points)
            self.spatiotemporal_reference_frame.set_momenta_t0(momenta)
            self.spatiotemporal_reference_frame.set_modulation_matrix_t0(modulation_matrix)
            self.spatiotemporal_reference_frame.set_t0(t0)
            self.spatiotemporal_reference_frame.set_tmin(tmin)
            self.spatiotemporal_reference_frame.set_tmax(tmax)
            self.spatiotemporal_reference_frame.update()

        else:
            if modified_individual_RER in ['onset_age', 'acceleration', 'all']:
                self.spatiotemporal_reference_frame.set_tmin(tmin, optimize=True)
                self.spatiotemporal_reference_frame.set_tmax(tmax, optimize=True)
                self.spatiotemporal_reference_frame.update()

            elif not modified_individual_RER == 'sources':
                raise RuntimeError('Unexpected modified_individual_RER: "' + str(modified_individual_RER) + '"')

        self.spatiotemporal_reference_frame_is_modified = False

        # t2 = time.time()
        # print('>> Total time           : %.3f seconds' % (t2 - t1))

    def _compute_residuals(self, dataset, template_data, absolute_times, sources, with_grad=True):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        targets = dataset.deformable_objects
        residuals = []  # List of list of torch 1D tensors. Individuals, time-points, object.

        if self.number_of_threads > 1 and not with_grad:

            # t1 = time.time()

            # Set arguments.
            args = []
            for i in range(len(targets)):
                residuals_i = []
                for j, (absolute_time, target) in enumerate(zip(absolute_times[i], targets[i])):
                    residuals_i.append(None)
                    args.append(
                        (i, j, Settings().serialize(),
                         self.spatiotemporal_reference_frame.get_template_points_exponential(absolute_time, sources[i]),
                         {key: value.clone() for key, value in template_data.items()}, self.template.clone(),
                         target, deepcopy(self.multi_object_attachment)))
                residuals.append(residuals_i)

            # Perform parallelized computations.
            # print('Perform parallelized computations.')
            with ThreadPoolExecutor(max_workers=self.number_of_threads) as pool:
                results = pool.map(compute_exponential_and_attachment, args)

            # Gather results.
            for result in results:
                i, j, residual = result
                residuals[i][j] = residual

                # t2 = time.time()
                # print('>> Total time           : %.3f seconds' % (t2 - t1))

        else:
            # t1 = time.time()

            # print('Perform sequential computations.')
            for i in range(len(targets)):
                residuals_i = []
                for j, (absolute_time, target) in enumerate(zip(absolute_times[i], targets[i])):
                    deformed_points = self.spatiotemporal_reference_frame.get_template_points(absolute_time, sources[i])
                    deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                    residuals_i.append(
                        self.multi_object_attachment.compute_distances(deformed_data, self.template, target))
                residuals.append(residuals_i)

                # t2 = time.time()
                # print('>> Total time           : %.3f seconds' % (t2 - t1))

        return residuals

    def _compute_absolute_times(self, times, onset_ages, accelerations):
        """
        Fully torch.
        """
        acceleration_std = math.sqrt(self.get_acceleration_variance())
        if acceleration_std > 1.0 and np.max(accelerations.data.cpu().numpy()) - 1.0 > 10.0 * acceleration_std:
            raise ValueError('Absurd numerical value for the acceleration factor: %.2f. Exception raised.'
                             'For reference, the acceleration std is %.2f.'
                             % (np.max(accelerations.data.cpu().numpy()), acceleration_std))

        reference_time = self.get_reference_time()
        reference_time_torch = torch.from_numpy(np.array(reference_time)).type(self.tensor_scalar_type)
        clamped_accelerations = torch.clamp(accelerations, 0.0)

        absolute_times = []
        for i in range(len(times)):
            absolute_times_i = []
            for j in range(len(times[i])):
                t_ij = torch.from_numpy(np.array(times[i][j])).type(self.tensor_scalar_type)
                absolute_times_i.append(clamped_accelerations[i] * (t_ij - onset_ages[i]) + reference_time_torch)
            absolute_times.append(absolute_times_i)

        tmin = min([subject_times[0].detach().cpu().numpy() for subject_times in absolute_times] + [reference_time])
        tmax = max([subject_times[-1].detach().cpu().numpy() for subject_times in absolute_times] + [reference_time])

        return absolute_times, tmin, tmax

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the input fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
                                       requires_grad=(not self.is_frozen['template_data'] and with_grad))
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
                                         requires_grad=(not self.is_frozen['template_data'] and with_grad))
                           for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            control_points = template_data
        else:
            control_points = self.fixed_effects['control_points']
            # control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
            #                           requires_grad=((not self.is_frozen['control_points']) and with_grad))
            control_points = Variable(
                torch.from_numpy(control_points).type(self.tensor_scalar_type),
                requires_grad=(not self.is_frozen['control_points'] and with_grad))

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(self.tensor_scalar_type),
                           requires_grad=((not self.is_frozen['momenta']) and with_grad))

        # Modulation matrix.
        modulation_matrix = self.fixed_effects['modulation_matrix']
        modulation_matrix = Variable(torch.from_numpy(modulation_matrix).type(self.tensor_scalar_type),
                                     requires_grad=((not self.is_frozen['modulation_matrix']) and with_grad))

        return template_data, template_points, control_points, momenta, modulation_matrix

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Sources.
        sources = individual_RER['sources']
        sources = Variable(torch.from_numpy(sources).type(self.tensor_scalar_type), requires_grad=with_grad)
        # Onset ages.
        onset_ages = individual_RER['onset_age']
        onset_ages = Variable(torch.from_numpy(onset_ages).type(self.tensor_scalar_type),
                              requires_grad=with_grad)
        # Accelerations.
        accelerations = individual_RER['acceleration']
        accelerations = Variable(torch.from_numpy(accelerations).type(self.tensor_scalar_type),
                                 requires_grad=with_grad)
        return sources, onset_ages, accelerations

    ####################################################################################################################
    ### Error handling methods:
    ####################################################################################################################

    def adapt_to_error(self, error):
        if error[:64] == 'Absurd required renormalization factor during parallel transport':
            self._augment_discretization()
        else:
            raise RuntimeError('Unknown response to the error: "%s"' % error)

    def _augment_discretization(self):
        current_concentration = self.spatiotemporal_reference_frame.get_concentration_of_time_points()
        momenta_factor = current_concentration / float(current_concentration + 1)
        print('Incrementing the concentration of time-points from %d to %d, and multiplying the momenta '
              'by a factor %.3f.' % (current_concentration, current_concentration + 1, momenta_factor))
        self.spatiotemporal_reference_frame.set_concentration_of_time_points(current_concentration + 1)
        self.set_momenta(momenta_factor * self.get_momenta())

    ####################################################################################################################
    ### Printing and writing methods:
    ####################################################################################################################

    def print(self, individual_RER):
        print('>> Model parameters:')

        # Noise variance.
        msg = '\t\t noise_std        ='
        noise_variance = self.get_noise_variance()
        for k, object_name in enumerate(self.objects_name):
            msg += '\t%.4f\t[ %s ]\t ; ' % (math.sqrt(noise_variance[k]), object_name)
        print(msg[:-4])

        # Reference time, time-shift std, acceleration std.
        print('\t\t reference_time   =\t%.3f' % self.get_reference_time())
        print('\t\t time_shift_std   =\t%.3f' % math.sqrt(self.get_time_shift_variance()))
        print('\t\t acceleration_std =\t%.3f' % math.sqrt(self.get_acceleration_variance()))

        # Empirical distributions of the individual parameters.
        print('>> Random effect empirical distributions:')
        print('\t\t onset_ages       =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['onset_age']), np.std(individual_RER['onset_age'])))
        print('\t\t accelerations    =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['acceleration']), np.std(individual_RER['acceleration'])))
        print('\t\t sources          =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(individual_RER['sources']), np.std(individual_RER['sources'])))

        # Spatiotemporal reference frame length.
        print('>> Spatiotemporal reference frame length: %.2f.' %
              (self.spatiotemporal_reference_frame.get_tmax() - self.spatiotemporal_reference_frame.get_tmin()))

    def write(self, dataset, population_RER, individual_RER, output_dir, update_fixed_effects=True,
              write_residuals=True):
        self._clean_output_directory(output_dir)

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=(update_fixed_effects or write_residuals))

        # Optionally update the fixed effects.
        if update_fixed_effects:
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Write residuals.
        if write_residuals:
            residuals_list = [[[residuals_i_j_k.detach().cpu().numpy() for residuals_i_j_k in residuals_i_j]
                               for residuals_i_j in residuals_i] for residuals_i in residuals]
            write_3D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(individual_RER, output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, template_points, control_points, momenta, modulation_matrix \
            = self._fixed_effects_to_torch_tensors(False)
        sources, onset_ages, accelerations = self._individual_RER_to_torch_tensors(individual_RER, False)
        targets = dataset.deformable_objects
        absolute_times, tmin, tmax = self._compute_absolute_times(dataset.times, onset_ages, accelerations)

        # Deform -------------------------------------------------------------------------------------------------------
        self._update_spatiotemporal_reference_frame(template_points, control_points, momenta, modulation_matrix,
                                                    tmin, tmax)

        # Write --------------------------------------------------------------------------------------------------------
        self.spatiotemporal_reference_frame.write(self.name, self.objects_name, self.objects_name_extension,
                                                  self.template, template_data, output_dir)

        # Write reconstructions and compute residuals ------------------------------------------------------------------
        residuals = []  # List of list of torch 1D tensors. Individuals, time-points, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            residuals_i = []
            for j, (time, absolute_time) in enumerate(zip(dataset.times[i], absolute_times[i])):
                deformed_points = self.spatiotemporal_reference_frame.get_template_points(absolute_time, sources[i])
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)

                if compute_residuals:
                    residuals_i.append(
                        self.multi_object_attachment.compute_distances(deformed_data, self.template, targets[i][j]))

                names = []
                for k, (object_name, object_extension) \
                        in enumerate(zip(self.objects_name, self.objects_name_extension)):
                    name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id \
                           + '__tp_' + str(j) + ('__age_%.2f' % time) + object_extension
                    names.append(name)
                self.template.write(output_dir, names,
                                    {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            residuals.append(residuals_i)

        return residuals

    def _write_model_parameters(self, individual_RER, output_dir):
        # Fixed effects ------------------------------------------------------------------------------------------------
        # Template.
        template_names = []
        for k in range(len(self.objects_name)):
            aux = self.name + '__EstimatedParameters__Template_' + self.objects_name[k] + '__tp_' \
                  + str(self.spatiotemporal_reference_frame.geodesic.backward_exponential.number_of_time_points - 1) \
                  + ('__age_%.2f' % self.get_reference_time()) + self.objects_name_extension[k]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Other class 1 fixed effects ----------------------------------------------------------------------------------
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        write_2D_array(self.get_modulation_matrix(), output_dir,
                       self.name + "__EstimatedParameters__ModulationMatrix.txt")

        # Class 2 fixed effects ----------------------------------------------------------------------------------------
        write_2D_array(np.zeros((1,)) + self.get_reference_time(), output_dir,
                       self.name + "__EstimatedParameters__ReferenceTime.txt")
        write_2D_array(np.zeros((1,)) + math.sqrt(self.get_time_shift_variance()), output_dir,
                       self.name + "__EstimatedParameters__TimeShiftStd.txt")
        write_2D_array(np.zeros((1,)) + math.sqrt(self.get_acceleration_variance()), output_dir,
                       self.name + "__EstimatedParameters__AccelerationStd.txt")
        write_2D_array(np.sqrt(self.get_noise_variance()), output_dir,
                       self.name + "__EstimatedParameters__NoiseStd.txt")

        # Random effects realizations ----------------------------------------------------------------------------------
        # Sources.
        write_2D_array(individual_RER['sources'], output_dir, self.name + "__EstimatedParameters__Sources.txt")
        # Onset age.
        write_2D_array(individual_RER['onset_age'], output_dir, self.name + "__EstimatedParameters__OnsetAges.txt")
        # Log-acceleration.
        write_2D_array(individual_RER['acceleration'], output_dir,
                       self.name + "__EstimatedParameters__Accelerations.txt")

    def _clean_output_directory(self, output_dir):
        files_to_delete = glob.glob(output_dir + '/*')
        for file in files_to_delete:
            if not os.path.isdir(file) and (len(file) > 1 and not file[-2:] == '.p'):
                os.remove(file)
