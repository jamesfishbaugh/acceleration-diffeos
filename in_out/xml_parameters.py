import logging
import math
import os
import warnings
import xml.etree.ElementTree as et
from sys import platform

import torch

from core import default

logger = logging.getLogger(__name__)


class XmlParameters:
    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.tensor_scalar_type = default.tensor_scalar_type
        self.tensor_integer_type = default.tensor_scalar_type

        self.model_type = default.model_type
        self.template_specifications = default.template_specifications
        self.deformation_kernel_width = 0
        self.deformation_kernel_type = 'torch'
        self.deformation_kernel_device = default.deformation_kernel_device
        self.number_of_time_points = default.number_of_time_points
        self.concentration_of_time_points = default.concentration_of_time_points
        self.number_of_sources = default.number_of_sources
        self.use_rk2_for_shoot = default.use_rk2_for_shoot
        self.use_rk2_for_flow = default.use_rk2_for_flow
        self.t0 = None
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.initial_cp_spacing = default.initial_cp_spacing
        self.dimension = default.dimension
        self.covariance_momenta_prior_normalized_dof = default.covariance_momenta_prior_normalized_dof

        self.dataset_filenames = default.dataset_filenames
        self.visit_ages = default.visit_ages
        self.subject_ids = default.subject_ids

        self.optimization_method_type = default.optimization_method_type
        self.optimized_log_likelihood = default.optimized_log_likelihood
        self.number_of_threads = default.number_of_threads
        self.max_iterations = default.max_iterations
        self.max_line_search_iterations = default.max_line_search_iterations
        self.save_every_n_iters = default.save_every_n_iters
        self.print_every_n_iters = default.print_every_n_iters
        self.sample_every_n_mcmc_iters = default.sample_every_n_mcmc_iters
        self.use_sobolev_gradient = default.use_sobolev_gradient
        self.sobolev_kernel_width_ratio = default.sobolev_kernel_width_ratio
        self.smoothing_kernel_width = default.smoothing_kernel_width
        self.initial_step_size = default.initial_step_size
        self.line_search_shrink = default.line_search_shrink
        self.line_search_expand = default.line_search_expand
        self.convergence_tolerance = default.convergence_tolerance
        self.memory_length = default.memory_length
        self.scale_initial_step_size = default.scale_initial_step_size
        self.downsampling_factor = default.downsampling_factor

        self.dense_mode = default.dense_mode

        self.use_cuda = default.use_cuda
        self._cuda_is_used = default._cuda_is_used  # true if at least one operation will use CUDA.
        self._keops_is_used = default._keops_is_used  # true if at least one keops kernel operation will take place.

        self.state_file = None
        self.load_state_file = False

        self.freeze_template = default.freeze_template
        self.freeze_control_points = default.freeze_control_points
        self.freeze_momenta = default.freeze_momenta
        self.freeze_modulation_matrix = default.freeze_modulation_matrix
        self.freeze_reference_time = default.freeze_reference_time
        self.freeze_time_shift_variance = default.freeze_time_shift_variance
        self.freeze_acceleration_variance = default.freeze_acceleration_variance
        self.freeze_noise_variance = default.freeze_noise_variance

        self.freeze_translation_vectors = False
        self.freeze_rotation_angles = False
        self.freeze_scaling_ratios = False

        # For metric learning atlas
        self.freeze_metric_parameters = default.freeze_metric_parameters
        self.freeze_p0 = default.freeze_p0
        self.freeze_v0 = default.freeze_v0

        self.initial_control_points = default.initial_control_points
        self.initial_momenta = default.initial_momenta
        
        self.initial_velocity = default.initial_velocity
        self.impulse_t = default.impulse_t
        self.use_intensity_model = False
        
        self.initial_modulation_matrix = default.initial_modulation_matrix
        self.initial_time_shift_variance = default.initial_time_shift_variance
        self.initial_acceleration_mean = default.initial_acceleration_mean
        self.initial_acceleration_variance = default.initial_acceleration_variance
        self.initial_onset_ages = default.initial_onset_ages
        self.initial_accelerations = default.initial_accelerations
        self.initial_sources = default.initial_sources
        self.initial_sources_mean = default.initial_sources_mean
        self.initial_sources_std = default.initial_sources_std

        self.initial_control_points_to_transport = default.initial_control_points_to_transport

        self.momenta_proposal_std = default.momenta_proposal_std
        self.onset_age_proposal_std = default.onset_age_proposal_std
        self.acceleration_proposal_std = default.acceleration_proposal_std
        self.sources_proposal_std = default.sources_proposal_std
        
        self.estimate_initial_velocity = default.estimate_initial_velocity
        self.initial_velocity_weight = default.initial_velocity_weight

        # For scalar inputs:
        self.group_file = default.group_file
        self.observations_file = default.observations_file
        self.timepoints_file = default.timepoints_file
        self.v0 = default.v0
        self.p0 = default.p0
        self.metric_parameters_file = default.metric_parameters_file
        self.interpolation_points_file = default.interpolation_points_file
        self.initial_noise_variance = default.initial_noise_variance
        self.exponential_type = default.exponential_type
        self.number_of_metric_parameters = default.number_of_metric_parameters  # number of parameters in metric learning.
        self.number_of_interpolation_points = default.number_of_interpolation_points
        self.latent_space_dimension = default.latent_space_dimension  # For deep metric learning

        self.normalize_image_intensity = default.normalize_image_intensity
        self.initialization_heuristic = default.initialization_heuristic

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Read the parameters from the three PyDeformetrica input xmls, and some further parameters initialization.
    def read_all_xmls(self, model_xml_path, dataset_xml_path, optimization_parameters_xml_path, output_dir):
        self._read_model_xml(model_xml_path)
        self._read_dataset_xml(dataset_xml_path)
        self._read_optimization_parameters_xml(optimization_parameters_xml_path)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    # Read the parameters from the model xml.
    def _read_model_xml(self, model_xml_path):

        model_xml_level0 = et.parse(model_xml_path).getroot()

        for model_xml_level1 in model_xml_level0:

            if model_xml_level1.tag.lower() == 'model-type':
                self.model_type = model_xml_level1.text.lower()

            elif model_xml_level1.tag.lower() == 'dimension':
                self.dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-control-points':
                self.initial_control_points = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-momenta':
                self.initial_momenta = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))
                    
            elif model_xml_level1.tag.lower() == 'initial-velocity':
                self.initial_velocity = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))
                    
            elif model_xml_level1.tag.lower() == 'impulse-t':
                self.impulse_t = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))
                
            elif model_xml_level1.tag.lower() == 'initial-modulation-matrix':
                self.initial_modulation_matrix = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-time-shift-std':
                self.initial_time_shift_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'initial-acceleration-std':
                self.initial_acceleration_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'initial-acceleration-mean':
                self.initial_acceleration_mean = float(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-onset-ages':
                self.initial_onset_ages = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-accelerations':
                self.initial_accelerations = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources':
                self.initial_sources = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources-mean':
                self.initial_sources_mean = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-sources-std':
                self.initial_sources_std = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-momenta-to-transport':
                self.initial_momenta_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-control-points-to-transport':
                self.initial_control_points_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-noise-std':
                self.initial_noise_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'latent-space-dimension':
                self.latent_space_dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'template':
                for model_xml_level2 in model_xml_level1:

                    if model_xml_level2.tag.lower() == 'dense-mode':
                        self.dense_mode = self._on_off_to_bool(model_xml_level2.text)

                    elif model_xml_level2.tag.lower() == 'object':

                        template_object = self._initialize_template_object_xml_parameters()
                        for model_xml_level3 in model_xml_level2:
                            if model_xml_level3.tag.lower() == 'deformable-object-type':
                                template_object['deformable_object_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'attachment-type':
                                template_object['attachment_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'kernel-width':
                                template_object['kernel_width'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'kernel-type':
                                template_object['kernel_type'] = model_xml_level3.text.lower()
                                if model_xml_level3.text.lower() == 'keops'.lower():
                                    if platform in ['darwin']:
                                        logger.warning(
                                            'The "keops" kernel is unavailable for Mac OS X platforms. '
                                            'Overriding with "torch" kernel. Beware: the memory consumption might '
                                            'explode for high-dimensional data.')
                                        template_object['kernel_type'] = 'torch'
                                    else:
                                        self._keops_is_used = True
                            elif model_xml_level3.tag.lower() == 'kernel-device':
                                template_object['kernel_device'] = model_xml_level3.text
                            elif model_xml_level3.tag.lower() == 'noise-std':
                                template_object['noise_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'filename':
                                template_object['filename'] = os.path.normpath(
                                    os.path.join(os.path.dirname(model_xml_path), model_xml_level3.text))
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-scale-std':
                                template_object['noise_variance_prior_scale_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-normalized-dof':
                                template_object['noise_variance_prior_normalized_dof'] = float(model_xml_level3.text)
                            else:
                                msg = 'Unknown entry while parsing the template > ' + model_xml_level2.attrib['id'] + \
                                      ' object section of the model xml: ' + model_xml_level3.tag
                                warnings.warn(msg)

                        self.template_specifications[model_xml_level2.attrib['id']] = template_object
                    else:
                        msg = 'Unknown entry while parsing the template section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)

            elif model_xml_level1.tag.lower() == 'deformation-parameters':
                for model_xml_level2 in model_xml_level1:
                    if model_xml_level2.tag.lower() == 'kernel-width':
                        self.deformation_kernel_width = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'exponential-type':
                        self.exponential_type = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'kernel-type':
                        self.deformation_kernel_type = model_xml_level2.text.lower()
                        if model_xml_level2.text.lower() == 'keops'.lower():
                            if platform in ['darwin']:
                                logger.warning(
                                    'The "keops" kernel is unavailable for Mac OS X platforms. '
                                    'Overriding with "torch" kernel. Beware: the memory consumption might '
                                    'explode for high-dimensional data.')
                                self.deformation_kernel_type = 'torch'
                            else:
                                self._keops_is_used = True
                    elif model_xml_level2.tag.lower() == 'kernel-device':
                        self.deformation_kernel_device = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'number-of-timepoints':
                        self.number_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'number-of-interpolation-points':
                        self.number_of_interpolation_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'concentration-of-timepoints':
                        self.concentration_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'number-of-sources':
                        self.number_of_sources = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 't0':
                        self.t0 = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmin':
                        self.tmin = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmax':
                        self.tmax = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'p0':
                        self.p0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'v0':
                        self.v0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'metric-parameters-file':  # for metric learning
                        self.metric_parameters_file = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'interpolation-points-file':  # for metric learning
                        self.interpolation_points_file = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'covariance-momenta-prior-normalized-dof':
                        self.covariance_momenta_prior_normalized_dof = float(model_xml_level2.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)
            #
            # elif model_xml_level1.tag.lower() == 'use-exp-parallelization':
            #     self.use_exp_parallelization = self._on_off_to_bool(model_xml_level1.text)

            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + model_xml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def _read_dataset_xml(self, dataset_xml_path):
        if dataset_xml_path is not None and dataset_xml_path != 'None':

            dataset_xml_level0 = et.parse(dataset_xml_path).getroot()
            data_set_xml_dirname = os.path.dirname(dataset_xml_path)

            dataset_filenames = []
            visit_ages = []
            subject_ids = []
            for dataset_xml_level1 in dataset_xml_level0:
                if dataset_xml_level1.tag.lower() == 'subject':
                    subject_ids.append(dataset_xml_level1.attrib['id'])

                    subject_filenames = []
                    subject_ages = []
                    for dataset_xml_level2 in dataset_xml_level1:
                        if dataset_xml_level2.tag.lower() == 'visit':

                            visit_filenames = {}
                            for dataset_xml_level3 in dataset_xml_level2:
                                if dataset_xml_level3.tag.lower() == 'filename':
                                    visit_filenames[dataset_xml_level3.attrib['object_id']] = os.path.normpath(
                                        os.path.join(data_set_xml_dirname, dataset_xml_level3.text))
                                elif dataset_xml_level3.tag.lower() == 'age':
                                    subject_ages.append(float(dataset_xml_level3.text))
                            subject_filenames.append(visit_filenames)
                    dataset_filenames.append(subject_filenames)
                    visit_ages.append(subject_ages)

                # For scalar input, following leasp model
                if dataset_xml_level1.tag.lower() == 'group-file':
                    self.group_file = dataset_xml_level1.text

                if dataset_xml_level1.tag.lower() == 'timepoints-file':
                    self.timepoints_file = dataset_xml_level1.text

                if dataset_xml_level1.tag.lower() == 'observations-file':
                    self.observations_file = dataset_xml_level1.text

            self.dataset_filenames = dataset_filenames
            self.visit_ages = visit_ages
            self.subject_ids = subject_ids

    # Read the parameters from the optimization_parameters xml.
    def _read_optimization_parameters_xml(self, optimization_parameters_xml_path):

        print(optimization_parameters_xml_path)

        optimization_parameters_xml_level0 = et.parse(optimization_parameters_xml_path).getroot()

        for optimization_parameters_xml_level1 in optimization_parameters_xml_level0:
            if optimization_parameters_xml_level1.tag.lower() == 'optimization-method-type':
                self.optimization_method_type = optimization_parameters_xml_level1.text.lower()
            elif optimization_parameters_xml_level1.tag.lower() == 'optimized-log-likelihood':
                self.optimized_log_likelihood = optimization_parameters_xml_level1.text.lower()
            elif optimization_parameters_xml_level1.tag.lower() == 'number-of-threads':
                self.number_of_threads = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'max-iterations':
                self.max_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'convergence-tolerance':
                self.convergence_tolerance = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'memory-length':
                self.memory_length = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'downsampling-factor':
                self.downsampling_factor = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'save-every-n-iters':
                self.save_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'print-every-n-iters':
                self.print_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sample-every-n-mcmc-iters':
                self.sample_every_n_mcmc_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-sobolev-gradient':
                self.use_sobolev_gradient = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sobolev-kernel-width-ratio':
                self.sobolev_kernel_width_ratio = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initial-step-size':
                self.initial_step_size = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-template':
                self.freeze_template = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-control-points':
                self.freeze_control_points = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-intensity-model':
                self.use_intensity_model = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-cuda':
                self.use_cuda = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                if self.use_cuda:
                    self._cuda_is_used = True
            elif optimization_parameters_xml_level1.tag.lower() == 'max-line-search-iterations':
                self.max_line_search_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'state-file':
                self.state_file = optimization_parameters_xml_level1.text
            elif optimization_parameters_xml_level1.tag.lower() == 'use-rk2-for-shoot':
                self.use_rk2_for_shoot = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-rk2':
                self.use_rk2_for_shoot = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                self.use_rk2_for_flow = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'momenta-proposal-std':
                self.momenta_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'onset-age-proposal-std':
                self.onset_age_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'acceleration-proposal-std':
                self.acceleration_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sources-proposal-std':
                self.sources_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'scale-initial-step-size':
                self.scale_initial_step_size = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initialization-heuristic':
                self.initialization_heuristic = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-v0':
                self.freeze_v0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-p0':
                self.freeze_p0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-modulation-matrix':
                self.freeze_modulation_matrix = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-reference-time':
                self.freeze_reference_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-time-shift-variance':
                self.freeze_time_shift_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-acceleration-variance':
                self.freeze_acceleration_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-reference-time':
                self.freeze_reference_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-noise-variance':
                self.freeze_noise_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-translation-vectors':
                self.freeze_translation_vectors = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-rotation-angles':
                self.freeze_rotation_angles = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-scaling-ratios':
                self.freeze_scaling_ratios = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'gradient-based-estimator':
                self.gradient_based_estimator = optimization_parameters_xml_level1.text
                
            elif optimization_parameters_xml_level1.tag.lower() == 'estimate-initial-velocity':
                self.estimate_initial_velocity = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initial-velocity-weight':
                self.initial_velocity_weight = float(optimization_parameters_xml_level1.text)
                
            

            else:
                msg = 'Unknown entry while parsing the optimization_parameters xml: ' \
                      + optimization_parameters_xml_level1.tag
                warnings.warn(msg)

    # Default xml parameters for any template object.
    @staticmethod
    def _initialize_template_object_xml_parameters():
        template_object = {}
        template_object['deformable_object_type'] = 'undefined'
        template_object['kernel_type'] = 'undefined'
        template_object['kernel_width'] = 0.0
        template_object['kernel_device'] = default.deformation_kernel_device
        template_object['noise_std'] = -1
        template_object['filename'] = 'undefined'
        template_object['noise_variance_prior_scale_std'] = None
        template_object['noise_variance_prior_normalized_dof'] = 0.01
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")
