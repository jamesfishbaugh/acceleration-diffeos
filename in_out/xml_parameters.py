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
        self.use_rk2_for_shoot = default.use_rk2_for_shoot
        self.use_rk2_for_flow = default.use_rk2_for_flow
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.initial_cp_spacing = default.initial_cp_spacing
        self.dimension = default.dimension

        self.dataset_filenames = default.dataset_filenames
        self.visit_ages = default.visit_ages
        self.subject_ids = default.subject_ids

        self.optimization_method_type = default.optimization_method_type
        self.max_iterations = default.max_iterations
        self.max_line_search_iterations = default.max_line_search_iterations
        self.save_every_n_iters = default.save_every_n_iters
        self.print_every_n_iters = default.print_every_n_iters
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

        self.initial_control_points = default.initial_control_points

        self.initial_velocity = default.initial_velocity
        self.impulse_t = default.impulse_t
        
        self.data_weight = default.data_weight
        self.regularity_weight = default.regularity_weight

        self.estimate_initial_velocity = default.estimate_initial_velocity
        self.initial_velocity_weight = default.initial_velocity_weight

        self.normalize_image_intensity = default.normalize_image_intensity

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

            elif model_xml_level1.tag.lower() == 'initial-velocity':
                self.initial_velocity = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))
                    
            elif model_xml_level1.tag.lower() == 'impulse-t':
                self.impulse_t = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

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
                            elif model_xml_level3.tag.lower() == 'data-weight':
                                template_object['data_weight'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'filename':
                                template_object['filename'] = os.path.normpath(os.path.join(os.path.dirname(model_xml_path), model_xml_level3.text))
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
                    elif model_xml_level2.tag.lower() == 'regularity-weight':
                        self.regularity_weight = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'initial-velocity-weight':
                        self.initial_velocity_weight = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmin':
                        self.tmin = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmax':
                        self.tmax = float(model_xml_level2.text)
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

            self.dataset_filenames = dataset_filenames
            self.visit_ages = visit_ages
            self.subject_ids = subject_ids

    # Read the parameters from the optimization_parameters xml.
    def _read_optimization_parameters_xml(self, optimization_parameters_xml_path):

        optimization_parameters_xml_level0 = et.parse(optimization_parameters_xml_path).getroot()

        for optimization_parameters_xml_level1 in optimization_parameters_xml_level0:
            if optimization_parameters_xml_level1.tag.lower() == 'optimization-method-type':
                self.optimization_method_type = optimization_parameters_xml_level1.text.lower()
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
            elif optimization_parameters_xml_level1.tag.lower() == 'estimate-initial-velocity':
                self.estimate_initial_velocity = self._on_off_to_bool(optimization_parameters_xml_level1.text)

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
        template_object['filename'] = 'undefined'
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")
