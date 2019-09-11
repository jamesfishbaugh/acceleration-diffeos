import logging
import math
import os
import time
from sys import platform

import torch

from core import default
from core.default import logger_format
from core.estimators.gradient_descent import GradientDescent
from core.estimators.scipy_optimize import ScipyOptimize
from core.models.acceleration_regression import AccelerationRegression
from core.models.acceleration_gompertz_regression_v2 import AccelerationGompertzRegressionV2
from launch import compute_acceleration_flow
from core.models.acceleration_gompertz_regression import AccelerationGompertzRegression
from in_out.dataset_functions import create_dataset
from in_out.deformable_object_reader import DeformableObjectReader

logger = logging.getLogger(__name__)


class AccelerationDiffeos:

    ####################################################################################################################
    # Constructor & destructor.
    ####################################################################################################################

    def __init__(self, output_dir=default.output_dir, verbosity='DEBUG'):
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # set logging level
        try:
            log_level = logging.getLevelName(verbosity)
            logging.basicConfig(level=log_level, format=logger_format)
        except ValueError:
            logger.warning('Logging level was not recognized. Using INFO.')
            log_level = logging.INFO

        logger.debug('Using verbosity level: ' + verbosity)
        logging.basicConfig(level=log_level, format=logger_format)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('AccelerationDiffeos.__exit__()')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ####################################################################################################################
    # Main methods.
    ####################################################################################################################

    def estimate_acceleration_regression(self, template_specifications, dataset_specifications,
                                         model_options={}, estimator_options={}, write_output=True):

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization('Regression', template_specifications, model_options,
                                                                                                dataset_specifications, estimator_options)

        # Instantiate dataset
        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_time_series()), "Cannot estimate an acceleration controlled regression from a non-time-series dataset."

        # Instantiate model
        statistical_model = AccelerationRegression(template_specifications, **model_options)

        # Instantiate estimator
        estimator = self.__instantiate_estimator(statistical_model, dataset, self.output_dir, estimator_options, default=ScipyOptimize)

        # Launch
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_acceleration_gompertz_regression(self, template_specifications, dataset_specifications,
                                                  model_options={}, estimator_options={}, write_output=True):

        # Check and completes the input parameters
        template_specifications, model_options, estimator_options = self.further_initialization('Regression', template_specifications, model_options,
                                                                                                dataset_specifications, estimator_options)

        # Instantiate dataset
        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_time_series()), "Cannot estimate an acceleration controlled regression from a non-time-series dataset."

        print("CALLING V2")

        # Instantiate model
        statistical_model = AccelerationGompertzRegressionV2(template_specifications, **model_options)

        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        last_object = target_objects[len(target_objects) - 1]
        the_image = last_object.get_data()
        # statistical_model.set_B(the_image['image_intensities'])

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, self.output_dir, estimator_options, default=ScipyOptimize)

        # Launch.
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def compute_acceleration_flow(self, template_specifications, model_options={}, write_output=True):
        """
        Compute acceleration flow.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, _ = self.further_initialization('AccelerationFlow', template_specifications, model_options)

        # Launch
        #compute_acceleration_flow(template_specifications, output_dir=self.output_dir, **model_options)

    ####################################################################################################################
    # Auxiliary methods.
    ####################################################################################################################

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """
        start_time = time.time()
        logger.info('Started estimator: ' + estimator.name)
        estimator.update()
        end_time = time.time()

        if write_output:
            estimator.write()

        if end_time - start_time > 60 * 60 * 24:
            print('>> Estimation took: %s' %
                  time.strftime("%d days, %H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60 * 60:
            print('>> Estimation took: %s' %
                  time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60:
            print('>> Estimation took: %s' %
                  time.strftime("%M minutes and %S seconds", time.gmtime(end_time - start_time)))
        else:
            print('>> Estimation took: %s' % time.strftime("%S seconds", time.gmtime(end_time - start_time)))

    def __instantiate_estimator(self, statistical_model, dataset, output_dir, estimator_options, default=ScipyOptimize):
        if estimator_options['optimization_method_type'].lower() == 'GradientAscent'.lower():
            estimator = GradientAscent
        elif estimator_options['optimization_method_type'].lower() == 'GradientDescent'.lower():
            estimator = GradientDescent
        elif estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower():
            estimator = ScipyOptimize
        elif estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():
            estimator = McmcSaem
        else:
            estimator = default
        return estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

    def further_initialization(self, model_type, template_specifications, model_options,
                               dataset_specifications=None, estimator_options=None):

        #
        # Consistency checks.
        #
        if dataset_specifications is None or estimator_options is None:
            assert model_type.lower() in ['AccelerationFlow'.lower()], \
                'Only the "acceleration flow" can run without a dataset and an estimator.'

        #
        # Initializes variables that will be checked.
        #
        if 'dimension' not in model_options:
            model_options['dimension'] = default.dimension
        if 'dense_mode' not in model_options:
            model_options['dense_mode'] = default.dense_mode
        if 'freeze_control_points' not in model_options:
            model_options['freeze_control_points'] = default.freeze_control_points
        if 'freeze_template' not in model_options:
            model_options['freeze_template'] = default.freeze_template
        if 'initial_control_points' not in model_options:
            model_options['initial_control_points'] = default.initial_control_points
        if 'initial_cp_spacing' not in model_options:
            model_options['initial_cp_spacing'] = default.initial_cp_spacing
        if 'deformation_kernel_width' not in model_options:
            model_options['deformation_kernel_width'] = default.deformation_kernel_width
        if 'deformation_kernel_type' not in model_options:
            model_options['deformation_kernel_type'] = default.deformation_kernel_type
        if 'downsampling_factor' not in model_options:
            model_options['downsampling_factor'] = default.downsampling_factor
        if 'use_sobolev_gradient' not in model_options:
            model_options['use_sobolev_gradient'] = default.use_sobolev_gradient
        if 'sobolev_kernel_width_ratio' not in model_options:
            model_options['sobolev_kernel_width_ratio'] = default.sobolev_kernel_width_ratio
        if 'estimate_initial_velocity' not in model_options:
            model_options['estimate_initial_velocity'] = default.estimate_initial_velocity
        if 'initial_velocity_weight' not in model_options:
            model_options['initial_velocity_weight'] = default.initial_velocity_weight

        if estimator_options is not None:
            if 'use_cuda' not in estimator_options:
                estimator_options['use_cuda'] = default.use_cuda
            if 'state_file' not in estimator_options:
                estimator_options['state_file'] = default.state_file
            if 'load_state_file' not in estimator_options:
                estimator_options['load_state_file'] = default.load_state_file
            if 'memory_length' not in estimator_options:
                estimator_options['memory_length'] = default.memory_length

        #
        # Global variables for this method.
        #

        if estimator_options is not None:
            cuda_is_used = estimator_options['use_cuda']
        else:
            cuda_is_used = False

        #
        # Check and completes the user-given parameters.
        #

        # If needed, infer the dimension from the template specifications.
        if model_options['dimension'] is None:
            model_options['dimension'] = self.__infer_dimension(template_specifications)

        # Smoothing kernel width.
        if model_options['use_sobolev_gradient']:
            model_options['smoothing_kernel_width'] = \
                model_options['deformation_kernel_width'] * model_options['sobolev_kernel_width_ratio']

        # Dense mode.
        if model_options['dense_mode']:
            print('>> Dense mode activated. No distinction will be made between template and control points.')
            assert len(template_specifications) == 1, \
                'Only a single object can be considered when using the dense mode.'
            if not model_options['freeze_control_points']:
                model_options['freeze_control_points'] = True
                msg = 'With active dense mode, the freeze_template (currently %s) and freeze_control_points ' \
                      '(currently %s) flags are redundant. Defaulting to freeze_control_points = True.' \
                      % (str(model_options['freeze_template']), str(model_options['freeze_control_points']))
                print('>> ' + msg)
            if model_options['initial_control_points'] is not None:
                model_options['initial_control_points'] = None
                msg = 'With active dense mode, specifying initial_control_points is useless. Ignoring this xml entry.'
                print('>> ' + msg)

        if model_options['initial_cp_spacing'] is None and model_options['initial_control_points'] is None \
                and not model_options['dense_mode']:
            print('>> No initial CP spacing given: using diffeo kernel width of '
                  + str(model_options['deformation_kernel_width']))
            model_options['initial_cp_spacing'] = model_options['deformation_kernel_width']

        # We also set the type to FloatTensor if keops is used.
        def keops_is_used():
            if model_options['deformation_kernel_type'].lower() == 'keops':
                return True
            for elt in template_specifications.values():
                if 'kernel_type' in elt and elt['kernel_type'].lower() == 'keops':
                    return True
            return False

        if keops_is_used():
            assert platform not in ['darwin'], 'The "keops" kernel is not available with the Mac OS X platform.'

            print(">> KEOPS is used at least in one operation, all operations will be done with FLOAT precision.")
            model_options['tensor_scalar_type'] = torch.FloatTensor

            if torch.cuda.is_available():
                print('>> CUDA is available: the KEOPS backend will automatically be set to "gpu".')
                cuda_is_used = True
            else:
                print('>> CUDA seems to be unavailable: the KEOPS backend will automatically be set to "cpu".')

        # Setting tensor types according to CUDA availability and user choices.
        if cuda_is_used:

            if not torch.cuda.is_available():
                msg = 'CUDA seems to be unavailable. All computations will be carried out on CPU.'
                print('>> ' + msg)

            else:
                print(">> CUDA is used at least in one operation, all operations will be done with FLOAT precision.")
                if estimator_options is not None and estimator_options['use_cuda']:
                    print(">> All tensors will be CUDA tensors.")
                    model_options['tensor_scalar_type'] = torch.cuda.FloatTensor
                    model_options['tensor_integer_type'] = torch.cuda.LongTensor
                else:
                    print(">> Setting tensor type to float.")
                    model_options['tensor_scalar_type'] = torch.FloatTensor

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError as error:
            print('>> Warning: ' + str(error) + ' [ in xml_parameters ]. Ignoring.')

        if estimator_options is not None:
            # Initializes the state file.
            if estimator_options['state_file'] is None:
                path_to_state_file = os.path.join(self.output_dir, "acceleration_diffeos-state.p")
                print('>> No specified state-file. By default, AccelerationDiffeos state will by saved in file: %s.' %
                      path_to_state_file)
                if os.path.isfile(path_to_state_file):
                    os.remove(path_to_state_file)
                    print('>> Removing the pre-existing state file with same path.')
                estimator_options['state_file'] = path_to_state_file
            else:
                if os.path.exists(estimator_options['state_file']):
                    estimator_options['load_state_file'] = True
                    print(
                        '>> AccelerationDiffeos will attempt to resume computation from the user-specified state file: %s.'
                        % estimator_options['state_file'])
                else:
                    msg = 'The user-specified state-file does not exist: %s. State cannot be reloaded. ' \
                          'Future AccelerationDiffeos state will be saved at the given path.' % estimator_options[
                              'state_file']
                    print('>> ' + msg)

            # Warning if scipy-LBFGS with memory length > 1 and sobolev gradient.
            if estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower() \
                    and estimator_options['memory_length'] > 1 \
                    and not model_options['freeze_template'] and model_options['use_sobolev_gradient']:
                print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                      'being larger than 1. Beware: that can be tricky.')

        # Checking the number of image objects, and moving as desired the downsampling_factor parameter.
        count = 0
        for elt in template_specifications.values():
            if elt['deformable_object_type'].lower() == 'image':
                count += 1
                if not model_options['downsampling_factor'] == 1:
                    if 'downsampling_factor' in elt.keys():
                        print('>> Warning: the downsampling_factor option is specified twice. '
                              'Taking the value: %d.' % elt['downsampling_factor'])
                    else:
                        elt['downsampling_factor'] = model_options['downsampling_factor']
                        print('>> Setting the image grid downsampling factor to: %d.' %
                              model_options['downsampling_factor'])
        if count > 1:
            raise RuntimeError('Only a single image object can be used.')
        if count == 0 and not model_options['downsampling_factor'] == 1:
            msg = 'The "downsampling_factor" parameter is useful only for image data, ' \
                  'but none is considered here. Ignoring.'
            print('>> ' + msg)

        return template_specifications, model_options, estimator_options

    @staticmethod
    def __infer_dimension(template_specifications):
        reader = DeformableObjectReader()
        max_dimension = 0
        for elt in template_specifications.values():
            object_filename = elt['filename']
            object_type = elt['deformable_object_type']
            o = reader.create_object(object_filename, object_type, dimension=None)
            d = o.dimension
            max_dimension = max(d, max_dimension)
        return max_dimension
