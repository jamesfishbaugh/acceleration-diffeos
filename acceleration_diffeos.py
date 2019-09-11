#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os

import api
from core import default
from core.default import logger_format
from in_out.xml_parameters import XmlParameters

logger = logging.getLogger(__name__)


def main():

    # common options
    common_parser = argparse.ArgumentParser()
    common_parser.add_argument('--parameters', '-p', type=str, help='parameters xml file')
    common_parser.add_argument('--output', '-o', type=str, help='output folder')
    # logging levels: https://docs.python.org/2/library/logging.html#logging-levels
    common_parser.add_argument('--verbosity', '-v',
                               type=str,
                               default='WARNING',
                               choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='set output verbosity')

    # main parser
    description = 'Statistical analysis of 2D and 3D shape data.'
    parser = argparse.ArgumentParser(prog='accelerationdiffeos', description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='command', dest='command')
    subparsers.required = True  # make 'command' mandatory

    # estimate command
    parser_estimate = subparsers.add_parser('estimate', add_help=False, parents=[common_parser])
    parser_estimate.add_argument('model', type=str, help='model xml file')
    parser_estimate.add_argument('dataset', type=str, help='dataset xml file')

    # compute command
    parser_compute = subparsers.add_parser('compute', add_help=False, parents=[common_parser])
    parser_compute.add_argument('model', type=str, help='model xml file')

    # gui command
    subparsers.add_parser('gui', add_help=False, parents=[common_parser])

    args = parser.parse_args()

    # set logging level
    try:
        log_level = logging.getLevelName(args.verbosity)
        logging.basicConfig(level=log_level, format=logger_format)
    except ValueError:
        logger.warning('Logging level was not recognized. Using INFO.')
        log_level = logging.INFO

    logger.debug('Using verbosity level: ' + args.verbosity)
    logging.basicConfig(level=log_level, format=logger_format)

    if args.command == 'gui':
        StartGui().start()
        return 0
    else:

        """
        Read xml files, set general settings, and call the adapted function.
        """
        output_dir = None
        try:
            if args.output is None:
                output_dir = default.output_dir
                logger.info('No output directory defined, using default: ' + output_dir)
                os.makedirs(output_dir)
            else:
                logger.info('Setting output directory to: ' + args.output)
                output_dir = args.output
        except FileExistsError:
            pass

        acceleration_diffeos = api.AccelerationDiffeos(output_dir=output_dir)

        file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='w')
        logger.addHandler(file_handler)

        # logger.info('[ read_all_xmls function ]')
        xml_parameters = XmlParameters()
        xml_parameters.read_all_xmls(args.model,
                                     args.dataset if args.command == 'estimate' else None,
                                     args.parameters, output_dir)

        # logger.debug('xml_parameters.tensor_scalar_type=' + str(xml_parameters.tensor_scalar_type))
        if xml_parameters.model_type == 'AccelerationRegression'.lower():
            acceleration_diffeos.estimate_acceleration_regression(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'AccelerationGompertzRegression'.lower():
            acceleration_diffeos.estimate_acceleration_gompertz_regression(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'AccelerationFlow'.lower():
            acceleration_diffeos.compute_acceleration_flow(
                xml_parameters.template_specifications,
                model_options=get_model_options(xml_parameters))

        else:
            raise RuntimeError(
                'Unrecognized model-type: "' + xml_parameters.model_type + '". Check the corresponding field in the model.xml input file.')


def get_dataset_specifications(xml_parameters):
    specifications = {}
    specifications['visit_ages'] = xml_parameters.visit_ages
    specifications['dataset_filenames'] = xml_parameters.dataset_filenames
    specifications['subject_ids'] = xml_parameters.subject_ids
    return specifications


def get_estimator_options(xml_parameters):
    options = {}

    if xml_parameters.optimization_method_type.lower() == 'GradientAscent'.lower():
        options['initial_step_size'] = xml_parameters.initial_step_size
        options['scale_initial_step_size'] = xml_parameters.scale_initial_step_size
        options['line_search_shrink'] = xml_parameters.line_search_shrink
        options['line_search_expand'] = xml_parameters.line_search_expand
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations

    elif xml_parameters.optimization_method_type.lower() == 'ScipyLBFGS'.lower():
        options['memory_length'] = xml_parameters.memory_length
        options['freeze_template'] = xml_parameters.freeze_template
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations

    # common options
    options['optimization_method_type'] = xml_parameters.optimization_method_type.lower()
    options['max_iterations'] = xml_parameters.max_iterations
    options['convergence_tolerance'] = xml_parameters.convergence_tolerance
    options['print_every_n_iters'] = xml_parameters.print_every_n_iters
    options['save_every_n_iters'] = xml_parameters.save_every_n_iters
    options['use_cuda'] = xml_parameters.use_cuda
    options['state_file'] = xml_parameters.state_file
    options['load_state_file'] = xml_parameters.load_state_file

    # logger.debug(options)
    return options


def get_model_options(xml_parameters):

    options = {
        'deformation_kernel_type': xml_parameters.deformation_kernel_type,
        'deformation_kernel_width': xml_parameters.deformation_kernel_width,
        'deformation_kernel_device': xml_parameters.deformation_kernel_device,
        'use_rk2_for_shoot': xml_parameters.use_rk2_for_shoot,
        'use_rk2_for_flow': xml_parameters.use_rk2_for_flow,
        'freeze_template': xml_parameters.freeze_template,
        'freeze_control_points': xml_parameters.freeze_control_points,
        'use_sobolev_gradient': xml_parameters.use_sobolev_gradient,
        'sobolev_kernel_width_ratio': xml_parameters.sobolev_kernel_width_ratio,
        'initial_control_points': xml_parameters.initial_control_points,
        'initial_cp_spacing': xml_parameters.initial_cp_spacing,
        'initial_velocity': xml_parameters.initial_velocity,
        'impulse_t': xml_parameters.impulse_t,
        'dense_mode': xml_parameters.dense_mode,
        'downsampling_factor': xml_parameters.downsampling_factor,
        'dimension': xml_parameters.dimension,
        'estimate_initial_velocity' : xml_parameters.estimate_initial_velocity,
        'initial_velocity_weight': xml_parameters.initial_velocity_weight,
        'regularity_weight': xml_parameters.regularity_weight,
        'data_weight': xml_parameters.data_weight
    }

    if xml_parameters.model_type.lower() == 'AccelerationRegression'.lower():
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax

    elif xml_parameters.model_type.lower() == 'AccelerationGompertzRegression'.lower():
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax

    elif xml_parameters.model_type.lower() == 'AccelerationFlow'.lower():
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax

    # logger.debug(options)
    return options


if __name__ == "__main__":
    # execute only if run as a script
    main()
