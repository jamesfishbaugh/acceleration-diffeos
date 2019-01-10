import logging

import torch
import math

from core import default
from core.model_tools.deformations.exponential import Exponential
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import create_regular_grid_of_points, remove_useless_control_points
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from support.probability_distributions.normal_distribution import NormalDistribution
from support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
import support.kernels as kernel_factory

logger = logging.getLogger(__name__)


# What's left TODO:
# - add the correct penalty (what is the correct penalty ?)
# - configure a functionnal test


class PrincipalGeodesicAnalysis(AbstractStatisticalModel):
    """
    Principal geodesic analysis object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dataset, template_specifications,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=None,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                 initial_cp_spacing=default.initial_cp_spacing,
                 freeze_template=default.freeze_template,
                 freeze_control_points=default.freeze_control_points,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,
                 dense_mode=default.dense_mode,
                 latent_space_dimension=default.latent_space_dimension,
                 number_of_threads=default.number_of_threads):

        assert (
        dataset.is_cross_sectional()), "Cannot estimate a principal geodesic analysis from a non-cross-sectional dataset."
        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas')

        self.dataset = dataset

        object_list, self.objects_name, self.objects_name_extension, self.objects_noise_variance, \
        self.multi_object_attachment = create_template_metadata(template_specifications, self.dataset.dimension,
                                                                self.dataset.tensor_scalar_type)
        self.template = DeformableMultiObject(object_list, self.dataset.dimension)

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dataset.dimension)
        #
        self.exponential = Exponential(dense_mode=dense_mode,
                                       kernel=kernel_factory.factory(deformation_kernel_type, deformation_kernel_width,
                                                                     device=deformation_kernel_device),
                                       shoot_kernel_type=shoot_kernel_type,
                                       number_of_time_points=number_of_time_points,
                                       use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        self.use_sobolev_gradient = use_sobolev_gradient
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type, smoothing_kernel_width,
                                                         device=deformation_kernel_device)

        self.smoothing_kernel_width = smoothing_kernel_width

        self.initial_cp_spacing = initial_cp_spacing
        self.number_of_subjects = None
        self.number_of_objects = None
        self.number_of_control_points = None
        self.bounding_box = None

        # Dictionary of numpy arrays.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['principal_directions'] = None
        self.fixed_effects['noise_variance'] = None

        # Dictionary of probability distributions
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        self.individual_random_effects['latent_positions'] = NormalDistribution()

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points
        self.freeze_principal_directions = False
        self.freeze_latent_positions = False
        self.freeze_noise_variance = False

        self.dense_mode = dense_mode
        self.number_of_threads = number_of_threads

        self.latent_space_dimension = latent_space_dimension

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp
        self.number_of_control_points = len(cp)

    def get_momenta(self):
        principal_directions = torch.from_numpy(self.get_principal_directions()).type(self.dataset.tensor_scalar_type)
        latent_positions = torch.from_numpy(self.get_latent_positions()).type(self.dataset.tensor_scalar_type)
        return self._momenta_from_latent_positions(principal_directions, latent_positions)

    def get_principal_directions(self):
        return self.fixed_effects['principal_directions']

    def set_principal_directions(self, pd):
        self.fixed_effects['principal_directions'] = pd

    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_principal_directions:
            out['principal_directions'] = self.fixed_effects['principal_directions']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_principal_directions:
            self.set_principal_directions(fixed_effects['principal_directions'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        self.template.update(self.dataset.dimension)
        self.number_of_objects = len(self.template.object_list)
        self.bounding_box = self.template.bounding_box

        self.set_template_data(self.template.get_data())
        if self.fixed_effects['control_points'] is None:
            self._initialize_control_points()
        else:
            self._initialize_bounding_box()

        self._initialize_noise_variance()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        template_data, template_points, control_points, principal_directions \
            = self._fixed_effects_to_torch_tensors(with_grad)

        latent_positions = self._individual_RER_to_torch_tensors(individual_RER, with_grad)

        momenta = self._momenta_from_latent_positions(principal_directions, latent_positions)

        residuals = self._compute_residuals(dataset, template_data, template_points, control_points, momenta)

        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals=residuals)

            self.update_fixed_effects(dataset, sufficient_statistics)

        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)

        regularity = 0.0
        if mode == 'complete':
            regularity += self._compute_random_effects_regularity(latent_positions)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, control_points)

        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    if self.use_sobolev_gradient:
                        gradient['landmark_points'] = self.sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()

            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()

            if not self.freeze_principal_directions:
                gradient['principal_directions'] = principal_directions.grad.detach().cpu().numpy()

            if mode == 'complete':
                gradient['latent_positions'] = latent_positions.grad.detach().cpu().numpy()

            # Return as appropriate.
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.detach().cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()
            elif mode == 'model':
                return attachments.detach().cpu().numpy()

    def _compute_residuals(self, dataset, template_data, template_points, control_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        for i, target in enumerate(targets):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

        return residuals

    def _compute_individual_attachments(self, residuals):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        attachments = torch.zeros((number_of_subjects,)).type(self.dataset.tensor_scalar_type)
        for i in range(number_of_subjects):
            attachments[i] = - 0.5 * torch.sum(residuals[i] / torch.from_numpy(
                self.fixed_effects['noise_variance']).type(self.dataset.tensor_scalar_type))
        return attachments

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None):
        """
        Compute the model sufficient statistics.
        """
        if residuals is None:
            # Initialize: conversion from numpy to torch ---------------------------------------------------------------

            # Template data.
            template_data = self.fixed_effects['template_data']
            template_data = torch.from_numpy(template_data).type(self.dataset.tensor_scalar_type)

            # Control points.
            control_points = self.fixed_effects['control_points']
            control_points = torch.from_numpy(control_points).type(self.dataset.tensor_scalar_type)

            # Principal directions
            principal_directions = self.fixed_effects['principal_directions']
            principal_directions = torch.from_numpy(principal_directions).type(self.dataset.tensor_scalar_type)

            # Latent positions
            latent_positions = self.fixed_effects['latent_positions']
            latent_positions = torch.from_numpy(latent_positions).type(self.dataset.tensor_scalar_type)

            # Momenta.
            momenta = self._momenta_from_latent_positions(principal_directions, latent_positions)

            # Compute residuals ----------------------------------------------------------------------------------------
            residuals = [torch.sum(residuals_i)
                         for residuals_i in self._compute_residuals(dataset, template_data, control_points, momenta)]

        # Compute sufficient statistics --------------------------------------------------------------------------------
        sufficient_statistics = {}

        # Empirical residuals variances, for each object.
        sufficient_statistics['S2'] = np.zeros((self.number_of_objects,))
        for i in range(dataset.number_of_subjects):
            sufficient_statistics['S2'] += residuals[i].detach().cpu().numpy()

        # Finalization -------------------------------------------------------------------------------------------------
        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        # Variance of the residual noise update.
        noise_variance = np.zeros((self.number_of_objects,))
        prior_scale_scalars = self.priors['noise_variance'].scale_scalars
        prior_dofs = self.priors['noise_variance'].degrees_of_freedom
        for k in range(self.number_of_objects):
            noise_variance[k] = (sufficient_statistics['S2'][k] + prior_scale_scalars[k] * prior_dofs[k]) \
                                / float(dataset.number_of_subjects * self.objects_noise_dimension[k] + prior_dofs[k])
        self.set_noise_variance(noise_variance)

    def _compute_random_effects_regularity(self, latent_positions):
        """
        Fully torch.
        """
        number_of_subjects = latent_positions.shape[0]
        regularity = 0.0

        # Momenta random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['latent_positions'].compute_log_likelihood_torch(
                latent_positions[i],
                self.dataset.tensor_scalar_type)

        return regularity

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Noise variance prior.
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, template_data, control_points):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # We regularize the principal directions. How to do this cleanly ?
        if not self.freeze_principal_directions:
            regularity += 0.0

        return regularity

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _initialize_control_points(self):
        """
        Initialize the control points fixed effect.
        """
        if not self.dense_mode:
            control_points = create_regular_grid_of_points(self.bounding_box, self.initial_cp_spacing,
                                                           self.dataset.dimension)
            for elt in self.template.object_list:
                if elt.type.lower() == 'image':
                    control_points = remove_useless_control_points(control_points, elt,
                                                                   self.exponential.get_kernel_width())
                    break
        else:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = self.template.get_points()['landmark_points']

        self.set_control_points(control_points)
        self.number_of_control_points = control_points.shape[0]
        logger.info('Set of ' + str(self.number_of_control_points) + ' control points defined.')

    def _initialize_latent_positions(self):
        """
        Initialize the momenta fixed effect.
        """

        assert (self.number_of_subjects > 0)
        latent_positions = np.zeros(
            (self.number_of_subjects, self.latent_space_dimension))
        self.set_latent_positions(latent_positions)
        logger.info('Latent positions initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

    def _initialize_bounding_box(self):
        """
        Initialize the bounding box. which tightly encloses all template objects and the atlas control points.
        Relevant when the control points are given by the user.
        """

        assert (self.number_of_control_points > 0)

        control_points = self.get_control_points()

        for k in range(self.number_of_control_points):
            for d in range(self.dataset.dimension):
                if control_points[k, d] < self.bounding_box[d, 0]:
                    self.bounding_box[d, 0] = control_points[k, d]
                elif control_points[k, d] > self.bounding_box[d, 1]:
                    self.bounding_box[d, 1] = control_points[k, d]

    def _initialize_noise_variance(self):
        self.set_noise_variance(np.asarray(self.priors['noise_variance'].scale_scalars))

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: torch.from_numpy(value).type(self.dataset.tensor_scalar_type)
                         for key, value in template_data.items()}

        for val in template_data.values():
            val.requires_grad_(not self.freeze_template and with_grad)

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: torch.from_numpy(value).type(self.dataset.tensor_scalar_type)
                           for key, value in template_points.items()}
        for val in template_points.values():
            val.requires_grad_(not self.freeze_template and with_grad)

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = torch.from_numpy(control_points).type(self.dataset.tensor_scalar_type)
            control_points.requires_grad_((not self.freeze_control_points and with_grad)
                                          or self.exponential.get_kernel_type() == 'keops')

        pd = self.fixed_effects['principal_directions']
        principal_directions = torch.from_numpy(pd).type(self.dataset.tensor_scalar_type)
        principal_directions.requires_grad_(not self.freeze_principal_directions and with_grad)

        return template_data, template_points, control_points, principal_directions

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        latent_positions = torch.from_numpy(individual_RER['latent_positions']).type(self.dataset.tensor_scalar_type)
        latent_positions.requires_grad_(with_grad)
        return latent_positions

    def _momenta_from_latent_positions(self, principal_directions, latent_positions):

        assert principal_directions.size()[1] == latent_positions.size()[1], 'Incorrect shape of principal directions ' \
                                                                             'or latent positions'
        a, b = self.get_control_points().shape

        return torch.mm(principal_directions, latent_positions.view(self.latent_space_dimension, -1)).reshape(
            len(latent_positions), a, b)

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(self.dataset, individual_RER, output_dir,
                                                  compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir)

        # Write the principal directions
        self._write_principal_directions(output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):

        # Initialize.
        template_data, template_points, control_points, principal_directions = self._fixed_effects_to_torch_tensors(
            False)

        latent_positions = self._individual_RER_to_torch_tensors(individual_RER, False)

        momenta = self._momenta_from_latent_positions(principal_directions, latent_positions)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):

            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, dataset.deformable_objects[i][0]))

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        # We write the latent space positions:
        write_2D_array(latent_positions.detach().cpu().numpy(), output_dir,
                       self.name + "__EstimatedParameters__LatentPositions.txt")

        return residuals

    def _write_model_parameters(self, output_dir):

        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Principal Directions
        write_2D_array(self.get_principal_directions(), output_dir,
                       self.name + '__EstimatedParameters__PrincipalDirections.txt')

    def _write_principal_directions(self, output_dir):

        template_data, template_points, control_points, principal_directions = self._fixed_effects_to_torch_tensors(
            False)

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        for i in range(self.latent_space_dimension):
            for pos in np.arange(-1., 1., 0.2):
                lp = np.zeros(self.latent_space_dimension)
                lp[i] = 1.
                lp = pos * lp
                lp_torch = torch.from_numpy(lp).type(self.dataset.tensor_scalar_type)
                momenta = torch.mv(principal_directions, lp_torch).view(control_points.size())

                self.exponential.set_initial_momenta(momenta)
                self.exponential.update()

                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                names = []
                for k, (object_name, object_extension) \
                        in enumerate(zip(self.objects_name, self.objects_name_extension)):
                    name = self.name + '__PrincipalDirection__{}_{}{}'.format(i, str(pos)[:min(5, len(str(pos)))],
                                                                              object_extension)
                    names.append(name)

                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})
