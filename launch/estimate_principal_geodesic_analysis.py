from core import default
from core.models.principal_geodesic_analysis import PrincipalGeodesicAnalysis
from in_out.array_readers_and_writers import *
from core.models.deterministic_atlas import DeterministicAtlas

import torch
import os
from scipy.linalg import sqrtm
from numpy.linalg import inv, eigh
import logging

logger = logging.getLogger(__name__)


def pca_fit_and_transform(n_components, observations):
    assert len(observations.shape) == 2, 'Wrong format of observations for pca.'
    nb_obs, dim = observations.shape
    assert dim >= n_components, 'Cannot estimate more components that the dimension of the observations'
    assert dim >= nb_obs, 'Cannot estimate more components than the number of observations'

    # We start by removing the mean of the observations
    observations_without_mean = observations - np.mean(observations, axis=0)

    X = np.matmul(observations_without_mean.transpose(), observations_without_mean)  # X is a  dim x dim matrix

    # Computing eigenvalues and the normalized eigenvectors
    eigenvalues, eigenvectors = eigh(X)

    components = eigenvectors[:n_components, :]

    # We now project the observations:
    latent_positions = np.array([np.matmul(components, elt) for elt in observations])

    return components, latent_positions


def run_tangent_pca(deformetrica, template_specifications, dataset, deformation_kernel, latent_space_dimension,
                    **kwargs):
    """
    Initialization for the principal geodesic analysis.
    """

    from core.estimators.scipy_optimize import ScipyOptimize

    # Standard estimator and options here.
    estimator_options = {'memory_length': 10,
                         'freeze_template': False,
                         'use_sobolev_gradient': True,
                         'max_iterations': 10,
                         'max_line_search_iterations': 10,
                         'print_every_n_iters': 5,
                         'save_every_n_iters': 20,
                         'optimized_log_likelihood': 'complete'}

    output_dir = os.path.join(deformetrica.output_dir, 'preprocessing')
    pga_output_dir = deformetrica.output_dir  # to restore later
    deformetrica.output_dir = output_dir

    if not os.path.isdir(deformetrica.output_dir):
        os.mkdir(deformetrica.output_dir)

    determ_atlas = DeterministicAtlas(template_specifications, dataset.number_of_subjects,
                                      deformation_kernel_type=deformation_kernel.kernel_type,
                                      deformation_kernel_width=deformation_kernel.kernel_width, **kwargs)
    determ_atlas.initialize_noise_variance(dataset)

    estimator = ScipyOptimize(determ_atlas, dataset, output_dir=deformetrica.output_dir, **estimator_options)

    logger.info('Estimating a deterministic atlas for initialization')
    estimator.update()
    logger.info('Done estimating the deterministic atlas')
    estimator.write()

    # We then read the result in the output dir and perform the pca
    control_points = read_2D_array(
        os.path.join(deformetrica.output_dir, 'DeterministicAtlas__EstimatedParameters__ControlPoints.txt'))
    a, b = control_points.shape
    momenta = read_3D_array(
        os.path.join(deformetrica.output_dir, 'DeterministicAtlas__EstimatedParameters__Momenta.txt'))

    control_points_torch = torch.from_numpy(control_points)

    kernel_matrix = deformation_kernel.get_kernel_matrix(control_points_torch).detach().numpy()
    sqrt_kernel_matrix = sqrtm(kernel_matrix)
    inv_sqrt_kernel_matrix = inv(sqrt_kernel_matrix)
    momenta_l2 = np.array([np.matmul(sqrt_kernel_matrix, elt).flatten() for elt in momenta])

    ### ALTERNATIVE SKLEARN VERSION#####
    # from sklearn.decomposition import PCA
    # Computing principal directions
    # pca = PCA(n_components=latent_space_dimension)
    # pca.fit(momenta_l2)

    # Now getting the components
    # components = np.array([np.matmul(inv_sqrt_kernel_matrix, elt.reshape(a, b)) for elt in pca.components_])\
    #     .reshape(a*b, latent_space_dimension)
    #
    # latent_positions = pca.transform(momenta_l2)
    ########################################

    components, latent_positions = pca_fit_and_transform(latent_space_dimension, momenta_l2)

    components = np.array([np.matmul(inv_sqrt_kernel_matrix, elt.reshape(a, b)) for elt in components]) \
        .reshape(a * b, latent_space_dimension)

    # Restoring the correct output_dir
    deformetrica.output_dir = pga_output_dir

    # As a final step, we normalize the distribution of the latent positions
    stds = np.std(latent_positions, axis=0)
    latent_positions /= stds
    for i in range(latent_space_dimension):
        components[:, i] *= stds[i]

    return control_points, components, latent_positions, determ_atlas.template


def instantiate_principal_geodesic_model(deformetrica, dataset, template_specifications,
                                         deformation_kernel=default.deformation_kernel,
                                         shoot_kernel=None, flow_kernel=None,
                                         number_of_time_points=default.number_of_time_points,
                                         use_rk2_for_shoot=default.use_rk2_for_shoot,
                                         use_rk2_for_flow=default.use_rk2_for_flow,
                                         freeze_template=default.freeze_template,
                                         freeze_control_points=default.freeze_control_points,
                                         use_sobolev_gradient=default.use_sobolev_gradient,
                                         smoothing_kernel_width=default.smoothing_kernel_width,
                                         initial_control_points=default.initial_control_points,
                                         initial_cp_spacing=default.initial_cp_spacing,
                                         initial_latent_positions=default.initial_latent_positions,
                                         initial_principal_directions=default.initial_principal_directions,
                                         latent_space_dimension=default.latent_space_dimension,
                                         number_of_threads=default.number_of_threads,
                                         **kwargs):
    if initial_cp_spacing is None:
        initial_cp_spacing = deformation_kernel.kernel_width

    if initial_latent_positions is not None and initial_principal_directions is None:
        raise ('The latent positions are given, not the principal directions: I cannot estimate PGA.')

    model = PrincipalGeodesicAnalysis(
        dataset,
        template_specifications,
        deformation_kernel,
        shoot_kernel=shoot_kernel, flow_kernel=flow_kernel,
        number_of_time_points=number_of_time_points,
        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
        freeze_template=freeze_template, freeze_control_points=freeze_control_points,
        use_sobolev_gradient=use_sobolev_gradient, smoothing_kernel_width=smoothing_kernel_width,
        latent_space_dimension=latent_space_dimension,
        number_of_threads=number_of_threads)

    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = initial_cp_spacing

    model.update()

    cp = model.get_control_points()
    a, b = cp.shape
    latent_positions = None

    if initial_principal_directions is not None:
        initial_principal_directions = read_2D_array(initial_principal_directions)
    else:
        # initialization by deterministic atlas
        control_points, initial_principal_directions, latent_positions, template = \
            run_tangent_pca(deformetrica, template_specifications, dataset, deformation_kernel,
                            latent_space_dimension=latent_space_dimension, **kwargs)
        model.set_control_points(control_points)
        model.template = template
        # Naive initialization: (note that 0 is no good)
        # initial_principal_directions = np.random.normal(size=(a * b, latent_space_dimension)) * 0.1 # -> fast but not smart

    model.set_principal_directions(initial_principal_directions)

    # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
    for k, object in enumerate(template_specifications.values()):
        model.priors['noise_variance'].degrees_of_freedom.append(dataset.number_of_subjects
                                                                 * object['noise_variance_prior_normalized_dof']
                                                                 * model.objects_noise_dimension[k])

    # Initial random effects realizations.
    individual_RER = {}

    if initial_latent_positions is not None:
        individual_RER['latent_positions'] = read_2D_array(initial_latent_positions)
    elif latent_positions is not None:
        individual_RER['latent_positions'] = latent_positions
    else:
        individual_RER['latent_positions'] = np.zeros((dataset.number_of_subjects, latent_space_dimension))

    # Prior on the latent positions:
    model.individual_random_effects['latent_positions'].mean = np.zeros((latent_space_dimension,))
    model.individual_random_effects['latent_positions'].covariance_inverse = np.eye(latent_space_dimension)

    """
    Prior on the noise variance (inverse Wishart: scale scalars parameters).
    """

    latent_positions = torch.from_numpy(individual_RER['latent_positions']).type(dataset.tensor_scalar_type)

    td, tp, cp, pd = model._fixed_effects_to_torch_tensors(False)
    mom = model._momenta_from_latent_positions(pd, latent_positions)

    residuals_per_object = sum(model._compute_residuals(dataset, td, tp, cp, mom))
    for k, object in enumerate(template_specifications.values()):
        if object['noise_variance_prior_scale_std'] is None:
            model.priors['noise_variance'].scale_scalars.append(
                0.01 * residuals_per_object[k].detach().cpu().numpy()
                / model.priors['noise_variance'].degrees_of_freedom[k])
        else:
            model.priors['noise_variance'].scale_scalars.append(object['noise_variance_prior_scale_std'] ** 2)
    model.update()

    model.name = 'PrincipalGeodesicAnalysis'

    # Return the initialized model.
    return model, individual_RER
