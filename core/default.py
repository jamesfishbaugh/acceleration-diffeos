import os
import torch


logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

tensor_scalar_type = torch.DoubleTensor
tensor_integer_type = torch.LongTensor
# deformation_kernel = kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=1.)
deformation_kernel = None

output_dir = os.path.join(os.getcwd(), 'output')
state_file = None
load_state_file = False

number_of_threads = 1

model_type = 'undefined'
template_specifications = {}
deformation_kernel_width = 1.0
deformation_kernel_type = 'keops'
deformation_kernel_device = 'auto'

shoot_kernel_type = None
number_of_time_points = 11
concentration_of_time_points = 10
number_of_sources = None
use_rk2_for_shoot = False
use_rk2_for_flow = False
t0 = None
tmin = float('inf')
tmax = - float('inf')
initial_cp_spacing = None
dimension = None
covariance_momenta_prior_normalized_dof = 0.001

dataset_filenames = []
visit_ages = []
subject_ids = []
optimization_method_type = None
optimized_log_likelihood = 'complete'
max_iterations = 100
max_line_search_iterations = 10
save_every_n_iters = 100
print_every_n_iters = 1
sample_every_n_mcmc_iters = 50
use_sobolev_gradient = True
sobolev_kernel_width_ratio = 1
smoothing_kernel_width = None
initial_step_size = None
line_search_shrink = 0.5
line_search_expand = 1.5
convergence_tolerance = 1e-4
memory_length = 10
scale_initial_step_size = True
downsampling_factor = 1

estimate_initial_velocity = True
initial_velocity_weight = 1.0

data_weight = 1.0
regularity_weight = 1.0

dense_mode = False
use_cuda = False
_cuda_is_used = False   # true if at least one operation will use CUDA.
_keops_is_used = False  # true if at least one keops kernel operation will take place.

freeze_template = False
freeze_control_points = True
freeze_momenta = False
freeze_modulation_matrix = False
freeze_reference_time = False
freeze_time_shift_variance = False
freeze_acceleration_variance = False
freeze_noise_variance = False

# affine atlas
freeze_translation_vectors = False
freeze_rotation_angles = False
freeze_scaling_ratios = False

# For metric learning atlas
freeze_metric_parameters = False
freeze_p0 = False
freeze_v0 = False
initial_control_points = None
initial_momenta = None
initial_velocity = None
impulse_t = None
initial_control_points_to_transport = None
initial_momenta_to_transport = None
initial_latent_positions = None
initial_principal_directions = None
initial_modulation_matrix = None
initial_time_shift_variance = None
initial_acceleration_mean = None
initial_acceleration_variance = None
initial_onset_ages = None
initial_accelerations = None
initial_sources = None
initial_sources_mean = None
initial_sources_std = None

sampler = None
individual_proposal_distributions = {}

momenta_proposal_std = 0.01
onset_age_proposal_std = 0.1
acceleration_proposal_std = 0.01
sources_proposal_std = 0.01

# For scalar inputs:
group_file = None
observations_file = None
timepoints_file = None
v0 = None
p0 = None
metric_parameters_file = None
interpolation_points_file = None
initial_noise_variance = None
exponential_type = None
number_of_metric_parameters = None  # number of parameters in metric learning.
number_of_interpolation_points = None
latent_space_dimension = None  # For deep metric learning
normalize_image_intensity = False
initialization_heuristic = False

verbose = 1
