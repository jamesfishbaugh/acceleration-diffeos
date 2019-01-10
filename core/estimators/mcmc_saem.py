import logging
import os.path

from core import default
from core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from core.estimators.abstract_estimator import AbstractEstimator
from core.estimators.gradient_ascent import GradientAscent
from in_out.array_readers_and_writers import *

logger = logging.getLogger(__name__)


class McmcSaem(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method_type='undefined', individual_RER={},
                 max_iterations=default.max_iterations,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 sampler=default.sampler,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 sample_every_n_mcmc_iters=None,
                 convergence_tolerance=default.convergence_tolerance,
                 callback=None, output_dir=default.output_dir,
                 scale_initial_step_size=default.scale_initial_step_size, initial_step_size=default.initial_step_size,
                 max_line_search_iterations=default.max_line_search_iterations,
                 line_search_shrink=default.line_search_shrink, line_search_expand=default.line_search_expand,
                 load_state_file=default.load_state_file, state_file=default.state_file,
                 **kwargs):

        AbstractEstimator.__init__(self, statistical_model=statistical_model, dataset=dataset, name='McmcSaem',
                                   # optimized_log_likelihood=optimized_log_likelihood,
                                   max_iterations=max_iterations,
                                   convergence_tolerance=convergence_tolerance,
                                   print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                                   individual_RER=individual_RER,
                                   callback=callback,
                                   state_file=state_file,
                                   output_dir=output_dir)

        assert optimization_method_type.lower() == self.name.lower()

        self.current_mcmc_iteration = 0

        self.sample_every_n_mcmc_iters = sample_every_n_mcmc_iters

        assert sampler.lower() == 'SrwMhwg'.lower(), \
            "The only available sampler for now is the Symmetric-Random-Walk Metropolis-Hasting-within-Gibbs " \
            "(SrwMhhwg) sampler."
        self.sampler = SrwMhwgSampler(individual_proposal_distributions=individual_proposal_distributions)

        self.sufficient_statistics = None  # Dictionary of numpy arrays.
        self.number_of_burn_in_iterations = None  # Number of iterations without memory.

        self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
        self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.

        self.memory_window_size = 1  # Size of the averaging window for the acceptance rates.
        self.current_acceptance_rates_in_window = None  # Memory of the last memory_window_size acceptance rates.
        self.average_acceptance_rates_in_window = None  # Moving average of current_acceptance_rates_in_window.

        self.model_parameters_trajectory = None  # Memory of the model parameters along the estimation.
        self.save_model_parameters_every_n_iters = None  # Resolution of the model parameters trajectory.

        self.individual_random_effects_samples_stack = None  # Stack of the last individual random effect samples.

        # Initialization of the gradient-based optimizer.
        # TODO let the possibility to choose all options (e.g. max_iterations, or ScipyLBFGS optimizer).
        self.gradient_based_estimator = GradientAscent(
            statistical_model, dataset,
            optimized_log_likelihood='class2',
            max_iterations=5, convergence_tolerance=convergence_tolerance,
            print_every_n_iters=1, save_every_n_iters=100000,
            scale_initial_step_size=scale_initial_step_size, initial_step_size=initial_step_size,
            max_line_search_iterations=max_line_search_iterations,
            line_search_shrink=line_search_shrink,
            line_search_expand=line_search_expand,
            output_dir=output_dir, individual_RER=individual_RER,
            optimization_method_type='GradientAscent'
        )

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Initialization -----------------------------------------------------------------------------------------------
        self._initialize_number_of_burn_in_iterations()
        self._initialize_acceptance_rate_information()
        sufficient_statistics = self._initialize_sufficient_statistics()
        self._initialize_model_parameters_trajectory()
        self._initialize_individual_random_effects_samples_stack()

        # Ensures that all the model fixed effects are initialized.
        self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)

        # Print initial console information.
        print('------------------------------------- Iteration: ' + str(
            self.current_iteration) + ' -------------------------------------')
        print('>> MCMC-SAEM algorithm launched for ' + str(self.max_iterations) + ' iterations (' + str(
            self.number_of_burn_in_iterations) + ' iterations of burn-in).')
        self.statistical_model.print(self.individual_RER)

        # Initialization of the average random effects realizations.
        averaged_population_RER = {key: np.zeros(value.shape) for key, value in self.population_RER.items()}
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.callback_ret and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            step = self._compute_step_size()

            # Simulation.
            current_model_terms = None
            for n in range(self.sample_every_n_mcmc_iters):
                self.current_mcmc_iteration += 1

                # Single iteration of the MCMC.
                self.current_acceptance_rates, current_model_terms = self.sampler.sample(
                    self.statistical_model, self.dataset, self.population_RER, self.individual_RER,
                    current_model_terms)

                # Adapt proposal variances.
                self._update_acceptance_rate_information()
                if not (self.current_mcmc_iteration % self.memory_window_size):
                    self.average_acceptance_rates_in_window = {
                    key: np.mean(self.current_acceptance_rates_in_window[key])
                    for key in self.sampler.individual_proposal_distributions.keys()}
                    self.sampler.adapt_proposal_distributions(self.average_acceptance_rates_in_window,
                                                              self.current_mcmc_iteration,
                                                              not self.current_iteration % self.print_every_n_iters and n == self.sample_every_n_mcmc_iters - 1)

            # Maximization for the class 1 fixed effects.
            sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
                self.dataset, self.population_RER, self.individual_RER, model_terms=current_model_terms)
            self.sufficient_statistics = {key: value + step * (sufficient_statistics[key] - value) for key, value in
                                          self.sufficient_statistics.items()}
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)

            # Maximization for the class 2 fixed effects.
            if self.current_iteration < self.number_of_burn_in_iterations:

                self.statistical_model.preoptimize(self.individual_RER)
            fixed_effects_before_maximization = self.statistical_model.get_fixed_effects()
            self._maximize_over_fixed_effects()
            fixed_effects_after_maximization = self.statistical_model.get_fixed_effects()
            fixed_effects = {key: value + step * (fixed_effects_after_maximization[key] - value) for key, value in
                             fixed_effects_before_maximization.items()}
            self.statistical_model.set_fixed_effects(fixed_effects)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                coefficient_1 = float(self.current_iteration + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_population_RER = {key: value * coefficient_2 + self.population_RER[key] / coefficient_1 for
                                           key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1 for
                                           key, value in averaged_individual_RER.items()}
                self._update_individual_random_effects_samples_stack()

            else:
                averaged_individual_RER = self.individual_RER
                averaged_population_RER = self.population_RER

            # Saving, printing, writing.
            if not (self.current_iteration % self.save_model_parameters_every_n_iters):
                self._update_model_parameters_trajectory()
            if not (self.current_iteration % self.print_every_n_iters):
                self.print()
            if not (self.current_iteration % self.save_every_n_iters):
                self.write()

                # Call user callback function ------------------------------------------------------------------------------
                # TODO: add user callback
                # if self.callback is not None:
                #     self.callback_ret = self.callback(self._get_callback_data(float(self.current_log_likelihood),
                #                                                               float(self.current_attachment),
                #                                                               float(self.current_regularity)))

        # Finalization -------------------------------------------------------------------------------------------------
        self.population_RER = averaged_population_RER
        self.individual_RER = averaged_individual_RER

    def print(self):
        """
        Prints information.
        """
        # Iteration number.
        print('')
        print('------------------------------------- Iteration: ' + str(
            self.current_iteration) + ' -------------------------------------')

        # Averaged acceptance rates over all the past iterations.
        print('>> Average acceptance rates (all past iterations):')
        for random_effect_name, average_acceptance_rate in self.average_acceptance_rates.items():
            print('\t\t %.2f \t[ %s ]' % (average_acceptance_rate, random_effect_name))

        # Let the model under optimization print information about itself.
        self.statistical_model.print(self.individual_RER)

    def write(self, population_RER=None, individual_RER=None):
        """
        Save the current results.
        """
        # Call the write method of the statistical model.
        if population_RER is None:
            population_RER = self.population_RER
        if individual_RER is None:
            individual_RER = self.individual_RER
        self.statistical_model.write(self.dataset, population_RER, individual_RER, self.output_dir,
                                     update_fixed_effects=False)

        # Save the recorded model parameters trajectory.
        # self.model_parameters_trajectory is a list of dictionaries
        np.save(os.path.join(self.output_dir, self.statistical_model.name + '__EstimatedParameters__Trajectory.npy'),
                np.array(
                    {key: value[:(1 + int(self.current_iteration / float(self.save_model_parameters_every_n_iters)))]
                     for key, value in self.model_parameters_trajectory.items()}))

        # Save the memorized individual random effects samples.
        if self.current_iteration > self.number_of_burn_in_iterations:
            np.save(os.path.join(self.output_dir,
                                 self.statistical_model.name + '__EstimatedParameters__IndividualRandomEffectsSamples.npy'),
                    {key: value[:(self.current_iteration - self.number_of_burn_in_iterations)] for key, value in
                     self.individual_random_effects_samples_stack.items()})

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available (i.e. based on sufficient
        statistics).
        """

        # Default optimizer, if not initialized in the launcher.
        # Should better be done in a dedicated initializing method. TODO.
        if self.statistical_model.has_maximization_procedure is not None \
                and self.statistical_model.has_maximization_procedure:
            self.statistical_model.maximize(self.individual_RER, self.dataset)

        else:
            self.gradient_based_estimator.initialize()

            if self.gradient_based_estimator.verbose > 0:
                print('')
                print('[ maximizing over the fixed effects with the %s optimizer ]'
                      % self.gradient_based_estimator.name)

            success = False
            while not success:
                try:
                    self.gradient_based_estimator.update()
                    success = True
                except RuntimeError as error:
                    print('>> ' + str(error.args[0]) + ' [ in mcmc_saem ]')
                    self.statistical_model.adapt_to_error(error.args[1])

            if self.gradient_based_estimator.verbose > 0:
                print('')
                print('[ end of the gradient-based maximization ]')

    ####################################################################################################################
    ### Other private methods:
    ####################################################################################################################

    def _compute_step_size(self):
        aux = self.current_iteration - self.number_of_burn_in_iterations + 1
        if aux <= 0:
            return 1.0
        else:
            return 1.0 / aux

    def _initialize_number_of_burn_in_iterations(self):
        if self.number_of_burn_in_iterations is None:
            # Because some models will set it manually (e.g. deep Riemannian models)
            if self.max_iterations > 4000:
                self.number_of_burn_in_iterations = self.max_iterations - 2000
            else:
                self.number_of_burn_in_iterations = int(self.max_iterations / 2)

    def _initialize_acceptance_rate_information(self):
        # Initialize average_acceptance_rates.
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

        # Initialize current_acceptance_rates_in_window.
        self.current_acceptance_rates_in_window = {key: np.zeros((self.memory_window_size,))
                                                   for key in self.sampler.individual_proposal_distributions.keys()}
        self.average_acceptance_rates_in_window = {key: 0.0
                                                   for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        # Update average_acceptance_rates.
        coefficient_1 = float(self.current_mcmc_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

        # Update current_acceptance_rates_in_window.
        for key in self.current_acceptance_rates_in_window.keys():
            self.current_acceptance_rates_in_window[key][(self.current_mcmc_iteration - 1) % self.memory_window_size] = \
            self.current_acceptance_rates[key]

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(self.dataset, self.population_RER,
                                                                                     self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics

    ####################################################################################################################
    ### Model parameters trajectory saving methods:
    ####################################################################################################################

    def _initialize_model_parameters_trajectory(self):
        number_of_trajectory_points = min(self.max_iterations, 500)
        self.save_model_parameters_every_n_iters = max(1, int(self.max_iterations / float(number_of_trajectory_points)))
        self.model_parameters_trajectory = {}
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            self.model_parameters_trajectory[key] = np.zeros((number_of_trajectory_points + 1, value.size))
            self.model_parameters_trajectory[key][0, :] = value.flatten()

    def _update_model_parameters_trajectory(self):
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            self.model_parameters_trajectory[key][
            int(self.current_iteration / float(self.save_model_parameters_every_n_iters)), :] = value.flatten()

    def _get_vectorized_individual_RER(self):
        return np.concatenate([value.flatten() for value in self.individual_RER.values()])

    def _initialize_individual_random_effects_samples_stack(self):
        number_of_concentration_iterations = self.max_iterations - self.number_of_burn_in_iterations
        self.individual_random_effects_samples_stack = {}
        for (key, value) in self.individual_RER.items():
            if number_of_concentration_iterations > 0:
                self.individual_random_effects_samples_stack[key] = np.zeros(
                    (number_of_concentration_iterations, value.size))
                self.individual_random_effects_samples_stack[key][0, :] = value.flatten()

    def _update_individual_random_effects_samples_stack(self):
        for (key, value) in self.individual_RER.items():
            self.individual_random_effects_samples_stack[key][
            self.current_iteration - self.number_of_burn_in_iterations - 1, :] = value.flatten()
