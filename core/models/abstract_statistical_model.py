class AbstractStatisticalModel:
    """
    AbstractStatisticalModel object class.
    A statistical model is a generative function, which tries to explain an observed stochastic process.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, name='undefined'):
        self.name = name
        self.fixed_effects = {}
        self.priors = {}
        self.population_random_effects = {}
        self.individual_random_effects = {}
        self.has_maximization_procedure = None

    ####################################################################################################################
    ### Common methods, not necessarily useful for every model.
    ####################################################################################################################

    def clear_memory(self):
        pass

    def preoptimize(self, individual_RER):
        pass


