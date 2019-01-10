import numpy as np
import torch
from torch.autograd import Variable


class NormalDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.mean = np.zeros((1,))
        # self.covariance = np.ones((1, 1))
        # self.covariance_sqrt = np.ones((1, 1))
        self.covariance_inverse = np.ones((1, 1))
        # self.covariance_log_determinant = 0

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_covariance(self, cov):
        # self.covariance = cov
        # self.covariance_sqrt = np.linalg.cholesky(cov)
        self.covariance_inverse = np.linalg.inv(cov)
        # self.covariance_log_determinant = np.linalg.slogdet(cov)[1]

    # def set_covariance_sqrt(self, cov_sqrt):
    #     self.covariance = np.dot(cov_sqrt, np.transpose(cov_sqrt))
    #     self.covariance_sqrt = cov_sqrt
    #     self.covariance_inverse = np.linalg.inv(self.covariance)

    def set_covariance_inverse(self, cov_inv):
        # self.covariance = np.linalg.inv(cov_inv)
        # self.covariance_sqrt = np.linalg.cholesky(np.linalg.inv(cov_inv))
        self.covariance_inverse = cov_inv
        # self.covariance_log_determinant = - np.linalg.slogdet(cov_inv)[1]

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return self.mean + np.dot(self.covariance_sqrt, np.random.standard_normal(self.mean.shape))

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        assert self.mean.shape == observation.ravel().shape
        delta = observation.ravel() - self.mean
        # return - 0.5 * (np.dot(delta, np.dot(self.covariance_inverse, delta)) + self.covariance_log_determinant)
        return - 0.5 * np.dot(delta, np.dot(self.covariance_inverse, delta))

    def compute_log_likelihood_torch(self, observation, tensor_scalar_type):
        """
        Torch inputs / outputs.
        Returns only the part that includes the observation argument.
        """
        mean = Variable(torch.from_numpy(self.mean).type(tensor_scalar_type), requires_grad=False)
        assert mean.view(-1, 1).size() == observation.view(-1, 1).size()
        covariance_inverse = Variable(torch.from_numpy(self.covariance_inverse).type(tensor_scalar_type),
                                      requires_grad=False)
        delta = observation.view(-1, 1) - mean.view(-1, 1)
        # return - 0.5 * (torch.dot(delta, torch.mm(covariance_inverse, delta)) + self.covariance_log_determinant)
        return - 0.5 * torch.dot(delta.view(-1), torch.mm(covariance_inverse, delta).view(-1))
