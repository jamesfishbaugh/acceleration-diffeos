import numpy as np
import torch
from torch.autograd import Variable

from core import default

import itk

import torch.nn.functional as f

class MultiObjectAttachment:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, attachment_types, kernels):
        # List of strings, e.g. 'varifold' or 'current'.
        self.attachment_types = attachment_types
        # List of kernel objects.
        self.kernels = kernels

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_weighted_distance(self, data, multi_obj1, multi_obj2, inverse_weights):
        """
        Takes two multiobjects and their new point positions to compute the distances
        """
        distances = self.compute_distances(data, multi_obj1, multi_obj2)
        assert distances.size()[0] == len(inverse_weights)
        inverse_weights_torch = torch.from_numpy(np.array(inverse_weights)).type(list(data.values())[0].type())
        return torch.sum(distances / inverse_weights_torch)

    def compute_distances(self, data, multi_obj1, multi_obj2):
        """
        Takes two multiobjects and their new point positions to compute the distances.
        """
        assert len(multi_obj1.object_list) == len(multi_obj2.object_list), \
            "Cannot compute distance between multi-objects which have different number of objects"
        distances = torch.zeros((len(multi_obj1.object_list),)).type(list(data.values())[0].type())

        pos = 0
        for i, obj1 in enumerate(multi_obj1.object_list):
            obj2 = multi_obj2.object_list[i]

            if self.attachment_types[i].lower() == 'current':
                distances[i] = self.current_distance(
                    data['landmark_points'][pos:pos + obj1.get_number_of_points()], obj1, obj2, self.kernels[i])
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == 'pointcloud':
                distances[i] = self.point_cloud_distance(
                    data['landmark_points'][pos:pos + obj1.get_number_of_points()], obj1, obj2, self.kernels[i])
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == 'varifold':
                distances[i] = self.varifold_distance(
                    data['landmark_points'][pos:pos + obj1.get_number_of_points()], obj1, obj2, self.kernels[i])
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == 'landmark':
                distances[i] = self.landmark_distance(
                    data['landmark_points'][pos:pos + obj1.get_number_of_points()], obj2)
                pos += obj1.get_number_of_points()

            elif self.attachment_types[i].lower() == 'l2':
                assert obj1.type.lower() == 'image' and obj2.type.lower() == 'image'
                distances[i] = self.L2_distance(data['image_intensities'], obj2)

            elif self.attachment_types[i].lower() == 'ncc':
                assert obj1.type.lower() == 'image' and obj2.type.lower() == 'image'
                distances[i] = self.cross_correlation(data['image_intensities'], obj2)

            else:
                assert False, "Please implement the distance {e} you are trying to use :)".format(
                    e=self.attachment_types[i])

        return distances

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    @staticmethod
    def current_distance(points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        """

        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target)

        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            return torch.dot(normals_1.view(-1), kernel.convolve(points_1, points_2, normals_2).view(-1))

        if target.norm is None:
            target.norm = current_scalar_product(c2, c2, n2, n2)

        return current_scalar_product(c1, c1, n1, n1) + target.norm - 2 * current_scalar_product(c1, c2, n1, n2)

    @staticmethod
    def point_cloud_distance(points, source, target, kernel):
        """
        Compute the point cloud distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        """

        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target)

        def point_cloud_scalar_product(points_1, points_2, normals_1, normals_2):
            return torch.dot(normals_1.view(-1),
                             kernel.convolve(points_1, points_2, normals_2, mode='pointcloud').view(-1))

        if target.norm is None:
            target.norm = point_cloud_scalar_product(c2, c2, n2, n2)

        return point_cloud_scalar_product(c1, c1, n1, n1) + target.norm - 2 * point_cloud_scalar_product(c1, c2, n1, n2)

    @staticmethod
    def varifold_distance(points, source, target, kernel):

        """
        Returns the current distance between the 3D meshes
        source and target are SurfaceMesh objects
        points are source points (torch tensor)
        """

        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target)

        # alpha = normales non unitaires
        areaa = torch.norm(n1, 2, 1)
        areab = torch.norm(n2, 2, 1)

        nalpha = n1 / areaa.unsqueeze(1)
        nbeta = n2 / areab.unsqueeze(1)

        def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
            return torch.dot(areaa.view(-1), kernel.convolve((x, nalpha), (y, nbeta), areab.view(-1, 1),
                                                             mode='varifold').view(-1))

        if target.norm is None:
            target.norm = varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta)

        return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) + target.norm \
               - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)

    @staticmethod
    def landmark_distance(points, target):
        """
        Point correspondance distance
        """
        target_points = target.get_points_torch(tensor_scalar_type=points.type())
        return torch.sum((points.view(-1) - target_points.view(-1)) ** 2)

    @staticmethod
    def L2_distance(intensities, target):
        """
        L2 image distance.
        """
        target_intensities = target.get_intensities_torch(tensor_scalar_type=intensities.type())
        return torch.sum((intensities.view(-1) - target_intensities.view(-1)) ** 2)
    
    @staticmethod
    def cross_correlation(intensities, target):
    
        target_intensities = target.get_intensities_torch(tensor_scalar_type=intensities.type())
        
        return 1-f.cosine_similarity(intensities.view(-1), target_intensities.view(-1), dim=0)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def __get_source_and_target_centers_and_normals(points, source, target):
        tensor_scalar_type = points.type()
        tensor_integer_type = {
            'cpu': 'torch.LongTensor',
            'cuda': 'torch.cuda.LongTensor'
        }[points.device.type]

        c1, n1 = source.get_centers_and_normals(points,
                                                tensor_scalar_type=tensor_scalar_type,
                                                tensor_integer_type=tensor_integer_type)
        c2, n2 = target.get_centers_and_normals(tensor_scalar_type=tensor_scalar_type,
                                                tensor_integer_type=tensor_integer_type)
        return c1, n1, c2, n2
