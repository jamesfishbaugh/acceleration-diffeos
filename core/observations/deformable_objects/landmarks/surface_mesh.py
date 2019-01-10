import numpy as np
import torch
from torch.autograd import Variable

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dimension):
        Landmark.__init__(self, dimension)
        self.type = 'SurfaceMesh'

        self.connectivity = None
        # All of these are torch tensor attributes.
        self.centers = None
        self.normals = None

    # Clone.
    def clone(self):
        clone = SurfaceMesh(self.dimension)
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        clone.connectivity = np.copy(self.connectivity)
        return clone

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        # self.get_centers_and_normals()
        Landmark.update(self)

    def get_centers_and_normals(self, points=None,
                                tensor_integer_type=default.tensor_integer_type,
                                tensor_scalar_type=default.tensor_scalar_type):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        connectivity_torch = torch.from_numpy(self.connectivity).type(tensor_integer_type)
        if points is None:
            if self.is_modified or self.centers is None:
                torch_points_coordinates = Variable(torch.from_numpy(self.points).type(tensor_scalar_type))
                a = torch_points_coordinates[connectivity_torch[:, 0]]
                b = torch_points_coordinates[connectivity_torch[:, 1]]
                c = torch_points_coordinates[connectivity_torch[:, 2]]
                centers = (a+b+c)/3.
                self.centers = centers
                self.normals = torch.cross(b-a, c-a)/2
        else:
            a = points[connectivity_torch[:, 0]]
            b = points[connectivity_torch[:, 1]]
            c = points[connectivity_torch[:, 2]]
            centers = (a+b+c)/3.
            self.centers = centers
            self.normals = torch.cross(b-a, c-a)/2
        return self.centers, self.normals
