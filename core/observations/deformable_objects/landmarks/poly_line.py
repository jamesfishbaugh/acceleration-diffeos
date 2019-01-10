import numpy as np
import torch
from torch.autograd import Variable

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark


class PolyLine(Landmark):
    """
    Lines in 2D or 3D space
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dimension):
        Landmark.__init__(self, dimension)
        self.type = 'PolyLine'

        self.connectivity = None
        # All these attributes are torch tensors.
        self.centers = None
        self.normals = None

    # Clone.
    def clone(self):
        clone = PolyLine(self.dimension,)
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
        to compute the new normals (which are tangents in this case) and centers
        It's also a lazy initialization of those attributes !
        """
        connectivity_torch = torch.from_numpy(self.connectivity).type(tensor_integer_type)
        if points is None:
            if self.is_modified or self.centers is None:
                torch_points_coordinates = Variable(torch.from_numpy(self.points).type(tensor_scalar_type))
                a = torch_points_coordinates[connectivity_torch[:, 0]]
                b = torch_points_coordinates[connectivity_torch[:, 1]]
                centers = (a+b)/2.
                self.centers = centers
                self.normals = b - a
        else:
            a = points[connectivity_torch[:, 0]]
            b = points[connectivity_torch[:, 1]]
            centers = (a+b)/2.
            self.centers = centers
            self.normals = b - a
        return self.centers, self.normals
