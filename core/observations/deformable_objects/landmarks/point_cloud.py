import numpy as np
import torch
from torch.autograd import Variable

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark


class PointCloud(Landmark):
    """
    Points in 2D or 3D space, seen as measures
    """

    def __init__(self, dimension):
        super().__init__(dimension)
        self.centers = None
        self.normals = None  # This is going to be point weights, uniform for now TODO: read somewhere e.g. in the vtk the weights of the points.

    def update(self):
        Landmark.update(self)
        # self.get_centers_and_normals()

    def get_centers_and_normals(self, points=None, tensor_scalar_type=default.tensor_scalar_type, **kwargs):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals (which are tangents in this case) and centers
        """
        if points is None or self.centers is None:
            if (self.normals is None) or (self.centers is None):
                self.centers = torch.from_numpy(self.points).type(tensor_scalar_type)
                self.normals = torch.from_numpy(np.array([[1. / len(self.points)]
                                                          for _ in self.points])).type(tensor_scalar_type)
        else:
            self.centers = points
            self.normals = torch.from_numpy(np.array([[1. / len(points)]
                                                      for _ in points])).type(tensor_scalar_type)

        return self.centers, self.normals
