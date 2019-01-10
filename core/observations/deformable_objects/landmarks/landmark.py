import os.path

import numpy as np
import torch
from torch.autograd import Variable

from core import default


class Landmark:
    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self, dimension):
        assert dimension is not None, 'dimension can not be None'
        self.dimension = dimension

        self.type = 'Landmark'
        self.points = None  # Numpy array.
        self.is_modified = True
        self.bounding_box = None
        self.norm = None
        self.connectivity = None

    # Clone.
    def clone(self):
        clone = Landmark(self.dimension)
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        return clone

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_number_of_points(self):
        return len(self.points)

    def set_points(self, points):
        """
        Sets the list of points of the poly data, to save at the end.
        """
        self.is_modified = True
        self.points = points

    def set_connectivity(self, connectivity):
        self.connectivity = connectivity
        self.is_modified = True

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def get_points(self):
        return self.points

    def get_points_torch(self, tensor_scalar_type=default.tensor_scalar_type):
        return Variable(torch.from_numpy(self.points).type(tensor_scalar_type))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self.update_bounding_box()
            self.is_modified = False

    # Compute a tight bounding box that contains all the landmark data.
    def update_bounding_box(self):
        self.bounding_box = np.zeros((self.dimension, 2))
        for d in range(self.dimension):
            self.bounding_box[d, 0] = np.min(self.points[:, d])
            self.bounding_box[d, 1] = np.max(self.points[:, d])

    def write(self, output_dir, name, points=None):
        connec_names = {2: 'LINES', 3: 'POLYGONS'}
        if points is None:
            points = self.points

        with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {} float\n'.format(len(self.points))
            f.write(s)
            for p in points:
                str_p = [str(elt) for elt in p]
                if len(p) == 2:
                    str_p.append(str(0.))
                s = ' '.join(str_p) + '\n'
                f.write(s)

            if self.connectivity is not None:
                a, connec_degree = self.connectivity.shape
                s = connec_names[connec_degree] + ' {} {}\n'.format(a, a * (connec_degree+1))
                f.write(s)
                for face in self.connectivity:
                    s = str(connec_degree) + ' ' + ' '.join([str(elt) for elt in face]) + '\n'
                    f.write(s)
