import os.path

import PIL.Image as pimg
import nibabel as nib
import numpy as np
import torch
from torch.autograd import Variable

from core import default
from in_out.image_functions import rescale_image_intensities, points_to_voxels_transform


class Image:
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

        self.type = 'Image'
        self.is_modified = True

        self.affine = None
        self.corner_points = None
        self.bounding_box = None
        self.downsampling_factor = 1

        self.intensities = None  # Numpy array.
        self.intensities_dtype = None

        self.min_intensity = 0
        self.max_intensity = 0

    # Clone.
    def clone(self):
        clone = Image(self.dimension)
        clone.is_modified = True

        clone.affine = np.copy(self.affine)
        clone.corner_points = np.copy(self.corner_points)
        clone.bounding_box = np.copy(self.bounding_box)
        clone.downsampling_factor = self.downsampling_factor

        clone.intensities = np.copy(self.intensities)
        clone.intensities_dtype = self.intensities_dtype

        clone.min_intensity = self.min_intensity
        clone.max_intensity = self.max_intensity

        return clone

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_number_of_points(self):
        raise RuntimeError("Not implemented for Image yet.")

    def set_affine(self, affine_matrix):
        """
        The affine matrix A is a 4x4 matrix that gives the correspondence between the voxel coordinates and their
        spatial positions in the 3D space: (x, y, z, 1) = A (u, v, w, 1).
        See the nibabel documentation for further details (the same attribute name is used here).
        """
        self.affine = affine_matrix

    def set_intensities(self, intensities):
        self.is_modified = True
        self.intensities = intensities

    def set_min_intensity(self, min_intensity):
        self.min_intensity = min_intensity

    def set_max_intensity(self, max_intensity):
        self.max_intensity = max_intensity

    def get_min_intensity(self):
        return self.min_intensity

    def get_max_intensity(self):
        return self.max_intensity

    def set_dtype(self, dtype):
        self.intensities_dtype = dtype

    def get_intensities(self):
        return self.intensities

    def get_intensities_torch(self, tensor_scalar_type=default.tensor_scalar_type):
        return torch.from_numpy(self.intensities).type(tensor_scalar_type)

    def get_points(self):

        image_shape = self.intensities.shape

        axes = []
        for d in range(self.dimension):
            axe = np.linspace(self.corner_points[0, d], self.corner_points[2 ** d, d],
                              image_shape[d] // self.downsampling_factor)
            axes.append(axe)

        points = np.array(np.meshgrid(*axes, indexing='ij')[:])
        for d in range(self.dimension):
            points = np.swapaxes(points, d, d + 1)

        return points

    # @jit(parallel=True)
    def get_deformed_intensities(self, deformed_points, intensities):
        """
        Torch input / output.
        Interpolation function with zero-padding.
        """
        tensor_scalar_type = deformed_points.type()
        tensor_integer_type = {
            'cpu': 'torch.LongTensor',
            'cuda': 'torch.cuda.LongTensor'
        }[deformed_points.device.type]

        image_shape = self.intensities.shape
        deformed_voxels = points_to_voxels_transform(deformed_points, self.affine)

        if self.dimension == 2:

            if not self.downsampling_factor == 1:
                shape = deformed_points.shape
                deformed_voxels = torch.nn.Upsample(size=self.intensities.shape, mode='bilinear', align_corners=True)(
                    deformed_voxels.permute(2, 0, 1).contiguous().view(
                        1, shape[2], shape[0], shape[1]))[0].permute(1, 2, 0).contiguous()

            u, v = deformed_voxels.view(-1, 2)[:, 0], deformed_voxels.view(-1, 2)[:, 1]

            u1 = np.floor(u.data.cpu().numpy()).astype(int)
            v1 = np.floor(v.data.cpu().numpy()).astype(int)

            u1 = np.clip(u1, 0, image_shape[0] - 1)
            v1 = np.clip(v1, 0, image_shape[1] - 1)
            u2 = np.clip(u1 + 1, 0, image_shape[0] - 1)
            v2 = np.clip(v1 + 1, 0, image_shape[1] - 1)

            fu = u - torch.from_numpy(u1).type(tensor_scalar_type)
            fv = v - torch.from_numpy(v1).type(tensor_scalar_type)
            gu = torch.from_numpy(u1 + 1).type(tensor_scalar_type) - u
            gv = torch.from_numpy(v1 + 1).type(tensor_scalar_type) - v

            deformed_intensities = (intensities[u1, v1] * gu * gv +
                                    intensities[u1, v2] * gu * fv +
                                    intensities[u2, v1] * fu * gv +
                                    intensities[u2, v2] * fu * fv).view(image_shape)

        elif self.dimension == 3:

            if not self.downsampling_factor == 1:
                shape = deformed_points.shape
                deformed_voxels = torch.nn.Upsample(size=self.intensities.shape, mode='trilinear', align_corners=True)(
                    deformed_voxels.permute(3, 0, 1, 2).contiguous().view(
                        1, shape[3], shape[0], shape[1], shape[2]))[0].permute(1, 2, 3, 0).contiguous()

            u, v, w = deformed_voxels.view(-1, 3)[:, 0], \
                      deformed_voxels.view(-1, 3)[:, 1], \
                      deformed_voxels.view(-1, 3)[:, 2]

            u1_numpy = np.floor(u.data.cpu().numpy()).astype(int)
            v1_numpy = np.floor(v.data.cpu().numpy()).astype(int)
            w1_numpy = np.floor(w.data.cpu().numpy()).astype(int)

            u1 = torch.from_numpy(np.clip(u1_numpy, 0, image_shape[0] - 1)).type(tensor_integer_type)
            v1 = torch.from_numpy(np.clip(v1_numpy, 0, image_shape[1] - 1)).type(tensor_integer_type)
            w1 = torch.from_numpy(np.clip(w1_numpy, 0, image_shape[2] - 1)).type(tensor_integer_type)
            u2 = torch.from_numpy(np.clip(u1_numpy + 1, 0, image_shape[0] - 1)).type(tensor_integer_type)
            v2 = torch.from_numpy(np.clip(v1_numpy + 1, 0, image_shape[1] - 1)).type(tensor_integer_type)
            w2 = torch.from_numpy(np.clip(w1_numpy + 1, 0, image_shape[2] - 1)).type(tensor_integer_type)

            fu = u - Variable(torch.from_numpy(u1_numpy).type(tensor_scalar_type))
            fv = v - Variable(torch.from_numpy(v1_numpy).type(tensor_scalar_type))
            fw = w - Variable(torch.from_numpy(w1_numpy).type(tensor_scalar_type))
            gu = Variable(torch.from_numpy(u1_numpy + 1).type(tensor_scalar_type)) - u
            gv = Variable(torch.from_numpy(v1_numpy + 1).type(tensor_scalar_type)) - v
            gw = Variable(torch.from_numpy(w1_numpy + 1).type(tensor_scalar_type)) - w

            deformed_intensities = (intensities[u1, v1, w1] * gu * gv * gw +
                                    intensities[u1, v1, w2] * gu * gv * fw +
                                    intensities[u1, v2, w1] * gu * fv * gw +
                                    intensities[u1, v2, w2] * gu * fv * fw +
                                    intensities[u2, v1, w1] * fu * gv * gw +
                                    intensities[u2, v1, w2] * fu * gv * fw +
                                    intensities[u2, v2, w1] * fu * fv * gw +
                                    intensities[u2, v2, w2] * fu * fv * fw).view(image_shape)

        else:
            raise RuntimeError('Incorrect dimension of the ambient space: %d' % self.dimension)

        return deformed_intensities

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self._update_corner_point_positions()
            self.update_bounding_box()
            self.is_modified = False

    def update_bounding_box(self):
        """
        Compute a tight bounding box that contains all the 2/3D-embedded image data.
        """
        self.bounding_box = np.zeros((self.dimension, 2))
        for d in range(self.dimension):
            self.bounding_box[d, 0] = np.min(self.corner_points[:, d])
            self.bounding_box[d, 1] = np.max(self.corner_points[:, d])

    def write(self, output_dir, name, intensities=None, should_rescale=True):

        if intensities is None:
            intensities = self.get_intensities()

        should_rescale = False

        if (should_rescale):
            intensities_rescaled = rescale_image_intensities(intensities, self.intensities_dtype)
        else:
            intensities_rescaled = intensities

        if name.find(".png") > 0 or name.find(".tif") > 0:
            pimg.fromarray(intensities_rescaled).save(os.path.join(output_dir, name))
        elif name.find(".nii") > 0:
            img = nib.Nifti1Image(intensities_rescaled, self.affine)
            nib.save(img, os.path.join(output_dir, name))
        elif name.find(".npy") > 0:
            np.save(os.path.join(output_dir, name), intensities_rescaled)
        else:
            raise ValueError('Writing images with the given extension "%s" is not coded yet.' % name)

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def _update_corner_point_positions(self):
        if self.dimension == 2:
            corner_points = np.zeros((4, 2))
            umax, vmax = np.subtract(self.intensities.shape, (1, 1))
            corner_points[0] = np.array([0, 0])
            corner_points[1] = np.array([umax, 0])
            corner_points[2] = np.array([0, vmax])
            corner_points[3] = np.array([umax, vmax])

        elif self.dimension == 3:
            corner_points = np.zeros((8, 3))
            umax, vmax, wmax = np.subtract(self.intensities.shape, (1, 1, 1))
            corner_points[0] = np.array([0, 0, 0])
            corner_points[1] = np.array([umax, 0, 0])
            corner_points[2] = np.array([0, vmax, 0])
            corner_points[3] = np.array([umax, vmax, 0])
            corner_points[4] = np.array([0, 0, wmax])
            corner_points[5] = np.array([umax, 0, wmax])
            corner_points[6] = np.array([0, vmax, wmax])
            corner_points[7] = np.array([umax, vmax, wmax])

        #################################
        # VERSION FOR IMAGE + MESH DATA #
        #################################
        # if self.dimension == 2:
        #     corner_points = np.zeros((4, 2))
        #     umax, vmax = np.subtract(self.intensities.shape, (1, 1))
        #     corner_points[0] = np.dot(self.affine[0:2, 0:2], np.array([0, 0])) + self.affine[0:2, 2]
        #     corner_points[1] = np.dot(self.affine[0:2, 0:2], np.array([umax, 0])) + self.affine[0:2, 2]
        #     corner_points[2] = np.dot(self.affine[0:2, 0:2], np.array([0, vmax])) + self.affine[0:2, 2]
        #     corner_points[3] = np.dot(self.affine[0:2, 0:2], np.array([umax, vmax])) + self.affine[0:2, 2]
        #
        # elif self.dimension == 3:
        #     corner_points = np.zeros((8, 3))
        #     umax, vmax, wmax = np.subtract(self.intensities.shape, (1, 1, 1))
        #     corner_points[0] = np.dot(self.affine[0:3, 0:3], np.array([0, 0, 0])) + self.affine[0:3, 3]
        #     corner_points[1] = np.dot(self.affine[0:3, 0:3], np.array([umax, 0, 0])) + self.affine[0:3, 3]
        #     corner_points[2] = np.dot(self.affine[0:3, 0:3], np.array([0, vmax, 0])) + self.affine[0:3, 3]
        #     corner_points[3] = np.dot(self.affine[0:3, 0:3], np.array([umax, vmax, 0])) + self.affine[0:3, 3]
        #     corner_points[4] = np.dot(self.affine[0:3, 0:3], np.array([0, 0, wmax])) + self.affine[0:3, 3]
        #     corner_points[5] = np.dot(self.affine[0:3, 0:3], np.array([umax, 0, wmax])) + self.affine[0:3, 3]
        #     corner_points[6] = np.dot(self.affine[0:3, 0:3], np.array([0, vmax, wmax])) + self.affine[0:3, 3]
        #     corner_points[7] = np.dot(self.affine[0:3, 0:3], np.array([umax, vmax, wmax])) + self.affine[0:3, 3]

        else:
            raise RuntimeError('Invalid dimension: %d' % self.dimension)

        self.corner_points = corner_points
