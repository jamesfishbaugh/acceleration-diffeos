import warnings
from copy import deepcopy
import support.kernels as kernel_factory
import torch

from core import default
from in_out.array_readers_and_writers import *

import torch.nn.functional as f

import logging

logger = logging.getLogger(__name__)


class AccelerationIntegrate:
    """
    Acceleration controlled flow of diffeomorphisms, that transforms the template objects according time varying point force vectors
    See "Estimation of smooth growth trajectories with controlled acceleration from time series shape data",
    Fishbaugh et al. (2011).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=default.deformation_kernel, shoot_kernel_type=None, number_of_time_points=None,
                 initial_control_points=None, control_points_t=None, impulse_t=None, initial_velocity=None,
                 initial_template_points=None, template_points_t=None):

        self.kernel = kernel

        if shoot_kernel_type is not None:
            self.shoot_kernel = kernel_factory.factory(shoot_kernel_type, kernel_width=kernel.kernel_width, device=kernel.device)
        else:
            self.shoot_kernel = self.kernel

        self.number_of_time_points = number_of_time_points
        # Initial position of control points
        self.initial_control_points = initial_control_points
        # Control points trajectory
        self.control_points_t = control_points_t
        # Momenta trajectory
        self.impulse_t = impulse_t
        self.initial_velocity = initial_velocity
        
        # Initial template points
        self.initial_template_points = initial_template_points
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_points_t = template_points_t
        
    def light_copy(self):
        light_copy = AccelerationIntegrate(deepcopy(self.kernel),
                                 self.number_of_time_points,
                                 self.initial_control_points, self.control_points_t,
                                 self.impulse_t, self.initial_velocity,
                                 self.initial_template_points, self.template_points_t)
        return light_copy

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_kernel_type(self):
        return self.kernel.kernel_type

    def get_kernel_width(self):
        return self.kernel.kernel_width

    def set_kernel(self, kernel):
        # TODO which kernel to set ?
        self.kernel = kernel

    def set_initial_template_points(self, td):
        self.initial_template_points = td

    def get_initial_template_points(self):
        return self.initial_template_points

    def set_initial_control_points(self, cps):
        self.initial_control_points = cps

    def get_initial_control_points(self):
        return self.initial_control_points
        
    def set_impulse_t(self, impulse_t):
        self.impulse_t = impulse_t

    def get_impulse_t(self):
        return self.impulse_t

    def set_initial_velocity(self, initial_velocity):
        self.initial_velocity = initial_velocity
        
    def get_initial_velocity(self):
        return self.initial_velocity

    def scalar_product(self, cp, impulse1, impulse2):
        """
        returns the scalar product 'impulse1 K(cp) impulse2 2'
        """
        return torch.sum(impulse1 * self.kernel.convolve(cp, cp, impulse2))
        
    def get_template_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the trajectory
        """
        if time_index is None:
            return {key: self.template_points_t[key][-1] for key in self.initial_template_points.keys()}
        return {key: self.template_points_t[key][time_index] for key in self.initial_template_points.keys()}

    def get_norm_squared(self):
        
        total_scalar_product = 0.0
                
        for i in range(0, self.number_of_time_points):
        
          total_scalar_product = total_scalar_product + self.scalar_product(self.initial_control_points, self.impulse_t[i,:,:], self.impulse_t[i,:,:])
        
        return total_scalar_product
        
    def get_velocity_norm_squared(self):
        
        return self.scalar_product(self.initial_control_points, self.initial_velocity, self.initial_velocity)
    
    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call flow on the deformation.
        """
        assert self.number_of_time_points > 0
        
        if self.initial_template_points is not None:
            self.flow()
        else:
            msg = "In AccelerationIntegrate update, I am not flowing because I don't have any template points to flow"
            logger.warning(msg)

    def flow(self):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        
        useCuda = True

        cuda0 = torch.device('cuda:0')
        
        # Initialization.
        dt = 1.0 / float(self.number_of_time_points - 1)
        dt2 = dt*dt
        self.template_points_t = {}
        [number_of_control_points, dimension] = self.initial_control_points.shape
        number_of_shape_points = 0
        if 'landmark_points' in self.initial_template_points.keys():
          [number_of_shape_points, dimension] = self.initial_template_points['landmark_points'].shape
          if (useCuda):
            zero_velocity = torch.zeros(number_of_shape_points, dimension, dtype=torch.float, device=cuda0)
          else:
            zero_velocity = torch.zeros(number_of_shape_points, dimension, dtype=torch.float)
        if 'image_points' in self.initial_template_points.keys():
          if dimension == 2:
            [rows, cols, d] = self.initial_template_points['image_points'].size()
            if (useCuda):
              zero_velocity = torch.zeros(rows, cols, d, dtype=torch.float, device=cuda0)
            else:
              zero_velocity = torch.zeros(rows, cols, d, dtype=torch.float)
          elif dimension == 3:
            [x, y, z, d] = self.initial_template_points['image_points'].size()
            if (useCuda):
              zero_velocity = torch.zeros(x, y, z, d, dtype=torch.float, device=cuda0)
            else:
              print("NOT USING CUDA")
              zero_velocity = torch.zeros(x, y, z, d, dtype=torch.float)
        
        velocity_t = []
                        
        if (len(self.initial_velocity) == 0):
          velocity_t.append(zero_velocity)
        else:
          velocity_t.append(self.initial_velocity)
        
        init_velocity_control_points = torch.zeros(number_of_control_points, dimension, dtype=torch.float, device=cuda0)
        velocity_control_points_t = []
        velocity_control_points_t.append(init_velocity_control_points)
        
        control_points_t = []
        control_points_t.append(self.initial_control_points)
        
        # Flow landmarks points.
        if 'landmark_points' in self.initial_template_points.keys():
        
            landmark_points = [self.initial_template_points['landmark_points']]

            # Loop over time
            for i in range(0, self.number_of_time_points - 1):
               
                # Compute acceleration at the landmark points
                accel = self.kernel.convolve(landmark_points[i], control_points_t[i], self.impulse_t[i,:,:])
                
                # Compute acceleration at the control points
                accel_ctrlpoints = self.kernel.convolve(control_points_t[i], control_points_t[i], self.impulse_t[i,:,:])
                control_points_tplus1 = control_points_t[i] + velocity_t[i]*dt + 0.5*accel_ctrlpoints*dt2
                
                # Compute velocity at the landmark points
                velocity_at_landmarkpoints = self.kernel.convolve(landmark_points[i], control_points_t[i], velocity_t[i])
                
                # Use acceleration to update the positions
                landmark_points_tplus1 = landmark_points[i] + velocity_at_landmarkpoints*dt + 0.5*accel*dt2
                
                # Compute acceleration at landmark points at time t+1
                accel_tplus1 = self.kernel.convolve(landmark_points_tplus1, control_points_tplus1, self.impulse_t[i+1,:,:])
                
                # Compute acceleration at the control points at time t+1
                accel_tplus1_ctrlpoints = self.kernel.convolve(control_points_tplus1, control_points_tplus1, self.impulse_t[i+1,:,:])
                velocity_control_points = velocity_t[i] + 0.5*(accel_ctrlpoints + accel_tplus1_ctrlpoints)*dt
                                
                # Append everything
                landmark_points.append(landmark_points_tplus1)
                velocity_t.append(velocity_control_points)
                control_points_t.append(control_points_t[i])
                
            self.template_points_t['landmark_points'] = landmark_points

        # Flow image points.
        if 'image_points' in self.initial_template_points.keys():
        
            image_points = [self.initial_template_points['image_points']]

            dimension = self.initial_control_points.size(1)
            image_shape = image_points[0].size()

            for i in range(0, self.number_of_time_points - 1):

                # Compute acceleration at the image points
                accel = self.kernel.convolve(image_points[i].contiguous().view(-1, dimension), control_points_t[i], self.impulse_t[i,:,:]).view(image_shape)
                dY = self._compute_image_explicit_euler_step_at_order_1(image_points[i], accel)
                
                # Compute acceleration at the control points
                accel_ctrlpoints = self.kernel.convolve(control_points_t[i], control_points_t[i], self.impulse_t[i,:,:])
                control_points_tplus1 = control_points_t[i] + velocity_t[i]*dt + 0.5*accel_ctrlpoints*dt2
                
                # Compute velocity at the image points
                velocity_at_imagepoints = self.kernel.convolve(image_points[i].contiguous().view(-1, dimension), control_points_t[i], velocity_t[i]).view(image_shape)
                dVelocity_at_imagepoints = self._compute_image_explicit_euler_step_at_order_1(image_points[i], velocity_at_imagepoints)
                                
                # Compute new image points at time t+1
                image_points_tplus1 = image_points[i] + dVelocity_at_imagepoints*dt + 0.5*dY*dt2
                
                # Compute acceleration at the image points at time t+1
                accel_tplus1 = self.kernel.convolve(image_points_tplus1.contiguous().view(-1, dimension), control_points_tplus1, self.impulse_t[i+1,:,:]).view(image_shape)
                dY2 = self._compute_image_explicit_euler_step_at_order_1(image_points_tplus1, accel_tplus1)
                
                # Compute acceleration at the control points at time t+1
                accel_tplus1_ctrlpoints = self.kernel.convolve(control_points_tplus1, control_points_tplus1, self.impulse_t[i+1,:,:])
                velocity_control_points = velocity_t[i] + 0.5*(accel_ctrlpoints + accel_tplus1_ctrlpoints)*dt
                
                # Append everything
                image_points.append(image_points_tplus1)
                velocity_t.append(velocity_control_points)
                control_points_t.append(control_points_tplus1)
                
            #    vf = self.kernel.convolve(image_points[0].contiguous().view(-1, dimension), self.control_points_t[i],
            #                              self.momenta_t[i]).view(image_shape)
            #    dY = self._compute_image_explicit_euler_step_at_order_1(image_points[i], vf)
            #    image_points.append(image_points[i] - dt * dY)

            #if self.use_rk2_for_flow:
            #    msg = 'RK2 not implemented to flow image points.'
            #    logger.warning(msg)

            self.template_points_t['image_points'] = image_points

        assert len(self.template_points_t) > 0, 'That\'s unexpected'

 
    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    @staticmethod
    def _euler_step(kernel, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        return cp + h * kernel.convolve(cp, cp, mom), mom - h * kernel.convolve_gradient(mom, cp)

    @staticmethod
    def _rk2_step(kernel, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        mid_cp = cp + h / 2. * kernel.convolve(cp, cp, mom)
        mid_mom = mom - h / 2. * kernel.convolve_gradient(mom, cp)
        if return_mom:
            return cp + h * kernel.convolve(mid_cp, mid_cp, mid_mom), mom - h * kernel.convolve_gradient(mid_mom,
                                                                                                         mid_cp)
        else:
            return cp + h * kernel.convolve(mid_cp, mid_cp, mid_mom)

    # TODO. Wrap pytorch of an efficient C code ? Use keops ? Called ApplyH in PyCa. Check Numba as well.
    # @staticmethod
    # @jit(parallel=True)
    def _compute_image_explicit_euler_step_at_order_1(self, Y, vf):
        dY = torch.zeros(Y.shape).type(vf.type())
        dimension = len(Y.shape) - 1

        if dimension == 2:
            ni, nj = Y.shape[:2]

            # Center.
            dY[1:ni - 1, :] = dY[1:ni - 1, :] + 0.5 * vf[1:ni - 1, :, 0] \
                .contiguous().view(ni - 2, nj, 1).expand(ni - 2, nj, 2) * (Y[2:ni, :] - Y[0:ni - 2, :])
            dY[:, 1:nj - 1] = dY[:, 1:nj - 1] + 0.5 * vf[:, 1:nj - 1, 1] \
                .contiguous().view(ni, nj - 2, 1).expand(ni, nj - 2, 2) * (Y[:, 2:nj] - Y[:, 0:nj - 2])

            # Borders.
            dY[0, :] = dY[0, :] + vf[0, :, 0].contiguous().view(nj, 1).expand(nj, 2) * (Y[1, :] - Y[0, :])
            dY[ni - 1, :] = dY[ni - 1, :] + vf[ni - 1, :, 0].contiguous().view(nj, 1).expand(nj, 2) \
                                            * (Y[ni - 1, :] - Y[ni - 2, :])

            dY[:, 0] = dY[:, 0] + vf[:, 0, 1].contiguous().view(ni, 1).expand(ni, 2) * (Y[:, 1] - Y[:, 0])
            dY[:, nj - 1] = dY[:, nj - 1] + vf[:, nj - 1, 1].contiguous().view(ni, 1).expand(ni, 2) \
                                            * (Y[:, nj - 1] - Y[:, nj - 2])

        elif dimension == 3:

            ni, nj, nk = Y.shape[:3]

            # Center.
            dY[1:ni - 1, :, :] = dY[1:ni - 1, :, :] + 0.5 * vf[1:ni - 1, :, :, 0] \
                .contiguous().view(ni - 2, nj, nk, 1).expand(ni - 2, nj, nk, 3) * (Y[2:ni, :, :] - Y[0:ni - 2, :, :])
            dY[:, 1:nj - 1, :] = dY[:, 1:nj - 1, :] + 0.5 * vf[:, 1:nj - 1, :, 1] \
                .contiguous().view(ni, nj - 2, nk, 1).expand(ni, nj - 2, nk, 3) * (Y[:, 2:nj, :] - Y[:, 0:nj - 2, :])
            dY[:, :, 1:nk - 1] = dY[:, :, 1:nk - 1] + 0.5 * vf[:, :, 1:nk - 1, 2] \
                .contiguous().view(ni, nj, nk - 2, 1).expand(ni, nj, nk - 2, 3) * (Y[:, :, 2:nk] - Y[:, :, 0:nk - 2])

            # Borders.
            dY[0, :, :] = dY[0, :, :] + vf[0, :, :, 0].contiguous().view(nj, nk, 1).expand(nj, nk, 3) \
                                        * (Y[1, :, :] - Y[0, :, :])
            dY[ni - 1, :, :] = dY[ni - 1, :, :] + vf[ni - 1, :, :, 0].contiguous().view(nj, nk, 1).expand(nj, nk, 3) \
                                                  * (Y[ni - 1, :, :] - Y[ni - 2, :, :])

            dY[:, 0, :] = dY[:, 0, :] + vf[:, 0, :, 1].contiguous().view(ni, nk, 1).expand(ni, nk, 3) \
                                        * (Y[:, 1, :] - Y[:, 0, :])
            dY[:, nj - 1, :] = dY[:, nj - 1, :] + vf[:, nj - 1, :, 1].contiguous().view(ni, nk, 1).expand(ni, nk, 3) \
                                                  * (Y[:, nj - 1, :] - Y[:, nj - 2, :])

            dY[:, :, 0] = dY[:, :, 0] + vf[:, :, 0, 2].contiguous().view(ni, nj, 1).expand(ni, nj, 3) \
                                        * (Y[:, :, 1] - Y[:, :, 0])
            dY[:, :, nk - 1] = dY[:, :, nk - 1] + vf[:, :, nk - 1, 2].contiguous().view(ni, nj, 1).expand(ni, nj, 3) \
                                                  * (Y[:, :, nk - 1] - Y[:, :, nk - 2])

        else:
            raise RuntimeError('Invalid dimension of the ambient space: %d' % dimension)

        return dY
