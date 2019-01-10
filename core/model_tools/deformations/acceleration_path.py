import warnings

import torch

from core import default
from core.model_tools.deformations.acceleration_integrate import AccelerationIntegrate
from in_out.array_readers_and_writers import *


class AccelerationPath:
    """
    Acceleration controlled path on LDDMM diffeomorphic manifold
    See "Estimation of smooth growth trajectories with controlled acceleration from time series shape data",
    Fishbaugh et al. (2011).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=default.deformation_kernel, shoot_kernel_type=None,
                 number_of_time_points=default.number_of_time_points):

        self.number_of_time_points = number_of_time_points
        self.tmax = None
        self.tmin = None

        self.control_points_tmin = None
        self.impulse_t = None
        self.template_points_tmin = None
        
        self.initial_velocity = None
        
        self.integrator = AccelerationIntegrate(kernel=kernel, shoot_kernel_type=shoot_kernel_type)


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_kernel_type(self):
        return self.integrator.get_kernel_type()

    def set_kernel(self, kernel):
        self.integrator.kernel = kernel

    def get_tmin(self):
        return self.tmin

    def set_tmin(self, tmin):
        self.tmin = tmin

    def get_tmax(self):
        return self.tmax

    def set_tmax(self, tmax):
        self.tmax = tmax

    def get_template_points_tmin(self):
        return self.template_points_tmin

    def set_template_points_tmin(self, td):
        self.template_points_tmin = td

    def set_control_points_tmin(self, cp):
        self.control_points_tmin = cp
        
    def set_initial_velocity(self, initial_velocity):
        self.initial_velocity = initial_velocity
        
    def get_initial_velocity(self):
        return self.initial_velocity

    def set_impulse_t(self, impulse_t):
        self.impulse_t = impulse_t
        
    def get_impulse_t(self):
        return self.impulse_t
        
    def get_template_points(self, time):
        """
        Returns the position of the landmark points, at the given time.
        Performs a linear interpolation between the two closest available data points.
        """

        #assert self.tmin <= time <= self.tmax
        
        times = self._get_times()

        # Standard case.
        for j in range(1, len(times)):
            if time - times[j] < 0: break

        weight_left = torch.Tensor([(times[j] - time) / (times[j] - times[j - 1])]).type(self.impulse_t.type())
        weight_right = torch.Tensor([(time - times[j - 1]) / (times[j] - times[j - 1])]).type(self.impulse_t.type())
        template_t = self._get_template_points_trajectory()
        deformed_points = {key: weight_left * value[j - 1] + weight_right * value[j]
                           for key, value in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """
        
        self.integrator.number_of_time_points = self.number_of_time_points
        self.integrator.set_impulse_t(self.impulse_t)
        self.integrator.set_initial_velocity(self.initial_velocity)
        self.integrator.set_initial_control_points(self.control_points_tmin)
        self.integrator.set_initial_template_points(self.template_points_tmin)
        self.integrator.update()
        

    def get_norm_squared(self):
        """
        Get the norm of the acceleration path
        """
        return self.integrator.get_norm_squared()

    def get_velocity_norm(self):
    
        return self.integrator.get_velocity_norm_squared()


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _get_times(self):
        return np.linspace(self.tmin, self.tmax, self.integrator.number_of_time_points).tolist()

    def _get_template_points_trajectory(self):
        
        template_t = {}
        for key in self.template_points_tmin.keys():

            template_t[key] = self.integrator.template_points_t[key]

        return template_t

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write2(self, root_name, objects_name, objects_extension, template, template_data, slope_image, output_dir, write_adjoint_parameters=False):

        # Core loop ----------------------------------------------------------------------------------------------------
        times = self._get_times()
        for t, time in enumerate(times):
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                #name = root_name + '__AccelerationFlow__' + object_name + '__tp_' + str(t) + ('__age_%.2f' % time) + object_extension
                name = '%s__AccelerationFlow__%s__%0.03d%s' %(root_name, object_name, t, object_extension)
                print(name)
                names.append(name)
            deformed_points = self.get_template_points(time)
            if slope_image is not None:
                linear_image_model = {}
                linear_image_model['image_intensities'] = slope_image*time + template_data['image_intensities']
                deformed_data = template.get_deformed_data(deformed_points, linear_image_model)
            else:
                deformed_data = template.get_deformed_data(deformed_points, template_data)
            template.write(output_dir, names, {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

    def write(self, root_name, objects_name, objects_extension, template, template_data, A, B, C,
              output_dir, write_adjoint_parameters=False):

        # Core loop ----------------------------------------------------------------------------------------------------
        times = self._get_times()
        for t, time in enumerate(times):
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                # name = root_name + '__AccelerationFlow__' + object_name + '__tp_' + str(t) + ('__age_%.2f' % time) + object_extension
                name = '%s__AccelerationFlow__%s__%0.03d%s' % (root_name, object_name, t, object_extension)
                print(name)
                names.append(name)
            deformed_points = self.get_template_points(time)
            linear_image_model = {}
            linear_image_model['image_intensities'] = A * torch.exp(-B * torch.exp(-C * time))
            deformed_data = template.get_deformed_data(deformed_points, linear_image_model)

            template.write(output_dir, names, {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

