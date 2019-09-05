# acceleration-diffeos

Nonparametric image and shape (2D or 3D) regression by acceleration controlled diffeomorphisms. If you use this software for research, please cite the following papers.

* Fishbaugh, J. and Gerig, G. *Acceleration Controlled Diffeomorphisms For Nonparametric Image Regression*. IEEE ISBI. 2019
* Fishbaugh, J., Durrleman, S., Gerig, G. *Estimation of smooth growth trajectories with controlled acceleration from time series shape data*. MICCAI. 2011.

**Requirements**

This software is built from the source code of **Deformetrica**. It is recommended you install deformetrica following the directions at http://www.deformetrica.org, which will install necessary dependencies.* 

**Running the application**

The application is called with the command:

* <span style="color:red">acceleration_diffeos.py estimate model.xml data_set.xml --p optimization_parameters.xml</span>

where

* <span style="color:red">*model.xml*</span> contains information about the template (baseline) shape as well as hyper-parameters for the deformation model.
* <span style="color:red">*data_set.xml*</span> contains the paths to the input objects which are the observed data.
* <span style="color:red">*optimization_parameters.xml*</span> contains optional details about optimization.
