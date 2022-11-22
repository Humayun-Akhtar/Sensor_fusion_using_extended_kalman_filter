# Sensor_fusion_using_extended_kalman_filter
An implementation of Extended Kalman Filter for fusion of lidar and radar sensor data for two different motion models.
# INTRODUCTION
This repository is an implementations of Kalman filter and extended Kalman filter (depeding on the linearatiy of the model) for the following different motion models using LIDAR and Radar measurement.

Process Models:

CV (constant velocity)

CTRV (constant turn rate and velocity magnitude)

The project relies on the Eigen library for vector and matrix operations.

An extended effort has been put in designing abstract of filter, process model, and measurement model. The code heavily relies on python templates. 

# RESOURCES
During the implementaion of different variations of Kalman filters, the notation from the book "Thrun, S., Burgard, W. and Fox, D., 2005. Probabilistic robotics. MIT press." was followed.

# FILTERS AND PROCESS MODELS
 
#### Basic Kalman Filter Algorithm 
![Algorithm_Kalman_filter](https://user-images.githubusercontent.com/115849836/203238318-84710acb-1f19-4586-a56d-6aed5c37e533.png)

The algorithm is directly referred from the book "Thrun, S., Burgard, W. and Fox, D., 2005. Probabilistic robotics. MIT press."

#### Extended Kalman Filter
![Algorithm_Extended_Kalman_filter](https://user-images.githubusercontent.com/115849836/203238535-1854dcbe-ed19-4f7f-bef5-b201178bae6f.png)

The algorithm is directly referred from the book "Thrun, S., Burgard, W. and Fox, D., 2005. Probabilistic robotics. MIT press."

## Process Models

### Constant Turn Rate and Velocity Model

In the constant turn rate and velocity process model the object moves with a constant turn rate and velocity, that is, with zero longitudinal and yaw accelerations. The state vector consists of 5 components---px, py, v, yaw, yaw_rate---where p* represents the position, v represents the velocity , yaw represents the yaw angle, and yaw_rate represents the yaw velocity. The leftmost column in the following equation represents the non-linear process noise; a_a represents longitudinal acceleration, and a_psi is yaw acceleration.
![CTRV_process_model](https://user-images.githubusercontent.com/115849836/203239277-34e20408-08ef-433d-9c28-c9b7b1ebf890.png)
![CTRV_process_model_alpha](https://user-images.githubusercontent.com/115849836/203239306-cb472efb-6588-4e47-a748-445a1011b243.png)
![CTRV_process_model_beta](https://user-images.githubusercontent.com/115849836/203239555-4a818ada-23ca-4ede-9a09-0d2cc4a18d77.png)


### Constant Velocity Model 

The constant velocity process model is a process model where the object moves linearly with constant velocity. The state vector consists of 4 components---px, py, vx, vy---where p* represents the position and v* represents the velocity. The leftmost column in the following equation represents the additive process noise; a* represents acceleration.
![CV_process_model](https://user-images.githubusercontent.com/115849836/203238659-e2dc5b17-a250-4504-b87c-3d9dbf1aa19a.png)

#### Radar Measurement Model 

![Radar_measurement_model_h](https://user-images.githubusercontent.com/115849836/203239390-fd4e1932-1019-402a-ab61-a7e5e0f029b7.png)



# Computing the Covariance, State transition and Jacobian Matrix
Two seperate documentation with handwritten notes to compute various matrices has been attached. Both prcoess models has been refereed:
1. Constant Turn rate and velocity model (Pdf with name constant turn rate model)
2. Constant Velocity Model (Pdf with name Constant velocity Model)
