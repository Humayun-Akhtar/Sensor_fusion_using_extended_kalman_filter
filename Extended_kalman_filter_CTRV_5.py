#!/usr/bin/env python
# coding: utf-8

# In[94]:


#**************Importing Required Libraries*************
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

# from sympy import init_printing
# init_printing(use_latex=True)
import math as math


# In[95]:





# In[420]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# In[421]:


def cart_to_polar(x_pos,y_pos,vx,vy):
    r = np.sqrt(x_pos**2+y_pos**2)
    phi = math.atan2(y_pos,x_pos)
    r_dot = (vx*x_pos + vy*y_pos)/r
    
    return(r,phi,r_dot)

def state_vector_to_scalars(state_vector):
    '''
    Returns the elements from the state_vector as a tuple of scalars.
    '''
    return (state_vector[0][0,0],state_vector[1][0,0],state_vector[2][0,0],state_vector[3][0,0],state_vector[4][0,0])

def cart_2_polar(state_vector):
    '''
    Transforms the state vector into the polar space.
    '''

    px,py,theta,v,theta_dot = state_vector_to_scalars(state_vector)
    ro      = np.sqrt(px**2 + py**2)

    phi     = np.arctan2(py,px)
    ro_dot  = (px*v*np.cos(theta) + py*v*np.sin(theta))/ro

    return np.matrix([ro, phi, ro_dot]).T

def polar_2_cart(ro, phi, ro_dot):
    '''
    ro: range
    phi: bearing
    ro_dot: range rate
    Takes the polar coord based radar reading and convert to cart coord x,y, v
    return (x,y, v)
    '''
    return (np.cos(phi) * ro, np.sin(phi) * ro, np.sqrt((ro_dot * np.cos(phi))**2 + (ro_dot * np.sin(phi))**2))



########### Initializing the state vector X ###############

#*************Declare Variables**************************
#Read Input File
measurements = pd.read_csv('data.csv',header = None)


# In[422]:


measurements


# In[98]:


measurements_lidar = measurements[measurements[0] == 'L']


# In[99]:


measurements_radar = measurements[measurements[0] == 'R']


# In[100]:


columns_lidar = ['sensor','x_pos','y_pos','time','gt_x_pos','gt_y_pos','gt_vx','gt_vy','gt_yaw','gt_yaw_rate','blank']


# In[101]:


columns_radar = ['sensor','r','heading','r_dot','time','gt_x_pos','gt_y_pos','gt_vx','gt_vy','gt_yaw','gt_yaw_rate']


# In[102]:


measurements_radar.columns = columns_radar


# In[103]:


measurements_radar


# In[104]:


measurements_radar['gt_r'] = np.sqrt(measurements_radar['gt_x_pos']**2 + measurements_radar['gt_y_pos']**2)


# In[105]:


measurements_radar['gt_heading'] = np.arctan2(measurements_radar['gt_y_pos'],measurements_radar['gt_x_pos'])


# In[106]:


measurements_radar['gt_r_dot'] =(measurements_radar['gt_vx']*measurements_radar['gt_x_pos'] + measurements_radar['gt_vy']*measurements_radar['gt_y_pos'])/measurements_radar['gt_r']


# In[474]:


measurements_radar['x_measured'], measurements_radar['y_measured'], measurements_radar['v_measured'] = polar_2_cart(measurements_radar['r'],measurements_radar['heading'],measurements_radar['r_dot'])


# In[475]:


measurements_radar


# In[109]:


measurements_lidar.columns = columns_lidar


if measurements[0][0] == 'L':
    prv_time = measurements[3][0]/1000000.0
    x, y, theta, v , theta_dot = measurements[1][0],measurements[2][0],0,0,0.006911322
elif measurements[0][0] == 'R':
        #we have the polar space measurements; we need to transform to cart space.
    prv_time = measurements[4][0]/1000000.0
    x,y,theta,v, theta_dot = polar_2_cart(measurements[1][0],measurements[2][0],measurement[3][0]),0,0.006911322

X = np.matrix([
    [x],
    [y],
    [theta],
    [v],
    [theta_dot]
])


# In[113]:


z_lidar = np.matrix([
    [0.],
    [0.]
])
z_radar = np.matrix([
    [0.],
    [0.],
    [0.]
])


# In[139]:


##### Defining the functions #########





# In[495]:


class EKF:
    def __init__(self,X):
        self.x = X
        self.xI = np.matrix([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0], [0,0,0,0,1]])
        self.A = None
        self.Q = np.matrix([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        

        self.P = np.matrix([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1100,0], [0,0,0,0,1100]])  ## As the from the LIDAR we can deduce the values of x,y,theta but are 

        self.H_L = np.matrix([[1,0,0,0,0],[0,1,0,0,0]])

        self.H_R = np.matrix([[0,0,0,0,0], [0,0,0,0,0],[0,0,0,0,0]])

        self.R_L = np.matrix([[0.02387,0],[0,0.02096]])

        self.R_R = np.matrix([[0.0947,0,0],[0,0.007,0],[0,0,0.0831]])
        
        #we can adjust these to get better accuracy
        self.noise_a = 9
        self.noise_alpha = 9
    
    
    def current_estimate(self):
        return (self.x, self.P)

    def init_state_vector(self, x,y, theta, v , theta_dot):
        self.x = np.matrix([[x,y,theta, v ,theta_dot]]).T
        
    def recompute_A_and_Q(self, dt):
        '''
        updates the motion model and process covar based on delta time from last measurement.
        '''

        #set A
        px,py,theta, v, theta_dot = state_vector_to_scalars(self.x)
        
        if theta_dot < 0.0001: # Driving straight 
#             px = px + v*dt * np.cos(theta)
#             py = py + v*dt * np.sin(theta)
              theta = theta
#             v = v
              theta_dot = 0.006911322 # avoid numerical issues in Jacobians

        else: # otherwise
#             px = px + (v/theta_dot) * (np.sin(theta_dot*dt+theta) - np.sin(theta))
#             py = py + (v/theta_dot) * (-np.cos(theta_dot*dt+theta)+ np.cos(theta))
              theta = (theta + theta_dot*dt + np.pi) % (2.0*np.pi) - np.pi
#             v = v
              theta_dot = theta_dot
        ### setting the Jacobian Gt and then finding A
        
        a13 = float((v/theta_dot) * (np.cos(theta_dot*dt+ theta) - np.cos(theta)))
        a14 = float((1.0/theta_dot) * (np.sin(theta_dot*dt+theta) - np.sin(theta)))
        a15 = float((dt*v/theta_dot)*np.cos(theta_dot*dt+theta) - (v/theta_dot**2)*(np.sin(theta_dot*dt+theta) - np.sin(theta)))
        a23 = float((v/theta_dot) * (np.sin(theta_dot*dt+theta) - np.sin(theta)))
        a24 = float((1.0/theta_dot) * (-np.cos(theta_dot*dt+theta) + np.cos(theta)))
        a25 = float((dt*v/theta_dot)*np.sin(theta_dot*dt+theta) - (v/theta_dot**2)*(-np.cos(theta_dot*dt+theta) + np.cos(theta)))
        self.A = np.matrix([[1.0, 0.0, a13, a14, a15],
                        [0.0, 1.0, a23, a24, a25],
                        [0.0, 0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])


        #set Q
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        e11 = dt4 * np.cos(theta)**2 * self.noise_a / 4
        e12 = dt4 * np.cos(theta) * np.sin(theta) * self.noise_a / 4
        e13 = dt3 * np.cos(theta) * self.noise_a / 2
        e21 = dt3 * np.cos(theta) * np.sin(theta) * self.noise_a / 4
        e22 = dt4 * np.sin(theta)**2 * self.noise_a / 4
        e23 = dt3 * np.sin(theta) * self.noise_a /  2
        e41 = dt3 * np.cos(theta) * self.noise_a / 2
        e42 = dt3 * np.sin(theta) * self.noise_a / 2
        e43 = dt2 * self.noise_a
        e34 = dt4 * self.noise_alpha / 4
        e35 = dt3 * self.noise_alpha / 2
        e54 = dt3 * self.noise_alpha / 2
        e55 = dt2 * self.noise_alpha
        

        self.Q = np.matrix([[e11, e12, e13, 0, 0],
                            [e21, e22, e23, 0, 0],
                            [0, 0, 0, e34, e35],
                            [e41, e42, e43, 0, 0],
                            [0, 0, 0, e54, e55]])
        
    def recompute_HR(self):
        '''
        calculate_jacobian of the current state.
        '''
        px,py,theta, v, theta_dot = state_vector_to_scalars(self.x)

        pxpy_squared = px**2+py**2
        pxpy_squared_sqrt = np.sqrt(pxpy_squared)
        pxpy_cubed = (pxpy_squared*pxpy_squared_sqrt)

        if pxpy_squared < 1e-4:
            self.H_R = np.zeros((3,5))
            return

        e11 = px/pxpy_squared_sqrt
        e12 = py/pxpy_squared_sqrt
        e21 = -py/pxpy_squared
        e22 = px/pxpy_squared
        e31 = py*(v*np.cos(theta)*py - v*np.sin(theta)*px)/pxpy_cubed
        e32 = px*(v*px*np.sin(theta) - v*py*np.cos(theta))/pxpy_cubed
        e33 = (py*v*np.cos(theta) -px*v*np.sin(theta))/pxpy_squared_sqrt
        e34 = (px*np.cos(theta) + py*np.sin(theta)/pxpy_squared_sqrt)

        self.H_R = np.matrix([[e11, e12, 0, 0, 0],
                               [e21, e22, 0, 0, 0],
                               [e31, e32, e33, e34, 0]])


    def predict(self):
        '''
        This is a projection step. we predict into the future.
        '''
        self.x = self.A * self.x
        self.P = (self.A * self.P * self.A.T) + self.Q

    def updateLidar(self,z):
        '''
        This is the projection correction; after we predict we use the sensor data
        and use the kalman gain to figure out how much of the correction we need.
        '''

        #this is the error of our prediction to the sensor readings
        y = z - self.H_L*self.x

        #pre compute for the kalman gain K
        PHLt = self.P * self.H_L.T
        S = self.H_L * PHLt + self.R_L
        K = PHLt*S.I

        #now we update our prediction using the error and kalman gain.
        self.x += K*y
        self.P = (self.xI - K*self.H_L) * self.P

    def updateRadar(self,z):
        '''
        This is the projection correction; after we predict we use the sensor data
        and use the kalman gain to figure out how much of the correction we need.
        This is a special case as we will need a Jocabian matrix to have a linear
        approximation of the transformation function h(x)
        '''

        y = z - cart_2_polar(self.x)
        #make sure the phi in y is -pi <= phi <= pi
        while (y[1] > np.pi): y[1] -= 2.*np.pi
        while (y[1] < -np.pi): y[1] += 2.*np.pi

        #recompute Jacobian
        self.recompute_HR()

        #pre compute for the kalman gain K
        #TODO: this code is not DRY should refactor here.
        S = self.H_R * self.P * self.H_R.T + self.R_R
        K = self.P*self.H_R.T*S.I

        #now we update our prediction using the error and kalman gain.
        self.x += K*y
        self.P = (self.xI - K*self.H_R) * self.P


# In[496]:


column = ['time','x_state','y_state','vx_state','vx_state','yaw_angle_state', 'yaw_rate_state', 'sensor_type','x_measured' ,'y_measured' ,'x_ground_truth' ,'y_ground_truth' ,'vx_ground_truth' ,'vy_ground_truth']


# In[497]:


output = pd.DataFrame(columns = column)


# In[498]:


estimations = [X]
list_ground_truth = []
ground_truth = np.matrix([[0.],[0.],[0.],[0.]])
sensor_type = []
prev_time = 1477010443000000
for i in range (1,len(measurements)):
    new_measurement = measurements.iloc[i, :].values
    if new_measurement[0] == 'L':
        #Calculate Timestamp and its power variables
        dt = 0.05
        cur_time = prev_time + dt*1000000.0
        prev_time = cur_time
        #Updating sensor readings
        sensor_type.append(new_measurement[0])
        z_lidar[0][0] = new_measurement[1]
        z_lidar[1][0] = new_measurement[2]

        #Call Kalman Filter Predict and Update functions.
        ekf = EKF(estimations[i-1])
        ekf.recompute_A_and_Q(dt)
        ekf.predict()
#         estimations = estimations.append(X)
        ekf.updateLidar(z_lidar)
        estimations.append(ekf.x)
        out = [cur_time,estimations[i][0,0],estimations[i][1,0],estimations[i][3,0]*np.cos(estimations[i][2,0]),estimations[i][3,0]*np.sin(estimations[i][2,0]),estimations[i][2,0],estimations[i][4,0],new_measurement[0],
              new_measurement[1],new_measurement[2],new_measurement[4],new_measurement[5],new_measurement[6],new_measurement[7]]
        output = output.append(pd.DataFrame([out], columns=column), ignore_index=True)
    
    elif new_measurement[0] == 'R':
        dt = 0.05
        cur_time = prev_time + dt*1000000.0
        prev_time = cur_time


        #Updating sensor readings
        sensor_type.append(new_measurement[0])
        z_radar[0][0] = new_measurement[1]
        z_radar[1][0] = new_measurement[2]
        z_radar[2][0] = new_measurement[3]
        
        z_radar_x_pos, z_radar_y_pos, z_radar_v = polar_2_cart(new_measurement[1],new_measurement[2],new_measurement[3])
        
#         print(ground_truth)
#         print(list_ground_truth.append(ground_truth))
        #Call Kalman Filter Predict and Update functions.
        ekf = EKF(estimations[i-1])
        ekf.recompute_A_and_Q(dt)
        ekf.predict()
        
#         estimations = estimations.append(X)
        ekf.updateRadar(z_radar)
        estimations.append(ekf.x)
        out = [cur_time,estimations[i][0,0],estimations[i][1,0],estimations[i][3,0]*np.cos(estimations[i][2,0]),estimations[i][3,0]*np.sin(estimations[i][2,0]),estimations[i][2,0],estimations[i][4,0],new_measurement[0],
            z_radar_x_pos,z_radar_y_pos,new_measurement[5],new_measurement[6],new_measurement[7],new_measurement[8]]
        output = output.append(pd.DataFrame([out], columns=column), ignore_index=True)



# In[499]:


output.to_csv('output_EKF.csv')


# In[500]:


output


# In[501]:


estimate_x_L = []
estimate_y_L = []
estimate_x_R = []
estimate_y_R = []

for i in range(output.shape[0]):
    out = output.iloc[i, :].values
    if out[7] =='L':
        estimate_x_L.append(out[1])
        estimate_y_L.append(out[2])
    
    elif out[7] == 'R':
        estimate_x_R.append(out[1])
        estimate_y_R.append(out[2])
        


# In[502]:


estimate_x = []
estimate_y = []


for i in range(output.shape[0]):
    out = output.iloc[i, :].values
    estimate_x.append(out[1])
    estimate_y.append(out[2])


# In[503]:


# plt.plot(estimate_x,estimate_y,'k', label = "EKF_Estimate")

# # plt.plot(estimate_x_R,estimate_y_R,'o', '-r', label = "EKF_Estimate_Radar")
# plt.plot(measurements_lidar['x_pos'],measurements_lidar['y_pos'],color = 'k',label = 'Lidar Measurements')
# plt.plot(measurements_radar['x_measured'],measurements_radar['y_measured'],color = 'r',label = 'Radar Measurements')
# plt.xlabel('x_pos')
# plt.ylabel('y_pos')
# plt.legend(loc = 'best')
# plt.title('comparison of x and y from EKF vs measurements from Lidar and Radar')
# plt.show()


# In[504]:


plt.plot(estimate_x,estimate_y, '-c', label = "EKF_Estimate")
# plt.plot(estimate_x_R,estimate_y_R,'o', '-r', label = "EKF_Estimate_Radar")
plt.plot(measurements_lidar['gt_x_pos'],measurements_lidar['gt_y_pos'],color = 'k',label = 'Ground Truth')
plt.xlabel('x_pos')
plt.ylabel('y_pos')
plt.legend(loc = 'best')
plt.title('comparison of estimated x and y from EKF vs Ground truth x and y')
plt.show()


# In[505]:


plt.plot(estimate_x_L,estimate_y_L, '-c', label = "EKF_Estimate_Lidar")
# plt.plot(estimate_x_R,estimate_y_R,'o', '-r', label = "EKF_Estimate_Radar")
plt.plot(measurements_lidar['gt_x_pos'],measurements_lidar['gt_y_pos'],color = 'k',label = 'Ground Truth')
plt.xlabel('x_pos')
plt.ylabel('y_pos')
plt.legend(loc = 'best')
plt.title('comparison of estimated x and y from LIDAR vs Ground truth x and y')
plt.show()


# In[506]:


plt.plot(estimate_x_R,estimate_y_R, '-r', label = "EKF_Estimate_Radar")
plt.plot(measurements_lidar['gt_x_pos'],measurements_lidar['gt_y_pos'],color = 'k',label = 'Ground Truth')
plt.xlabel('x_pos')
plt.ylabel('y_pos')
plt.legend(loc = 'best')
plt.title('comparison of estimated x and y from RADAR vs Ground truth x and y')
plt.show()


# In[ ]:





# In[ ]:




