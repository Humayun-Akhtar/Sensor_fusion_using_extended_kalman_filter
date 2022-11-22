import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from math import sin, cos, sqrt, atan2


data = pd.read_excel('data_ekf.xlsx')

lidar_heading = data.columns

radar_heading = data.iloc[0,:].values




#Defining Initail condition


prev_time = 1477010443000000/1000000

XO = np.matrix([0.3122427,0.5803398,0,0]).T

#Z_lidar = 


def state_vector_to_scalars(state_vector):

    return (state_vector[0][0,0],state_vector[1][0,0],state_vector[2][0,0],state_vector[3][0,0])


def cartesian_to_polar(state_vector):


    px,py,vx,vy = state_vector_to_scalars(state_vector)

    ro= sqrt(px**2 + py**2)

    phi     = atan2(py,px)

    ro_dot  = (px*vx + py*vy)/ro

    return np.matrix([ro, phi, ro_dot]).T
       


class EKF:

    def __init__(self):

        self.x = None
        

        self.I = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.p = np.matrix ([[1,0,0,0],[0,1,0,0],[0,0,1000,0],[0,0,0,1000]])

        self.H_lidar = np.matrix ([[1,0,0,0],[0,1,0,0]])

        self.R_lidar = np.matrix([[0.0228,0],[0,0.0212]])

        self.R_radar = np.matrix([[0.0928,0,0],[0,5.58,0],[0,0,0.0831]])


    def init_state_vector(self, x,y, vx, vy):

        self.x = np.matrix([[x,y,vx,vy]]).T


    def current_estimate(self):

        return (self.x, self.p)


    def recompute_F_and_Q(self,dt):

        self.F = np.matrix([[1, 0, dt, 0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])

        sigma2_ax = 12.57

        sigma2_ay = 1.875

        dt2 = dt*dt

        dt3 = dt2*dt

        dt4 = dt3*dt


        a11 = dt4*sigma2_ax/4

        a13 = dt3*sigma2_ax/2

        a22 = dt4*sigma2_ay/4

        a24 = dt3*sigma2_ay/2

        a31 = dt3*sigma2_ax/2

        a33 = dt2*sigma2_ax

        a42 = dt3*sigma2_ay/2

        a44 = dt2*sigma2_ay


        self.Q = np.matrix([[a11,0,a13,0],[0,a22,0,a24],[a31,0,a33,0],[0,a42,0,a44]])

    



    def recompute_H_radar(self):

        px,

        d = (Xt**2+Yt**2)

        sqrtd = sqrt(d)

        drootcubed = d*sqrt(d)


        h11 = self.Xt

        h31 = (Vx*Xt+Vy*Yt)/ro
                
        
        

    def recompute_HR(self):

        px,py,vx,vy = state_vector_to_scalars(self.x)


        pxpy_squared = px**2+py**2

        pxpy_squared_sqrt = sqrt(pxpy_squared)

        pxpy_cubed = (pxpy_squared*pxpy_squared_sqrt)


        if pxpy_squared < 1e-4:

            self.H_radar = np.matlib.zeros((3,4))

            return


        e11 = px/pxpy_squared_sqrt

        e12 = py/pxpy_squared_sqrt

        e21 = -py/pxpy_squared

        e22 = px/pxpy_squared

        e31 = py*(vx*py - vy*px)/pxpy_cubed

        e32 = px*(px*vy - py*vx)/pxpy_cubed


        self.H_radar = np.matrix([[e11, e12, 0, 0],
                            [e21, e22, 0, 0],

                            [e31, e32, e11, e12]])

    

    def predict(self):

        self.x = self.F * self.x

        self.p = (self.F * self.p * self.F.T) + self.Q
    

    def updatelidar(self):

        z = np.matrix([data['x_position'][i], data['y_position'][i]]).T
        

        y = z - self.H_lidar*self.x

        Si = self.H_lidar *self.p*self.H_lidar.T + self.R_lidar

        K = (self.p * self.H_lidar.T)*Si.I

        self.x += K*y

        self.p = (self.I - K*self.H_lidar)*self.p


    def updateradar(self):

        z = np.matrix([data['x_position'][i], data['y_position'][i], data['time'][i]]).T
        

        y = z - cartesian_to_polar(self.x)
        

        #make sure the phi in y is -pi <= phi <= pi

        while (y[1] > np.pi): y[1] -= 2.*np.pi

        while (y[1] < -np.pi): y[1] += 2.*np.pi


        #recompute Jacobian

        self.recompute_HR()
        
        

        S = self.H_radar * self.p * self.H_radar.T + self.R_radar

        K = self.p*self.H_radar.T*S.I
        

        self.x += K*y
        

        self.p = (self.I - K*self.H_radar)*self.p
        





class iteration:

    def __init__(self):


        self.ekf = EKF()


        self.ekf.current_estimate()

initial_time = data['time'][1]
prediction = []

Estimation = []

Estimation.append(XO)

ekf = EKF()

#

for i in range(2,len(data['Lidar'])):

    if data['Lidar'][i] == 'R':

        dt = 0.05

    if data['Lidar'][i] == 'L':

        dt = 0.05
    

    a,b,c,d = state_vector_to_scalars(Estimation[i-2])

    ekf.init_state_vector(a,b,c,d)

    ekf.recompute_F_and_Q(dt)


    ekf.predict()
    

    prediction.append(ekf.x)

    if data['Lidar'][i] == 'R':

        ekf.updateradar()

        Estimation.append(ekf.x)
        
        

    if data['Lidar'][i] == 'L':

        ekf.updatelidar()

        Estimation.append(ekf.x)  
       
        

X = []

Y = []

F = []

G = []

k =500

for i in range(len(Estimation)):

    x = Estimation[i][0]
    x = x.tolist()

    X.append(x)

for i in range(len(Estimation)):

    y = Estimation[i][1]
    y = y.tolist()

    Y.append(y)    

for i in range(len(X)):

    F.append(X[i][0])


for j in range(len(Y)):

    G.append(Y[j][0])    



num = pd.read_excel('data_ekf_lidar.xlsx')

num1 = pd.read_excel('data_ekf_radar.xlsx')

x = num[column := 'gt_x_position']	

y = num[column := 'gt_y_position']

x1 = num[column := 'x_position']	

y1 = num[column := 'y_position']

X = num1[column := 'distance']

Y = num1[column := 'h']

Xr = []

Yr = []

for i in range(len(X)):

    Xr.append(cos(Y[i])*X[i])

    Yr.append(sin(Y[i])*X[i])

plt.xlabel('X_position')

plt.ylabel('Y_position')

plt.title('Prediction of the path using Exteneded Kalman Filter')
plt.scatter(Xr,Yr, color = 'y', label = 'Radar Data')
plt.scatter(x1,y1, color = 'r', label = 'Lidar Data')
plt.scatter(F,G, color = 'k', label = 'EKF')

plt.legend(loc = 'best')

plt.show()