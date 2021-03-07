import numpy as np
from numpy.random import rand, randint, choice, random_sample
import matplotlib.pyplot as plt
from pprint import pprint
from time import time

from requirement_functions_v2 import *

np.random.seed(7)


def _rand_between(rg, size=None):

    if size:
        return random_sample(size)*np.diff(rg) + np.min(rg)
    else:
        return float(random_sample(size)*np.diff(rg) + np.min(rg))


def _generate_points(search_space, num_pts):
    point_dict = {}

    for key, value in search_space.items():
        if key in ['pulley_tooth_pitch', 'pulley_tooth_qty']:
            point_dict[key] = randint(*search_space[key], num_pts)
        if key in ['microstep']:
            point_dict[key] = choice(search_space[key], num_pts)
        else:
            point_dict[key] = _rand_between(search_space[key], num_pts)

    return point_dict


def _map(**sample_space):
    dx = inplane_plate_deflection(mass = 1.2, **sample_space)
    dy = inplane_plate_deflection(**sample_space)
    Dx = x_travel(**sample_space)
    Dy = y_travel(**sample_space)
    v = nozzle_speed(**sample_space)
    res = resolution(**sample_space)

    return np.vstack((dx,dy,Dx,Dy,v,res))


def _check_points(R, C_points, sample_space):
    truth = np.vstack((
        np.logical_and(C_points[0,:] > min(R[0]), C_points[0,:] < max(R[0])),
        np.logical_and(C_points[1,:] > min(R[1]), C_points[1,:] < max(R[1])),
        np.logical_and(C_points[2,:] > min(R[2]), C_points[2,:] < max(R[2])),
        np.logical_and(C_points[3,:] > min(R[3]), C_points[3,:] < max(R[3])),
        np.logical_and(C_points[4,:] > min(R[4]), C_points[4,:] < max(R[4])),
        np.logical_and(C_points[5,:] > min(R[5]), C_points[5,:] < max(R[5])),
    ))

    passing = np.all(truth,0)

    F_points = np.array([val for val in sample_space.values()])
    F_pass_pts = F_points[:,passing]
    F_fail_pts = F_points[:,~passing]

    # C_pass_pts = C_points[:,passing]
    # C_fail_pts = C_points[:,~passing]

    return F_pass_pts, F_fail_pts


################################################################################
################################################################################
################################################################################
################################################################################


def pairwise_scatter(data, data2=None, sz=1,labels=None):
    
    # assert data.shape[0] < data.shape[1], "Data matrix must be horizontal"
    # if data2:
    #     assert data.shape[0]==data2.shape[0], "Data sets must have the same number of axes"

    dim, _ = data.shape

    # fig, axs = plt.subplots(dim-1,dim-1,figsize=(9,9),constrained_layout=True)
    fig, axs = plt.subplots(dim-1,dim-1,figsize=(9,9))

    if dim > 2:

        for i in range(dim-1):

            for ii in range(i):
                axs[i,ii].set_axis_off()

            for ii in range(i+1,dim):
                ax = axs[i,ii-1]
                if np.sum(data2):
                    ax.scatter(data2[ii,:],data2[i,:], s=sz, color='red')
                ax.scatter(data[ii,:],data[i,:], s=sz, color='blue')
                ax.set_aspect(1.0/ax.get_data_ratio())
                if labels:
                    ax.set_ylabel(labels[i])
                    ax.set_xlabel(labels[ii])

    else:

        axs.scatter(data[0,:],data[1,:])
        
        if np.sum(data2):
            axs.scatter(data2[0,:], data2[1,:], frame_side_len=sz, color='red')
        
        axs.set_aspect(1.0/axs.get_data_ratio())
        
        if labels:
            axs.set_xlabel(labels[0])
            axs.set_ylabel(labels[1])
    
    fig.tight_layout(pad=3.0)

    return fig, axs


################################################################################
################################################################################
################################################################################
################################################################################

#################### Main ########################
# Search space
Omega = {
    'frame_side_length': [21.0, 27.0],
    'plate_width': [2.0, 10.0],
    'plate_thickness': [0.1, 0.2],
    'modulus': [9*10**6, 12*10**6],
    'gantry_length': [10.0, 18.00],
    'max_motor_current': [1.0, 2.0],
    'motor_voltage': [2.5, 12.0],
    'motor_inductance': [0.002, 0.005],
    'step_angle': [1.5, 2.0],
    'microstep': [1/16, 1/8, 1/4],
    'pulley_tooth_pitch': [1.0, 3.0],
    'pulley_tooth_qty': [15.0, 50.0],
}

labels = ['s','q','t','E','G','I_max','V','L','phi','mu','p','n']

R_dx =  [-0.001,   0.001]
R_dy =  [-0.005,   0.005]
R_dG =  [-0.001,   0.001]
R_Dx =  [8.0,   np.inf]
R_Dy =  [8.0,   np.inf]
R_T_w = [200.0, np.inf]
R_v =   [16.0,  np.inf]
R_res = [0.0,   0.001]

R = [
    R_dx,
    R_dy,
    R_Dx,
    R_Dy,
    R_v,
    R_res,
]

N = 1000000
t0 = time()
print("\nTime started")
F = _generate_points(Omega, N)
C = _map(**F)
# P, notP = _check_points(R,C,F)
S, notS = _check_points(R, C, F)

# print(P)
# print(notP)

# fig, ax = pairwiseScatter(P,notP)
fig, ax = pairwise_scatter(S, notS, labels=labels)
print(f'Process complete for {N} samples in {np.round(time()-t0,2)} seconds')
plt.show()