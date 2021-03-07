from warnings import warn
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# np.random.seed(1231)
np.random.seed(1236)

############### Frame stiffness ############################
def area_moment_of_inertia_x(xDim,yDim):
    """
    Calculate bending moment of a solid rectangular beam in x-direction.

    Arguments:
        xDim (float): cross-sectional dimension in direction of bending
        yDim (float): cross-sectional dimension perpendicular to direction of bending

    Returns (float): area moment of inertia
    """

    return yDim * xDim**3 / 12


def _stiffness_matrix(cross_sect_area,E,area_inertia,L):
    """
    Calculate the stiffness matrix for a single member of a frame.

    Arguments:
        cross_sect_area (float): thickness of the plate (minimum dimension)
        E (float): modulus of elasticity of plate material [psi x 10^-6] (i.e. enter 7 for 7x10^6 psi)
        area_inertia (float): dimension of the plate along the direction of the shear force
        L (float): length of plate, perpendicular to line of action of the shear force
            
    Returns:
        K (np.ndarray): stiffness matrix
    """
    
    E = E * 10**6

    a = cross_sect_area*E/L
    b = 12*E*area_inertia/L**3
    c = 6*E*area_inertia/L**2
    d = 4*E*area_inertia/L
    e = 2*E*area_inertia/L

    K = np.array([[ a,  0,  0, -a,  0,  0],
                  [ 0,  b,  c,  0, -b,  c],
                  [ 0,  c,  d,  0, -c,  e],
                  [-a,  0,  0,  a,  0,  0],
                  [ 0, -b, -c,  0,  b, -c],
                  [ 0,  c,  e,  0, -c,  d]])

    return K


def forces_to_deflect_frame_member(plate_thickness,plate_dim,frame_side_len,E,delta_max):
    """
    Calculate the forces needed to cause deflection of a plate with a shear force applied in-plane at one edge.

    Arguments:
        plate_thickness (float): thickness of the plate (minimum dimension)
        plate_dim (float): dimension of the plate along the direction of the shear force
        frame_side_len (float): length of plate, perpendicular to line of action of the shear force
        E (float): modulus of elasticity of plate material [psi x 10^-6] (i.e. enter 7 for 7x10^6 psi)
        d (float): allowable in-plane displacement
    
    Returns (np.ndarray): forces needed to produce specified deflection of plate in-plane
    """

    E = E * 10**6
    # F = np.array([0, f, -f*frame_side_len/2, 0, -f, f*frame_side_len/2]).T
    I = area_moment_of_inertia_x(plate_dim,plate_thickness)
    K = _stiffness_matrix(plate_thickness*plate_dim, E, I, frame_side_len)

    dy = delta_max
    dx = frame_side_len - np.sqrt(frame_side_len**2 - dy**2)
    phi = 0

    delta = np.array([dy, dx, phi, 0, 0, 0]).T

    return K @ delta
    

def frame_deflection(frame_side_len, mass, accel, plate_dim, plate_thickness, E):
    """
    Calculate the deflection of reinforcing plates caused by accelerations.

    Arguments:
        plate_thickness (float): thickness of the plate (minimum dimension)
        plate_dim (float): dimension of the plate along the direction of the shear force
        frame_side_len (float): length of plate, perpendicular to line of action of the shear force
        E (float): modulus of elasticity of plate material [psi x 10^-6] (i.e. enter 7 for 7x10^6 psi)
        mass (float): mass under acceleration
        accel (float): max acceleration (deceleration) of the print head
    
    Returns (float): resulting in-plane deformation of the frame
    """
    E = E * 10**6
    return frame_side_len * mass *accel / (plate_dim * plate_thickness * E)

############### Gantry stiffness (ANSYS integration) #######


############### Rapid speed ################################
def _effective_pulley_radius(pulley_tooth_qty, pulley_tooth_pitch):
    """
    Calculate the effective pulley radius

    Arguments:
        pulley_tooth_qty (float): number of pulley teeth
        pulley_tooth_pitch (float): arc length between teeth [mm]
    
    Returns (float): effective pulley radius [mm]
    """
    
    return pulley_tooth_qty*pulley_tooth_pitch / (2*np.pi)


def _stepper_speed(motor_voltage,motor_inductance,max_motor_current,phi):
    # Credit: https://dyzedesign.com/2016/11/printing-300-mm-s-part-2-calculations/
    #         https://www.daycounter.com/Calculators/Stepper-Motor-Calculator.phtml
    """
    Calculate the speed of the motor at max torque

    Arguments:
        motor_voltage (float): operating voltage [V]
        motor_inductance (float): phase inductance [H]
        max_motor_current (float): maximum current draw [A]
        steps (float): steps per revolution
    
    Returns (float): motor speed [Hz]
    """

    steps = 360 / phi
    # Convert step angle to radians
    # This is the change in position for one step
    dTheta = np.pi*phi / 180

    # Calculate dt as the time needed to complete one step
    # Use the time derivative of formula relating V, I, and L
    #       dt = (L/V)*dI
    # One step: 0 -> I -> 0, so dI = 2I
    dt = 2*max_motor_current*motor_inductance/motor_voltage
    
    return dTheta / dt


def rapid_speed(pulley_tooth_qty, pulley_tooth_pitch, motor_voltage, motor_inductance, max_motor_current, phi):
    """
    Calculate the speed of the extrueder at max pulley speed

    Arguments:
        pulley_tooth_qty (float): number of pulley teeth
        pulley_tooth_pitch (float): arc length between teeth
        motor_voltage (float): operating voltage
        motor_inductance (float): phase inductance
        max_motor_current (float): maximum current draw
        phi (float): step angle [degree]
    
    Returns (float): extruder speed [mm/s]
    """

    n_p = _stepper_speed(motor_voltage, motor_inductance, max_motor_current, phi)
    r = _effective_pulley_radius(pulley_tooth_qty, pulley_tooth_pitch)

    return r*n_p


############### X-Y Resolution #############################
def resolution(pulley_tooth_pitch,pulley_tooth_qty,phi,uStep):
    """
    Calculate the speed of the extrueder at max pulley speed

    Arguments:
        pulley_tooth_qty (float): number of pulley teeth
        pulley_tooth_pitch (float): arc length between teeth
        phi (float): step angle [degree]
        uStep (float): micro-step [fraction]
    
    Returns (float): x-y resolution [mm]
    """

    # Equation:
    ### Circumference: C = p*N
    ### Pulley radius: r = C/2*pi = p*N/2*pi
    ### Step angle to rad: theta = 2*pi*phi/180
    ### Distance for full step: d = r*theta = 2*pi*p*N*phi/2*pi*180
    ### Distance for microstep (2*pi canceled): d_u = uStep*p*N*phi/180
    ### Divide by 2 due to coreXY design: d_u = uStep*p*N*phi/360

    return uStep*pulley_tooth_pitch*pulley_tooth_qty*phi/360


############### X-Y travel #################################
def f_y_travel(frame_side_len,dim_8020):
    """
    Calculate the available y-travel

    Arguments:
        frame_side_len (float): side length of frame
        dim_8020 (float): cross-sectional side dimension of frame bar
    
    Returns (float): y-travel
    """

    return frame_side_len - 2*dim_8020


def f_x_travel(frame_side_len,gantry_length,print_head_width):
    """
    Calculate the available x-travel in inches

    Arguments:
        frame_side_len (float): side length of frame [in]
        gantry_length (float): length of gantry bar [in]
        print_head_width (float): width (x-dim) of print head assy [in]
    
    Returns (float): x-travel [in]
    """

    # offset between each end of gantry bar and limits of travel
    R = 0.5 + (frame_side_len - gantry_length)/2
    
    return gantry_length - 2*(R-print_head_width)


############### Material working temperature ###############
# Table look up


############### Material working temperature ###############
def _map(gantry_length, frame_side_len, plate_thickness, plate_dim, E, max_motor_current,
            motor_voltage, motor_inductance, pulley_tooth_pitch, pulley_tooth_qty, phi,
            uStep, dim_8020=1.77, m_head=1.25, m_G=2.5, accel=200, print_head_width=3):
    """
    Returns coordinates in the problem space mapped to by input point in solution space.

    Arguments:
        gantry_length (float): length of gantry bar
		print_head_width (float): width (x-dim) of print head assy
		max_motor_current (float): maximum current draw
		frame_side_len (float): side length of frame
		m_G: (float): mass of the gantry bar
		pulley_tooth_pitch (float): arc length between teeth
		motor_voltage (float): operating voltage
		dim_8020 (float): cross-sectional side dimension of frame bar
		m_head (float): mass of the print head
		accel (float): max acceleration (deceleration) of the print head
		plate_thickness (float): thickness of the plate (minimum dimension)
		plate_dim (float): dimension of the plate along the direction of the shear force
		pulley_tooth_qty (int): number of pulley teeth
		E (float): modulus of elasticity of plate material [psi x 10^-6] (i.e. enter 7 for 7x10^6 psi)
		phi (float): step angle [degree]
		uStep (float): micro-step [fraction]
		motor_inductance (float): phase inductance
    
    Returns (list): Problem space coordinate values
    """
    
    dx = frame_deflection(frame_side_len, m_head, accel, plate_dim, plate_thickness, E)
    dy = frame_deflection(frame_side_len, m_head + m_G, accel, plate_dim, plate_thickness, E)
    # dG = TBD
    Dx = f_x_travel(frame_side_len, gantry_length, print_head_width)
    Dy = f_y_travel(frame_side_len, dim_8020)
    # T_w = TBD
    v = rapid_speed(pulley_tooth_qty, pulley_tooth_pitch, motor_voltage, motor_inductance, max_motor_current, phi)
    res = resolution(pulley_tooth_pitch, pulley_tooth_qty, phi, uStep)

    return dx, dy, Dx, Dy, v, res


############### Plotting ###################################
# def pairwiseScatter(data, sz=1,labels=None):
#     dim,_ = data.shape

#     fig, axs = plt.subplots(dim-1,dim-1,figsize=(9,9))
#     if dim > 2:
#         for i in range(dim-1):
#             for ii in range(i):
#                 axs[i,ii].set_axis_off()
#             for ii in range(i+1,dim):
#                 ax = axs[i,ii-1]
#                 ax.scatter(data[ii,:],data[i,:], frame_side_len=sz)
#                 ax.set_aspect(1.0/ax.get_data_ratio())
#                 if labels:
#                     ax.set_ylabel(labels[i])
#                     ax.set_xlabel(labels[ii])
#     else:
#         axs.scatter(data[0,:],data[1,:])
#         axs.set_aspect(1.0/axs.get_data_ratio())
#         if labels:
#                     ax.set_xlabel(labels[0])
#                     ax.set_ylabel(labels[1])
    
#     fig.tight_layout(pad=3.0)

#     return fig, axs


def pairwiseScatter(data, data2=None, sz=1,labels=None):
    
    # assert data.shape[0] < data.shape[1], "Data matrix must be horizontal"
    # if data2:
    #     assert data.shape[0]==data2.shape[0], "Data sets must have the same number of axes"

    dim,_ = data.shape

    fig, axs = plt.subplots(dim-1,dim-1,figsize=(9,9))
    if dim > 2:
        for i in range(dim-1):
            for ii in range(i):
                axs[i,ii].set_axis_off()
            for ii in range(i+1,dim):
                ax = axs[i,ii-1]
                if np.sum(data2):
                    ax.scatter(data2[ii,:],data2[i,:], s=sz, color='red')
                ax.scatter(data[ii,:],data[i,:], s=10*sz, color='blue')
                ax.set_aspect(1.0/ax.get_data_ratio())
                if labels:
                    ax.set_ylabel(labels[i])
                    ax.set_xlabel(labels[ii])
    else:
        axs.scatter(data[0,:],data[1,:])
        if np.sum(data2):
                    ax.scatter(data2[ii,:],data2[i,:], frame_side_len=sz, color='red')
        axs.set_aspect(1.0/axs.get_data_ratio())
        if labels:
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
    
    fig.tight_layout(pad=3.0)

    return fig, axs


def pairwise_problem_space(R, sz=1, labels=None):
    dim = len(R)

    fig, axs = plt.subplots(dim-1,dim-1,figsize=(9,9),constrained_layout=True)
    fig.suptitle("Problem Space")
    if dim > 2:
        for i in range(dim-1):
            for ii in range(i):
                axs[i,ii].set_axis_off()
            for ii in range(i+1,dim):
                ax = axs[i,ii-1]

                x_min = R[ii][0]
                x_max = R[ii][1]
                y_min = R[i][0]
                y_max = R[i][1]

                x_lims = [max(0,x_min - 0.2*x_max), 1.2*x_max]
                y_lims = [max(0,y_min - 0.2*y_max), 1.2*y_max]

                x = np.linspace(*R[ii],5)
                y = np.linspace(*R[i],5)

                ax.set_xlim(*x_lims)
                ax.set_ylim(*y_lims)

                ax.axhline(y_min, color='k', lw=2, alpha=0.5)
                ax.axhline(y_max, color='k', lw=2, alpha=0.5)
                ax.axvline(x_min, color='k', lw=2, alpha=0.5)
                ax.axvline(x_max, color='k', lw=2, alpha=0.5)

                ax.fill_between(x, y_min, y_max, facecolor='blue', alpha=0.5)
                ax.fill_betweenx(y, x_min, x_max, facecolor='red', alpha=0.5)

                if labels and i+1==ii:
                    ax.set_ylabel(labels[i])
                    ax.set_xlabel(labels[ii])
    else:
        axs.scatter(data[0,:],data[1,:])
        axs.set_aspect(1.0/axs.get_data_ratio())
        if labels:
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
    
    # fig.tight_layout(pad=0.04)

    return fig, axs


def rand_between(rg,size=None):

    if size:
        return list(np.random.random_sample(size)*np.diff(rg) + np.min(rg))
    else:
        return float(np.random.random_sample(size)*np.diff(rg) + np.min(rg))


def _ID_requirement(i):
    req = {
        0: 'R_del_x',
        1: 'R_del_y',
        2: 'R_D_x',
        3: 'R_D_y',
        4: 'R_v',
        5: 'R_del_x'
    }

    return req[i]


def _check_point(R,point):
    """
        Assumes requirement is a single, continuous range.
    """
    candidate = _map(*point)
    n = len(R)
    for i in range(n):
        if not (min(R[i]) <= candidate[i] <= max(R[i])):
            # print(f" Failed: {_ID_requirement(i)}\t\tAllowed range: {R[i]}\t\tValue at pt: {np.round(candidate[i],4)}\n")
            return False
    
    return True


def _generate_point(form_search):

    # m = len(form_search)
    # point = []
    # for i in range(m):
    #     if i==9:
    #         val = np.random.randint(*form_search[i])
    #     elif i==11:
    #         val = np.random.choice(form_search[i])
    #     else:
    #         val = rand_between(form_search[i])
    #     point.append(val)
    
    # return point

    frame_side_len = rand_between(form_search[1])
    plate_thickness = rand_between(form_search[2])
    E = rand_between(form_search[4])
    max_motor_current = rand_between(form_search[5])
    motor_voltage = rand_between(form_search[6])
    motor_inductance = rand_between(form_search[7])
    pulley_tooth_pitch = rand_between(form_search[8])
    pulley_tooth_qty = np.random.randint(*form_search[9])
    phi = rand_between(form_search[10])
    uStep = np.random.choice(form_search[11])

    gantry_length = np.random.randint(min(form_search[0]),np.floor(frame_side_len))
    plate_dim = np.random.randint(min(form_search[3]),np.floor(frame_side_len))

    return [gantry_length, frame_side_len, plate_thickness, plate_dim, E, max_motor_current,
            motor_voltage, motor_inductance, pulley_tooth_pitch, pulley_tooth_qty, phi,
            uStep]


def _generate_sample_points(form_search, num_pts):

    return [_generate_point(form_search) for _ in range(num_pts)]


def solution_space_search(R, form_search, num_pts=200, labels=None):

    pts = _generate_sample_points(form_search, num_pts)

    soln_pts = []
    fail_pts = []
    for pt in pts:
        if _check_point(R,pt):
            soln_pts.append(pt)
        else:
            fail_pts.append(pt)

    if soln_pts:
        solns = np.array(soln_pts).T
        fails = np.array(fail_pts).T
        fig, ax = pairwiseScatter(solns, data2=fails, sz=1, labels=labels)
    else:
        warn("No solutions found")
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    return fig, ax

if __name__ == "__main__":

# ############### Frame Deflection #####################
#     print("\n")

#     frame_side_len = 24          # long dimension (side length of frame) [in]
#     plate_dim = 4         # short dimension of plate [in]
#     plate_thickness = 0.125       # Plate thickness [in]
#     E = 11.4e6  # Young's modulus of aluminum allow [psi]

#     dy = 0.001      # Maximum allowable in-plane displacement

#     forces = forces_to_deflect_frame_member(plate_thickness,plate_dim,frame_side_len,E,dy)

#     a_ext = 200     # Extruder acceleration [in/s^2]
#     m_G = 2.5         # mass of gantry [lb]
#     m_E = 1.25      # mass of extruder [lb]
#     fy = (m_G + m_E) * a_ext / 12   # Max force on frame in x [lb]

#     print(
#         f"Acceleration Force: {fy} lb\n",
#         f"Deformation Force: {max(abs(forces))} lb"
#         )

# ############### Rapid Speed ##########################
#     print("\n")

#     N = 20
#     p = 2
#     V = 3.06
#     L = 3.8e-3
#     I_max = 1.7
#     steps = 200
#     phi = 1.8

#     n_p = _stepper_speed(V, L, I_max, phi)
#     r = _effective_pulley_radius(N, p)
#     v = rapid_speed(N, p, V, L, I_max, phi)
#     # v = _stepper_speed(V,L,I_max,steps)

#     print(
#         f"Pulley Radius: {r} mm\n",
#         f"Motor Speed: {n_p} Hz\n",
#         f"Extruder velocity: {v} mm/s"
#         )

# ############### X-Y Resolution #######################
#     print("\n")

#     phi = 1.8
#     uStep = 1/8
#     d = resolution(p,N,phi,uStep)

#     print(f"X-Y Resolution: {d} mm\n")

# ############### X & Y Travels ########################
#     print("\n")

############### Problem Space Pairwise Plot ########################

    res = 50

    R_del_x =   [0.0,   0.001]
    R_del_y =   [0.0,   0.005]
    R_del_G =   [0.0,   0.001]
    R_D_x =     [8.0,   1200.0]
    R_D_y =     [8.0,   1200.0]
    R_T_w =     [200.0, 1000.0]
    R_v =       [16.0,  1200.0]
    R_del_x =   [0.0,   0.001]

    R = [
        R_del_x,
        R_del_y,
        R_del_G,
        R_D_x,
        R_D_y,
        R_T_w,
        R_v,
        R_del_x,
    ]

    n = len(R)

    axs = np.stack(
        (
            np.linspace(*R_del_x,res),
            np.linspace(*R_del_y,res),
            np.linspace(*R_del_G,res),
            np.linspace(*R_D_x,res),
            np.linspace(*R_D_y,res),
            np.linspace(*R_T_w,res),
            np.linspace(*R_v,res),
            np.linspace(*R_del_x,res)
        )
    )

    labels = ['Frame Def. x', 'Frame Def. y', 'Gantry Def.', 'x-Travel', 'y-Travel', 'Working Temp', 'Fast-travel', 'Lat. Res.']
    # fig, axs = pairwise_problem_space(R,labels=labels)

    # plt.show()


############### Solution Axes Pairwise Plot ########################
    # print("\n")

    # samples = 100

    # dim_8020 = [20, 40]     # 8020 cross-section dimention [mm]
    # gantry_length = [0, 1000]    # Gantry length [mm]
    # frame_side_len = [0, 1000]    # Frame side dimension [mm]
    # plate_dim = [0, 500]     # Plate dimension [mm]
    
    # # ## Discrete Vars
    # # # Steppers
    # # V = 3.06            # Rated voltage for steppers [V]
    # # I_max = 1.7         # Rated current for steppers [A]
    # # L = 0.0038          # Rated inductance for steppers [mH]
    # # phi = 1.8           # Step angle for steppers
    # # steps = 360 / phi   # Steps per revolution for steppers
    # # uStep = 1/8         # Microstep fraction for steppers

    # # #Pulleys
    # # N = 20              # Number of pulley teeth
    # # p = 2               # Tooth pitch for pulleys

    # d_samples = rand_between(dim_8020,samples)
    # G_samples = rand_between(gantry_length,samples)
    # s_samples = rand_between(frame_side_len,samples)
    # q_samples = rand_between(plate_dim,samples)

    # X = np.array([
    #     d_samples,
    #     G_samples,
    #     s_samples,
    #     q_samples
    # ])

    # labels = ['8020 dim', 'Gantry length', 'Frame dim', 'Plate dim']

    # fig, ax = pairwiseScatter(X,sz=5,labels=labels)

    # plt.show()
########## Old Search ########################
    # gantry_length = [8, 20]
    # frame_side_len = [20, 30]
    # plate_thickness = [0.1, 0.2]
    # plate_dim = [1,15]
    # E = [7, 15]  # [psi x 10^-6]
    # max_motor_current = [1, 10]
    # motor_voltage = [1, 12]
    # motor_inductance = [0.0020, 0.0050]
    # pulley_tooth_pitch = [0.5, 3.5]
    # pulley_tooth_qty = [15, 25]
    # phi = [1, 3]
    # uStep = [1/2, 1/4, 1/8, 1/16, 1/32]

    gantry_length = [18, 20]
    frame_side_len = [22, 27]
    plate_thickness = [0.12, 0.15]
    plate_dim = [10,15]
    E = [7, 10]  # [psi x 10^-6]
    max_motor_current = [1, 5]
    motor_voltage = [1, 5]
    motor_inductance = [0.0020, 0.0025]
    pulley_tooth_pitch = [0.5, 1]
    pulley_tooth_qty = [17, 22]
    phi = [1, 1.5]
    uStep = [1/2, 1/4, 1/8]

    form_search = [
        gantry_length,
        frame_side_len,
        plate_thickness,
        plate_dim,
        E,
        max_motor_current,
        motor_voltage,
        motor_inductance,
        pulley_tooth_pitch,
        pulley_tooth_qty,
        phi,
        uStep,
    ]
    
    Reqs = [
        R_del_x,
        R_del_y,
        R_D_x,
        R_D_y,
        R_v,
        R_del_x
    ]
    
    labels = ['G', 's', 't', 'q', 'E', 'Imax',
            'V', 'L', 'p', 'N', 'phi',
            'mu']
    fig, ax = solution_space_search(Reqs, form_search, num_pts=100000, labels=labels)

    plt.show()