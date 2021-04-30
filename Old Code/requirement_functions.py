from warnings import warn
import sys
import matplotlib.pyplot as plt
import numpy as np


## Frame deflection
def inplane_plate_deflection(mass=3.7, acceleration=200.0, **kwargs):
    """
    Calculates the maximum inplane bending deflection of rectangular plates due
    to acceleration of the print head.

    Arguments:
        acceleration (float): Expected max acceleration of the print head [in/s^2]
        frame_side_length (float or array): Side length of cuboid frame [in]
        mass (float or array): Total mass of objects under acceleration [lbm]
        plate_width (float or array): planar width of plate [in]
        plate_thickness (float or array): out-of plane dimension of plate [in]
        modulus (float or array): Plate material modulus of elasticity [psi]

    Returns (np.ndarray): Maximum deflection values
    """

    assert all([type(i) in [float, np.ndarray] for i in kwargs.values()]), \
            "All arguments must be floats or numpy arrays"    
    
    L = kwargs['frame_side_length']
    F = mass * acceleration / (12 * 32.174)
    I = kwargs['plate_width']**3 * kwargs['plate_thickness'] / 12
    E = kwargs['modulus']
    
    return F * L**3 / (12 * E * I)

## Gantry deflection
    # Ansys interface needed

## Lateral travel
def x_travel(head_width_x=2.5, **kwargs):
    """
    Calculates the maximum nozzle travel in the x-direction allowed by frame.

    Arguments:
        head_width_x (float): Dimension of print head in x-direction [in]
        gantry_length (float or array): Unsupported length of gantry beam [in]

    Returns (np.ndarray): Maximum travel values [in]
    """
    assert all([type(i) in [float, np.ndarray] for i in kwargs.values()]), \
            "All arguments must be floats or numpy arrays"

    return kwargs['gantry_length'] - head_width_x


def y_travel(extrusion_dimension=1.77, head_width_y=2.5, **kwargs):
    """
    Calculates the maximum nozzle travel in the x-direction allowed by frame.

    Arguments:
        extrusion_dimension (float): Dimension of aluminum extrusion [in]
        head_width_y (float): Dimension of print head in x-direction [in]
        frame_side_length (float or array): Side length of cuboid frame [in]

    Returns (np.ndarray): Maximum travel values [in]
    """
    assert all([type(i) in [float, np.ndarray] for i in kwargs.values()]), \
            "All arguments must be floats or numpy arrays"

    return kwargs['frame_side_length'] - head_width_y - 2*extrusion_dimension


## Working temperature

## Fast-travel velocity
def _stepper_speed(**kwargs):
    # Credit: https://dyzedesign.com/2016/11/printing-300-mm-s-part-2-calculations/
    #         https://www.daycounter.com/Calculators/Stepper-Motor-Calculator.phtml
    """
    Calculate the speed of the motor at max torque

    Arguments:
        motor_voltage (float or array): operating voltage [V]
        motor_inductance (float or array): phase inductance [H]
        max_motor_current (float or array): maximum current draw [A]
        step_angle (float or array): angle between steps [deg]
    
    Returns (float): motor speed [Hz]
    """

    # Convert step angle to radians
    # This is the change in position for one step
    dTheta = np.pi*kwargs['step_angle'] / 180

    # Calculate dt as the time needed to complete one step
    # Use the time derivative of formula relating V, I, and L
    #       dt = (L/V)*dI
    # One step: 0 -> I -> 0, so dI = 2I
    dt = 2*kwargs['max_motor_current']*kwargs['motor_inductance'] \
        / kwargs['motor_voltage']
    
    return dTheta / dt


def nozzle_speed(**kwargs):
    """
    Calculate the speed of the extrueder nozzle at max motor speed

    Arguments:
        pulley_tooth_qty (float): number of pulley teeth
        pulley_tooth_pitch (float): arc length between teeth
    
    Returns (float): nozzle speed [mm/s]
    """

    omega = _stepper_speed(**kwargs)
    r = kwargs['pulley_tooth_pitch'] * kwargs['pulley_tooth_qty']/ (2*np.pi)

    return r * omega / 25.4


## Lateral print resolution
def resolution(**kwargs):
    """
    Calculate the speed of the extrueder at max pulley speed

    Arguments:
        pulley_tooth_qty (float or array): number of pulley teeth [#]
        pulley_tooth_pitch (float or array): arc length between teeth [mm]
        step_angle (float or array): step angle [degree]
        microstep (float or array): micro-step [fraction]
    
    Returns (float): x-y resolution [mm]
    """

    # Equation:
    ### Circumference: C = p*N
    ### Pulley radius: r = C/2*pi = p*N/2*pi
    ### Step angle to rad: theta = 2*pi*step_angle/180
    ### Distance for full step: d = r*theta = 2*pi*p*N*step_angle/2*pi*180
    ### Distance for microstep (2*pi canceled): d_u = uStep*p*N*step_angle/180
    ### Divide by 2 due to coreXY design: d_u = uStep*p*N*step_angle/360

    return 0.0393701 * kwargs['microstep'] * kwargs['pulley_tooth_pitch'] \
            * kwargs['pulley_tooth_qty'] * kwargs['step_angle'] / 360


## Unit Tests
def _test_deflection(test_point):

    tol = 0.0000000000001
    tgt = 0.0000394284649
    ans = inplane_plate_deflection(**test_point)
    assert abs(ans - tgt) < tol, "Deflection function error"
    

def _test_travels(test_point):

    tgt_x = 11.5
    ans_x = x_travel(**test_point)
    assert ans_x==tgt_x, "x-travel function error"

    tgt_y = 17.96
    ans_y = y_travel(**test_point)
    assert ans_y==tgt_y, "y-travel function error"


def _test_speeds(test_point):

    # Stepper speed function test
    tol = 0.0000000000001
    tgt1 = 7.4406141795547
    ans1 = _stepper_speed(**test_point)
    assert abs(ans1 - tgt1) < tol, "Stepper speed function error"

    # Carriage speed function test
    tgt2 = 1.8648984666390391
    ans2 = nozzle_speed(**test_point)
    assert abs(ans2 - tgt2) < tol, "Nozzle speed function error"


def _test_resolution(test_point):
    # Stepper speed function test
    tol = 0.0000000001
    tgt = 0.0009842525
    ans = resolution(**test_point)
    assert abs(ans - tgt) < tol, "Resolution function error"


def _test_array_output(test_point):
    for key,item in test_point.items():
        test_point[key] = np.array([item, item])
    
    try:
        d = inplane_plate_deflection(**test_point)
        x = x_travel(**test_point)
        y = y_travel(**test_point)
        v = nozzle_speed(**test_point)
        res = resolution(**test_point)
    except:
        e = sys.exc_info()[0]
        print(f'Encountered error {e} when attempting array inputs')

    assert all([type(i)==np.ndarray for i in [d,x,y,v,res]]), \
            "Array output error"


def run_test():
    test_point = {
        'frame_side_length': 24.0,
        'plate_width': 8.0,
        'plate_thickness': 0.125,
        'modulus': 10.5*10**6,
        'gantry_length': 14.0,
        'motor_voltage': 3.06,
        'motor_inductance': 0.0038,
        'max_motor_current': 1.7,
        'step_angle': 1.8,
        'pulley_tooth_pitch': 2.0,
        'pulley_tooth_qty': 20.0,
        'microstep': 1/8,
    }
    
    _test_deflection(test_point)
    _test_travels(test_point)
    _test_speeds(test_point)
    _test_resolution(test_point)
    _test_array_output(test_point)

######## Unit Test ###########

if __name__ == "__main__":
    # Execute unit testing
    run_test()

# # Params
#     s = [24.0]
#     q = [8.0]
#     t = [0.125]
#     E = [10.5*10**6]
#     G = [14.0]
#     I = [1.7]
#     V = [3.06]
#     L = [0.0038]
#     phi = [1.8]
#     mu = [1/8]
#     p = [2]
#     n = [20]

# Params
    s = [21.95894239, 23.11840423, 26.98397245]
    q = [5.11657883, 5.64964438, 4.33289803]
    t = [0.12985497, 0.14821085, 0.13849184]
    E = [10732443.43825923,  9549690.13575877, 10529823.94495566]
    G = [16.74548962, 13.91176004, 12.67051795]
    I = [1. , 1.5, 1.5]
    V = [6., 4., 4.]
    L = [0.0106, 0.0032, 0.0032]
    phi = [1.8, 1.8, 1.8]
    mu = [0.25  , 0.0625, 0.0625]
    p = [1.01252723, 2.19347265, 2.55178315]
    n = [33, 42, 39]

# Dict
    test_case = {
        'frame_side_length': np.array(s),
        'plate_width': np.array(q),
        'plate_thickness': np.array(t),
        'modulus': np.array(E),
        'gantry_length': np.array(G),
        'max_motor_current': np.array(I),
        'motor_voltage': np.array(V),
        'motor_inductance': np.array(L),
        'step_angle': np.array(phi),
        'microstep': np.array(mu),
        'pulley_tooth_pitch': np.array(p),
        'pulley_tooth_qty': np.array(n),
    }

# Function calls
    d = inplane_plate_deflection(**test_case)
    x = x_travel(**test_case)
    y = y_travel(**test_case)
    v = nozzle_speed(**test_case)
    res = resolution(**test_case)

    print(d,x,y,v,res)