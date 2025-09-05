import numpy as np

def Integrator(x0, a, dt):
    v = x0[1] + a*dt
    x = x0[0] + x0[1]*dt +0.5*a/2*dt**2
    return np.array([x, v])