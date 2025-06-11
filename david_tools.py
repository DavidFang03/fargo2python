import numpy as np

def computephi(x,y):
    # return np.atan2(x,y)
    if x>0:
        if y>=0:
            phi = np.atan(y/x)
        elif y<0:
            phi = np.atan(y/x) + 2*np.pi
    elif x<0:
        phi = phi = np.atan(y/x) + np.pi
    elif x==0:
        if y>0:
            phi = np.pi/2
        elif y<0:
            phi = 3*np.pi/2
        elif y==0:
            return 0
    return phi%(2*np.pi)

