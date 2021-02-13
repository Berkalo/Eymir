import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Finds angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

if __name__ == "__main__":

    L_predef = np.array([-10, 10,    0.   ])  # blue vector
    on_x = np.array([1, 0, 0])
    a_btw = angle_between(on_x, L_predef)
    new_vect = z_rotation(L_predef, 1/8*np.pi)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(np.linspace(0, L_predef[0]), np.linspace(0, L_predef[1]), np.linspace(0, L_predef[2]))
    ax.plot(np.linspace(0, new_vect[0]), np.linspace(0, new_vect[1]), np.linspace(0, new_vect[2]))

    print(new_vect)
    plt.show()

