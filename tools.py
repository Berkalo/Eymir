import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from skspatial.objects import Point, Line
from skspatial.plotting import plot_2d
import shapely
from shapely.geometry import LineString, Point


def closest(pt, arr):
    '''Finds the closest point to a given A point
    format of A: [##, ##]
    format of B: [[##, ##],[##, ##],[##, ##]....]'''
    x1 = pt[0]
    y1 = pt[1]

    x2 = arr[:, 0]
    y2 = arr[:, 1]

    dists = np.sqrt(np.power(x2-x1, 2) + np.power(y2-y1, 2))

    info = {"Distance": min(dists)["min"],
            "Point": arr[min(dists)["idx"]]}
    return info

def min(arr):
    count = 0
    minv = arr[0]
    min_idx = 0

    for x in arr:
        if x < minv:
            minv = x
            min_idx = count
        count += 1

    result = {"min": minv,
              "idx": min_idx}

    return result

def line2pts(A, B):
    '''in the form of y = mX+B that passes through points A ,and B'''
    m = (B[1] - A[1])/(B[0]-B[0])
    line = {"m": m,
            "b": A[1]-m*B[0]}
    return line #y = mx+b

def project_on_vector(L, R, M):
    '''gets left and closest Right and measured points, returns projection on the vector btw. R&L of M'''
    line = Line(point = L[0:2], direction=R[0:2])
    point = Point(M[0:2])

    line_const = line2pts(L, R)

    point_projected = line.project_point(point)
    line_projection = Line.from_points(point, point_projected)

    result = {"Point" : point_projected,
              "Line" : line_projection,
              "Distance": distance_pt2line(line_const["m"], line_const["b"], M[0:2])}
    return result

def get_idx(pt, L):
    """gives the index of the given point in the given array L"""
    ct = 0
    for l in L:
        if pt == l:
            return pt
        ct +=1

    return False

def distance_pt2line(m, b, pt):
    "distance of the point pt and line having slope of m and constant of b"
    return abs(-m*pt[0]+ pt[1] - b)/math.sqrt((m**2+4))

def dist2pt(A, B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def pt_intersection(A,B, C,D):

    line1 = LineString([A[:2], B[:2]])
    line2 = LineString([C[:2], D[:2]])

    int_pt = line1.intersection(line2)
    #point_of_intersection = int_pt.x, int_pt.y

    return int_pt
def normalize_by_axis(A):
    norm_x = np.linalg.norm(A[:, 0])
    norm_y = np.linalg.norm(A[:, 1])

    A[:, 0] = A[:, 0] / norm_x *1000
    A[:, 1] = A[:, 1] / norm_y *1000

    result = {"norm_x": norm_x,
              "norm_y": norm_y,
              "normalized_v": A}
    return result

def unnormalize_by_axis(norm_x, norm_y, A):
    A[:, 0] = A[:, 0] * norm_x / 1000
    A[:, 1] = A[:, 1] * norm_y / 1000
    return A


def get_region(A):
    if A[0]> 0:
        if A[1] > 0:
            return 1
        elif A[1]< 0:
            return 4
    elif A[0] < 0:
        if A[1] > 0:
            return 2
        elif A[1]< 0:
            return 3
    else:
        return 0

def poly_model(x_kn, x_end, z_known):
    """ form of the fitting model is z = Ax(x-B)
        where B = x_end
        function
    """
    B = x_end
    A = z_known/(x_kn*(x_kn -B))

    return {"B": B, "A": A}

def predict(model, x):
    "predicts x for a given model"
    B = model["B"]
    A = model["A"]
    return A*x*(x-B)



if __name__ == "__main__":
    #test functions
    print(np.arange(30).reshape(15,2) + 10)
    print(closest(np.array([20, 20]),np.arange(30).reshape(15,2) + 10))

    line = Line(point=[0, 0], direction=[1, 1])
    point = Point([1, 4])

    point_projected = line.project_point(point)
    line_projection = Line.from_points(point, point_projected)
    print(point_projected)