import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import  seaborn as sns
import  tools
from scipy.interpolate import griddata

import pyproj
ls = 50
data = pd.read_csv("Data/Out/bathy_predict_HV_BORDER_10_3S.csv").to_numpy()

Lat_f =  min(data[:,0])
Lon_f = min(data[:,1])

Lat = data[:,0] - Lat_f
Lon = data[:,1] - Lon_f
Dep = data[:,2]

xLon = np.linspace(min(Lon), max(Lon), ls)
xLat = np.linspace(min(Lat), max(Lat), ls)

X, Y = np.meshgrid(xLon, xLat)

Z = grid_z1 = griddata((Lat, Lon), Dep, (X, Y), method='linear')


Z = Z.reshape(ls**2, 1)
X = X.reshape(ls**2, 1)
Y = Y.reshape(ls**2, 1)

X += Lat_f
Y += Lon_f

#Y, X = pyproj.transform(UTM36N, wgs84, X, Y)

result = np.hstack((X, Y, Z))


df =  pd.DataFrame(result,columns=["X_lat", "Y_lon", "Z_depth"])

df.to_csv("/home/berkalp/GeoDBs/Qgis/UTM_36N_predict_pts_no_border.csv", index= False)
