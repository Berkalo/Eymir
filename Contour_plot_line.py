import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import  seaborn as sns
import  tools
from scipy.interpolate import griddata

data = pd.read_csv("Data/Out/bathy_predict_HV_BORDER_10_3S.csv").to_numpy()

Lat_f =  min(data[:,0])
Lon_f = min(data[:,1])

Lat = data[:,0] - Lat_f
Lon = data[:,1] - Lon_f
Dep = data[:,2]

xLon = np.linspace(min(Lon), max(Lon), 50)
xLat = np.linspace(min(Lat), max(Lat), 50)

X, Y = np.meshgrid(xLon, xLat)

Z = grid_z1 = griddata((Lat, Lon), Dep, (X, Y), method='linear')

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
cbar = fig.colorbar(cp) # Add a colorbar to a plot

ax.set_title('Filled Contours Plot')
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')

cbar.ax.set_ylabel('Depth (m)', rotation=270)
plt.show()
