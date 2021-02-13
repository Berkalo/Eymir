import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# some 3-dim points
R_df = pd.read_csv("Data/Right_UTM.xls", sep =";")
L_df = pd.read_csv("Data/UTM_Left.xls", sep =";")
M_df = pd.read_csv("Data/Meas_UTM.xls", sep =";")

#convert df's to numpy array
R = R_df.to_numpy()
L = L_df.to_numpy()
M = M_df.to_numpy()
All_dat = pd.concat([M_df , R_df , L_df ]).to_numpy()

data = np.c_[All_dat[:,0].tolist(), All_dat[:,1].tolist(), All_dat[:,2].tolist()]

# regular grid covering the domain of the data
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX = X.flatten()
YY = Y.flatten()


A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis("auto")
ax.axis('tight')
plt.show()