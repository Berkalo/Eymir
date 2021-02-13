import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tools


#read data frames
R_df = pd.read_csv("Data/Right_UTM.xls", sep =";")
L_df = pd.read_csv("Data/UTM_Left.xls", sep =";")
M_df = pd.read_csv("Data/Meas_UTM.xls", sep =";")

#convert df's to numpy array
R = R_df.to_numpy()
L = L_df.to_numpy()
M = M_df.to_numpy()

All_dat = pd.concat([M_df , R_df , L_df ])

'''
print(R_df.head(5))
print("_________________")
print(L_df.head(5))
print("_________________")
print(M_df.head(5))
print("_________________")
print(M_df["East (X)"])
'''

"""#plot the figure preliminary
sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = All_dat['East (X)']
y = All_dat['North (Y)']
z = All_dat['Z__depth_']

ax.set_xlabel("Lon")
ax.set_ylabel("Lat")
ax.set_zlabel("Depth")

ax.scatter(x, y, z)
plt.show()
"""
L_best =np.array([0]*3)
R_best =np.array([0]*3)

for m in M:
    L_closest = tools.closest(m, L)["Point"]
    mL_Line = tools.line2pts(m, L_closest)
    slope_line = (m[1]-L_closest[1])/(m[0]-L_closest[0])
    ktsy = 1
    x_extend = 800
    y_extend = x_extend*slope_line
    m_extend = np.array([m[0]+ x_extend, m[1]+ y_extend])
    cond = 0

    L_best = np.vstack((L_best, L_closest))
    min_dist = 100000
    for r in range(1, R.shape[0]):

        Rpt1 = R[r-1]
        Rpt2 = R[r]
        intersect = tools.pt_intersection(m,m_extend,  Rpt1, Rpt2)

        if intersect.geom_type != 'LineString':
            to_be_selected_R= np.array(intersect)
            min_dist = 100000
            dist = tools.dist2pt(to_be_selected_R, m)

            if dist < min_dist:
                selected_R = to_be_selected_R

            selected_R = np.hstack([selected_R, 0])

    R_best = np.vstack((R_best,selected_R))
    print(selected_R)
    print("_____")



L_best = L_best[1:]
R_best = R_best[1:]
print(R_best.shape)
arr = np.vstack((L_best, M,R_best))

#plot the figure preliminary
sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = arr[a:90:30,0]
y = arr[a:90:30,1]
z = arr[a:90:30,2]


ax.set_xlabel("Lon")
ax.set_ylabel("Lat")
ax.set_zlabel("Depth")

ax.scatter(x, y, z)
plt.show()

