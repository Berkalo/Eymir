import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tools
import Rotate


#read data frames
R_df = pd.read_csv("Data/Right_cs_TableToExcel.csv", sep =",").iloc[:,3:]
L_df = pd.read_csv("Data/Left_cs_TableToExcel.csv", sep =",").iloc[:,3:]
M_df = pd.read_csv("Data/Meas_UTM.xls", sep =";")

"""print(R_df.head(5))
print("_________________")
print(L_df.head(5))
print("_________________")
print(M_df.head(5))
"""
R = R_df.to_numpy()
L = L_df.to_numpy()
M = M_df.to_numpy()
x_unit = np.array([1, 0, 0])

"""R_Nres = tools.normalize_by_axis(R)
L_Nres = tools.normalize_by_axis(L)
M_Nres = tools.normalize_by_axis(M)

R = R_Nres["normalized_v"]
L = L_Nres["normalized_v"]
M = M_Nres["normalized_v"]"""

"""
All_dat = pd.concat([M_df , R_df , L_df ])
print(All_dat)

#plot the figure preliminary
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
overall_pred = np.array([0,0,0])
for num in range(0, 30):
    l = L[num,:]
    r = R[num, :]
    m = M[num, :]

    f_shiftx = l[0]
    f_shifty = l[1]

    l[0] = l[0] - f_shiftx
    l[1] = l[1] - f_shifty

    r[0] = r[0] - f_shiftx
    r[1] = r[1] - f_shifty

    m[0] = m[0] - f_shiftx
    m[1] = m[1] - f_shifty

    vector_region = tools.get_region(m)

    if vector_region  == 1:
        angle_wx =  Rotate.angle_between(x_unit,np.hstack((m[:2], 0)))
        m[:3] = Rotate.z_rotation(m[:3],  -angle_wx)

    elif vector_region  == 2:
        angle_wx =  Rotate.angle_between(x_unit,np.hstack((m[:2], 0)))
        m[:3] = Rotate.z_rotation(m[:3], angle_wx)

    elif vector_region  == 3:
        angle_wx =  Rotate.angle_between(x_unit,np.hstack((m[:2], 0)))
        m[:3] = Rotate.z_rotation(m[:3], angle_wx)


    elif vector_region  == 4:
        angle_wx =  Rotate.angle_between(x_unit,np.hstack((m[:2], 0)))
        m[:3] = Rotate.z_rotation(m[:3], angle_wx)


    if vector_region  == 1:
        angle_wxr =  Rotate.angle_between(x_unit,np.hstack((r[:2], 0)))
        r[:3] = Rotate.z_rotation(r[:3],  -angle_wxr)

    elif vector_region  == 2:
        angle_wxr=  Rotate.angle_between(x_unit,np.hstack((r[:2], 0)))
        r[:3] = Rotate.z_rotation(m[:3], angle_wxr)

    elif vector_region  == 3:
        angle_wxr =  Rotate.angle_between(x_unit,np.hstack((r[:2], 0)))
        r[:3] = Rotate.z_rotation(r[:3], angle_wxr)

    elif vector_region  == 4:
        angle_wxr =  Rotate.angle_between(x_unit,np.hstack((r[:2], 0)))
        r[:3] = Rotate.z_rotation(r[:3], angle_wxr)


    m[1] = 0
    r[1] = 0

    #angle used will be based on left pt and the measurement pt, r to be decided by distance
    model = tools.poly_model(m[0], r[0], m[2])
    z_peak = r[0]/2
    division = 30
    x_peak_left = np.linspace(0, z_peak, division)
    x_peak_right = np.linspace(z_peak,r[0], division)

    x_pred_pts = np.hstack((x_peak_left, x_peak_right))
    z_predicted = np.zeros(x_pred_pts.shape)

    for x in range(0,x_pred_pts.shape[0]):
        z_predicted[x] = tools.predict(model, x_pred_pts[x])

    y_pred_pts = np.array([0]*2*division)

    z_predicted = z_predicted
    x_pred_pts = x_pred_pts

    pred = np.vstack((x_pred_pts, y_pred_pts, z_predicted)).T

    rtd_back = np.array([0, 0, 0])
    for num in range(1,pred.shape[0]):
        v = pred[num]

        if vector_region == 1:
            v[:3] = Rotate.z_rotation(v[:3], angle_wx)

        elif vector_region == 2:
            v[:3] = Rotate.z_rotation(v[:3], -angle_wx)

        elif vector_region == 3:
            v[:3] = Rotate.z_rotation(v[:3], -angle_wx)

        elif vector_region == 4:
            v[:3] = Rotate.z_rotation(v[:3], -angle_wx)


        v[0] = v[0] + f_shiftx
        v[1] = v[1] + f_shifty
        rtd_back = np.vstack((rtd_back, v))

    rtd_back = rtd_back[1:,:]
    print(rtd_back.shape)
    overall_pred = np.vstack((overall_pred, rtd_back))

overall_pred = overall_pred[1:, :]


R_all = pd.read_csv("Data/Right_UTM.xls", sep =";")
L_all = pd.read_csv("Data/UTM_Left.xls", sep =";")

R_all = R_all.to_numpy()
L_all = L_all.to_numpy()

data_plot = np.vstack((R_all, L_all, overall_pred))

#plot the figure preliminary
sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = data_plot[:,0]
y = data_plot[:,1]
z = data_plot[:,2]

ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Depth")

ax.scatter(x, y, z)
plt.show()

to_csv =  pd.DataFrame(overall_pred,columns=["X_lat", "Y_lon", "Z_depth"])
print(to_csv)