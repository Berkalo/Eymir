import pandas as pd
import numpy as np
import scipy as sp
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


#find the closest 2 points on M and R wrt x

r_best = np.array([0,0,0])
wing_dist = np.array([0])
for l in L:
    #find closest right point to the given left wing on right wing
    r_pt = tools.closest(l,R)["point"]
    r_dist = tools.closest(l,R)["distance"]

    #collect closest wing points
    r_best = np.vstack((r_best,r_pt))
    wing_dist = np.vstack((wing_dist, r_dist))

r_best = np.delete(r_best,0,0)
wing_dist = np.delete(wing_dist,0,0)

#find closest 2 points of coupled closest RL couple

RL_best = np.hstack((L[:,:2],r_best[:,:2]))
print(RL_best.shape)

trps = np.array([0]*10) #keep record of triplets
for m in M:
    dst_best = 9999999999
    for rl in RL_best:
        pt =  tools.project_on_vector(rl[0:2], rl[2:], m[:3])["Point"]
        ln =  tools.project_on_vector(rl[0:2], rl[2:], m[:3])["Line"]
        dst = tools.project_on_vector(rl[0:2], rl[2:], m[:3])["Distance"]

        if dst < dst_best:

            m_sel_arr = np.hstack((m,rl,pt,dst)) #get closest line for the given measurement

    trps = np.vstack((trps, m_sel_arr))
    """0,1,2 m
        3,4 --> L 
        5,6 --> R 
        7,8 --> pt
        9 --> dist"""

trps = trps[1:]
print(trps[1])

print(pt)

#plot the figure preliminary

sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
print(RL_best[:,0].shape)
print(RL_best[:,1].shape)
print(pt[0])

zrs = np.array([0]*2*trps[:,6].shape[0])
x = np.hstack((trps[:,3],trps[:,5], trps[:,7]))
y = np.hstack((trps[:,4], trps[:,6], trps[:,8]))
z = np.hstack((zrs, trps[:,2]))


ax.set_xlabel("Lon")
ax.set_ylabel("Lat")
ax.set_zlabel("Depth")

ax.scatter(x, y, z)
plt.show()
