
import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
sns.set_theme(style="darkgrid")

df = pd.read_csv("Data/bathy_predict_HV_BORDER_10S.csv")

df.columns = ["X", "Y", "Z"]

# And transform the old column name in something numeric
df['X'] = pd.Categorical(df['X'])
df['X'] = df['X'].cat.codes

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
# to Add a color bar which maps values to colors.
surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)


# Rotate it
ax.view_init(30, 45)
plt.show()
