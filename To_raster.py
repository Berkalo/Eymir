import numpy as np
from osgeo import gdal
from osgeo import osr
import matplotlib.pylab as plt
from scipy.interpolate import griddata
import pandas as pd
# to install gdal
#pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
data = pd.read_csv("Data/bathy_predict_HV_BORDER_10S.csv").to_numpy()


Lat = data[:,0] - min(data[:,0])
Lon = data[:,1] - min(data[:,1])
Dep = data[:,2]

xLon = np.linspace(min(Lon), max(Lon), 100)
xLat = np.linspace(min(Lat), max(Lat), 100)

lat, lon = np.meshgrid(xLon, xLat)

X, Y = np.meshgrid(xLon, xLat)
array = grid_z1 = griddata((Lat, Lon), Dep, (X, Y), method='linear')

# For each pixel I know it's latitude and longitude.
# As you'll see below you only really need the coordinates of
# one corner, and the resolution of the file.

xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
nrows,ncols = np.shape(array)
xres = (xmax-xmin)/float(ncols)
yres = (ymax-ymin)/float(nrows)
geotransform=(xmin,xres,0,ymax,0, -yres)
# That's (top left x, w-e pixel resolution, rotation (0 if North is up),
#         top left y, rotation (0 if North is up), n-s pixel resolution)
# I don't know why rotation is in twice???

output_raster = gdal.GetDriverByName('GTiff').Create('myraster.tif',ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
srs = osr.SpatialReference()                 # Establish its coordinate encoding
srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                             # Anyone know how to specify the
                                             # IAU2000:49900 Mars encoding?
output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
                                                   # to the file
output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster

output_raster.FlushCache()
