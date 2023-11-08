# cd /sword/apom/notebooks

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster)

# %%
# load modules
## Data processing and DA modules
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
## Dealing with big data and netcdf
import xarray as xr
from netCDF4 import Dataset
## ROMS packages
from xgcm import Grid
## color maps
import cmaps
import cmocean
## mapping packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
## System tools and python configuration
import os
import glob
import repackage
repackage.add('../../')
repackage.add('../')

os.chdir('/sword/apom/notebooks')

# %%

def processROMSGrid(ds):

    coords={'X':{'center':'xi_rho'}, 
        'Y':{'center':'eta_rho'}, 
        'Z':{'center':'s_rho', 'outer':'s_w'}}

    grid = Grid(ds, coords=coords, periodic=[])

    if ds.Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + (ds.zeta+ds.zice) * (1 + Zo_rho/ds.h)
        Zo_w = ds.hc * (ds.s_w - ds.Cs_w) + ds.Cs_w * ds.h
        z_w = Zo_w + (ds.zeta+ds.zice) * (1 + Zo_w/ds.h)
        del Zo_rho, Zo_w
    elif ds.Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = (ds.zeta+ds.zice) + ((ds.zeta+ds.zice) + ds.h) * Zo_rho
        Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
        z_w = Zo_w * ((ds.zeta+ds.zice) + ds.h) + (ds.zeta+ds.zice)
        del Zo_rho, Zo_w
    print('making vertical coordinates')
    
    ds.coords['z_w'] = z_w.where(ds.mask_rho, 0).transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho')
    ds.coords['z_rho'] = z_rho.where(ds.mask_rho, 0).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
    del z_rho, z_w

     # interpolate depth of levels at U and V points
    # ds['z_u'] = grid.interp(ds['z_rho'], 'X', boundary='fill')
    # ds['z_v'] = grid.interp(ds['z_rho'], 'Y', boundary='fill')
    print('made V transform coordinates')

    print('making x/y metrics')

    ds['dx'] = 1/ds.pm
    ds['dy'] = 1/ds.pn

    print('making z metrics')

    ds['dz'] = grid.diff(ds.z_w, 'Z', boundary='fill')


    ds['dA'] = ds.dx * ds.dy

    metrics = {
        ('X',): ['dx'], # X distances
        ('Y',): ['dy'], # Y distances
        ('Z',): ['dz'],# 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v'], # Z distances
        ('X', 'Y'): ['dA'] # Areas
    }
    grid = Grid(ds, coords=coords, metrics=metrics, periodic=[])
    # print('drop some unused vars')
    # ds = ds.drop_vars({'dz_u','dz_w','dz_v','dz_w_u','dz_w_v','z_u','z_v'})
    print('finished')
    print('ds size:',ds.nbytes/1e9,'G')
    return ds,grid

# %%
# initialise OHC arrays and constants
TotalOHC = np.array(())
TotalOHC_1000 = np.array(())
TotalOHC_subIce = np.array(())
time = np.array(())
rho0=1026
cp0=4181.3

# %%
# load grid from first file
datelist = np.array(range(14,17,1))
FilePath = '../data/raw/roms_avg_'

dates = datelist[0]
filename=FilePath+str(dates).zfill(4)+'.nc'
print('opening ',filename)
ds_raw=xr.open_dataset(filename)
ds_raw = ds_raw[['temp','zeta','zice','h','Vtransform','hc','Cs_r','s_w','s_rho','Cs_w','mask_rho','pm','pn']]
print(ds_raw.nbytes/1e9,'G')
ds_raw,grid = processROMSGrid(ds_raw)
dV = ds_raw.dA*ds_raw.dz.mean(dim='ocean_time')


# %%
# load and loop single ROMS netcdf with xr
for dates in datelist:
    filename=FilePath+str(dates).zfill(4)+'.nc'
    print('opening ',filename)
    ds=xr.open_dataset(filename)
    ds = ds[['temp','zeta','zice','h','Vtransform','hc','Cs_r','s_w','s_rho','Cs_w','mask_rho','pm','pn']]
    print(ds.nbytes/1e9,'G')
    
    

    # ds,grid = processROMSGrid(ds_raw)
    # del ds_raw
    
    OHC=1026*4181.3*dV*(ds.temp+273.15)
    
    # # calculate the time-series of OHC for this netcdf, prepare to append them to long series
    # TotalOHC = OHC.sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho')
    # TotalOHC_1000 = OHC.where(ds.z_rho.mean(dim='ocean_time')>-1000, drop=True).sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho')
    # TotalOHC_subIce = OHC.where(ds.zice<0,drop=True).sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho')
    
    TotalOHC = np.append(TotalOHC,OHC.sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho'))
    TotalOHC_1000 = np.append(TotalOHC_1000,OHC.where(ds_raw.z_rho.mean(dim='ocean_time')>-1000, drop=True).sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho'))
    TotalOHC_subIce =np.append(TotalOHC_subIce,OHC.where(ds_raw.zice<0,drop=True).sum(dim='xi_rho').sum(dim='eta_rho').sum(dim='s_rho'))
    # melt = np.append(melt,ds.m)
    time = np.append(time,ds.ocean_time.values.astype('float64')/(86400*365*1e9))
    del ds, OHC

# %%
# plot final metrics
plt.plot(time,TotalOHC)
plt.show()
plt.plot(time,TotalOHC_1000)
plt.show()
plt.plot(time,TotalOHC_subIce)
plt.show()


# %%
