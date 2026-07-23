import xarray as xr
import numpy as np
import scipy.interpolate
import dask as da

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    newlength = int(len(x) * scale)
    y = np.linspace(x0, xlast, num=newlength, endpoint=False)
    return y

def blocks(data, width=352):
    #n = data.t.shape[0]
    w = data.x.shape[0]
    h = data.y.shape[0]
    d = data.band.shape[0]

    hs = np.arange(0, h, width)
    ws = np.arange(0, w, width)
    blocks = []
    for hindex in hs:
        if hindex+width > h:
            hindex = h - width

        for windex in ws:
            if windex+width > w:
                windex = w - width
            blocks.append(data.sel(y=data.y.values[hindex:hindex+width],
                                   x=data.x.values[windex:windex+width]))
    return blocks

def block_dask_array(arr, axis, size=128, stride=128):
    arr = da.array.swapaxes(arr, axis, 0)
    n = arr.shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr[j:j+size])
    stack = da.array.stack(stack)
    stack = da.array.swapaxes(stack, axis+1, 1)
    return stack

def block_array(arr, axis, size=128, stride=128):
    arr = np.swapaxes(arr, axis, 0)
    n = arr.shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr[np.newaxis,j:j+size])
    stack = np.concatenate(stack, 0)
    stack = np.swapaxes(stack, axis+1, 1)
    return stack

def xarray_to_block_list(arr, dim, size=128, stride=128):
    n = arr[dim].shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr.isel({dim: np.arange(j,j+size)}))
    return stack

def interp(da, scale, fillna=False):
    xnew = interp_dim(da['x'].values, scale)
    ynew = interp_dim(da['y'].values, scale)
    newcoords = dict(x=xnew, y=ynew)
    return da.interp(newcoords)

def regrid_2km(da, band):
    if band == 2:
        return interp(da, 1. / 4, fillna=False)
    elif band in [1, 3, 5]:
        return interp(da, 1. / 2, fillna=False)
    return da

def regrid_1km(da, band):
    if band == 2: #(0.5 km)
        return interp(da, 1./2, fillna=False)
    elif band not in [1, 3, 5]: # 2km
        return interp(da, 2., fillna=False)
    return da

def regrid_500m(da, band):
    if band == 2: # 500m
        return da
    elif band in [1, 3, 5]: # 1km
        return interp(da, 2., fillna=False)
    return interp(da, 4., fillna=False) # 2km


def cartesian_to_speed(da, time_seconds=1800):
    lat = da.lat.values
    lon = da.lon.values
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError('cartesian_to_speed requires regular 1D lat and lon coordinates')

    if len(lat) < 2 or len(lon) < 2:
        raise ValueError('cartesian_to_speed requires at least two latitude and longitude points')

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = np.diff(lat_rad).mean()
    dlon = np.diff(lon_rad).mean()
    earth_radius = 6371000.0

    dy = earth_radius * dlat
    dx = earth_radius * np.cos(lat_rad)[:, None] * dlon

    da['U'] = da['U'] * dx / time_seconds
    da['V'] = da['V'] * dy / time_seconds
    return da
    
    
def speed_to_cartesian(da, time_seconds=1800):
    lat = da.lat.values
    lon = da.lon.values
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError('speed_to_cartesian requires regular 1D lat and lon coordinates')

    if len(lat) < 2 or len(lon) < 2:
        raise ValueError('speed_to_cartesian requires at least two latitude and longitude points')

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = np.diff(lat_rad).mean()
    dlon = np.diff(lon_rad).mean()
    earth_radius = 6371000.0

    dy = earth_radius * dlat
    dx = earth_radius * np.cos(lat_rad)[:, None] * dlon

    da['U'] = da['U'] / dx * time_seconds
    da['V'] = da['V'] / dy * time_seconds
    return da
