import os
import re
import datetime as dt

import h5py
import numpy as np
import xarray as xr
import pyresample


def _as_scalar_float(val, default=0.0):
    if val is None:
        return float(default)
    try:
        arr = np.asarray(val)
        if arr.size > 0:
            return float(arr.flat[0])
    except Exception:
        pass
    return float(default)


def _decode_scale_offset(dataset, arr):
    scale_factor = _as_scalar_float(dataset.attrs.get('scale_factor', 1.0), 1.0)
    add_offset = _as_scalar_float(dataset.attrs.get('add_offset', 0.0), 0.0)
    return arr.astype(np.float32) * np.float32(scale_factor) + np.float32(add_offset)



class INSATL1BBand(object):
    """
    Class to manipulate INSAT L1B HDF5 files for the water vapor channel.
    """
    def __init__(self, fpath):
        self.fpath = fpath
        self.data = None
        self.attrs = {}
        self.datetime = self._datetime_from_filename(fpath)

        with h5py.File(self.fpath, 'r') as f:
            self.attrs = {k: v for k, v in f.attrs.items()}
            if self.datetime is None:
                date_string = self.attrs.get('Acquisition_Date', b'').decode('utf-8')
                start_time = self.attrs.get('Acquisition_Start_Time', b'').decode('utf-8')
                if date_string and start_time:
                    try:
                        date_obj = dt.datetime.strptime(date_string, '%d%B%Y')
                    except ValueError:
                        date_obj = None
                    try:
                        time_obj = dt.datetime.strptime(start_time, '%d-%b-%YT%H:%M:%S')
                    except ValueError:
                        time_obj = None
                    if date_obj is not None and time_obj is not None:
                        self.datetime = dt.datetime(
                            date_obj.year,
                            date_obj.month,
                            date_obj.day,
                            time_obj.hour,
                            time_obj.minute,
                            time_obj.second,
                        )

    def _datetime_from_filename(self, fpath):
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d{2}[A-Z]{3}\d{4})_(\d{4})_', fname)
        if not m:
            return None
        date_part = m.group(1)
        time_part = m.group(2)
        try:
            dt_obj = dt.datetime.strptime(date_part + time_part, '%d%b%Y%H%M')
        except ValueError:
            try:
                dt_obj = dt.datetime.strptime(date_part + time_part, '%d%B%Y%H%M')
            except ValueError:
                return None
        return dt_obj

    def open_dataset(self, rescale=True, force=False, chunks=None):
        if (not hasattr(self, 'data')) or self.data is None or force:
            with h5py.File(self.fpath, 'r') as f:
                counts = f['IMG_WV'][0].astype(np.int32)
                fill_value = f['IMG_WV'].attrs.get('_FillValue', None)
                temp_lut = f['IMG_WV_TEMP'][()]
                radiance_lut = f['IMG_WV_RADIANCE'][()]

                if rescale:
                    arr = temp_lut[counts]
                else:
                    arr = counts.astype(np.float32)

                if fill_value is not None:
                    mask = counts == fill_value
                    arr = arr.astype(np.float32)
                    arr[mask] = np.nan
                else:
                    mask = np.zeros_like(arr, dtype=bool)

                lat = _decode_scale_offset(f['Latitude'], f['Latitude'][()])
                lon = _decode_scale_offset(f['Longitude'], f['Longitude'][()])
                lat = lat.astype(np.float32)
                lon = lon.astype(np.float32)
                lat[~np.isfinite(lat)] = np.nan
                lon[~np.isfinite(lon)] = np.nan
                # mask scaled fill values (dataset stores integer fill; after scaling it becomes fractional)
                lat_fill = _as_scalar_float(f['Latitude'].attrs.get('_FillValue', 32767), 32767.0)
                lat_scale = _as_scalar_float(f['Latitude'].attrs.get('scale_factor', 1.0), 1.0)
                lat_fill_scaled = lat_fill * lat_scale

                lon_fill = _as_scalar_float(f['Longitude'].attrs.get('_FillValue', 32767), 32767.0)
                lon_scale = _as_scalar_float(f['Longitude'].attrs.get('scale_factor', 1.0), 1.0)
                lon_fill_scaled = lon_fill * lon_scale

                lat[np.isclose(lat, lat_fill_scaled)] = np.nan
                lon[np.isclose(lon, lon_fill_scaled)] = np.nan
                geo_x = f['GeoX'][()] if 'GeoX' in f else np.arange(arr.shape[1])
                geo_y = f['GeoY'][()] if 'GeoY' in f else np.arange(arr.shape[0])

                da = xr.DataArray(
                    arr,
                    coords=dict(y=geo_y, x=geo_x),
                    dims=('y', 'x'),
                    attrs=dict(
                        units='K' if rescale else 'count',
                        long_name='INSAT Water Vapor Brightness Temperature'
                        if rescale
                        else 'INSAT Water Vapor Counts',
                    ),
                )

                ds = xr.Dataset(dict(Rad=da))
                ds['Latitude'] = xr.DataArray(lat, coords=dict(y=geo_y, x=geo_x), dims=('y', 'x'))
                ds['Longitude'] = xr.DataArray(lon, coords=dict(y=geo_y, x=geo_x), dims=('y', 'x'))
                ds['IMG_WV'] = xr.DataArray(counts, coords=dict(y=geo_y, x=geo_x), dims=('y', 'x'))
                ds['IMG_WV_TEMP_LUT'] = xr.DataArray(temp_lut, dims=('count',))
                ds['IMG_WV_RADIANCE_LUT'] = xr.DataArray(radiance_lut, dims=('count',))
                ds.attrs.update(self.attrs)

                self.data = ds

        return self.data

    def latlon(self):
        if not hasattr(self, 'data'):
            self.open_dataset()
        return self.data['Latitude'].values, self.data['Longitude'].values

    def reproject_to_latlon(self, chunks=None, bounds=None, resolution=0.04):
        data = self.open_dataset(chunks=chunks)
        lats = data['Latitude'].values.astype(np.float32)
        lons = data['Longitude'].values.astype(np.float32)
        rad = data['Rad'].values.astype(np.float32)

        if bounds is None:
            lat_min = np.nanmin(lats)
            lat_max = np.nanmax(lats)
            lon_min = np.nanmin(lons)
            lon_max = np.nanmax(lons)
        else:
            lat_min, lat_max, lon_min, lon_max = bounds

        # ensure lat/lon grid covers the satellite coverage
        lat_range = np.arange(lat_min, lat_max + resolution, resolution, dtype=np.float32)
        lon_range = np.arange(lon_min, lon_max + resolution, resolution, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

        source_def = pyresample.geometry.SwathDefinition(lats=lats, lons=lons)
        target_def = pyresample.geometry.GridDefinition(lats=lat_grid, lons=lon_grid)
        neighbor_info = pyresample.kd_tree.get_neighbour_info(
            source_def,
            target_def,
            radius_of_influence=50000,
            neighbours=1,
        )

        result = pyresample.kd_tree.get_sample_from_neighbour_info(
            'nn',
            target_def.shape,
            rad,
            neighbor_info[0],
            neighbor_info[1],
            neighbor_info[2],
            distance_array=neighbor_info[3],
            fill_value=np.nan,
        )

        ds = xr.Dataset(
            dict(Rad=(('lat', 'lon'), result)),
            coords=dict(lat=lat_range, lon=lon_range),
        )
        return ds
