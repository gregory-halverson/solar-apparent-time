
from typing import Union
from datetime import datetime, timedelta

from geopandas import GeoSeries
import numpy as np
import pandas as pd
import rasters as rt
from rasters import SpatialGeometry


def _parse_time(time_UTC: Union[datetime, str, list, np.ndarray]) -> np.ndarray:
    """
    Convert a time or list/array of times to a numpy array of datetime64 objects.
    Accepts a single datetime, string, or a list/array of either.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        The UTC time(s) as datetime object(s), string(s), or array-like.

    Returns
    -------
    np.ndarray
        Array of datetime64 objects.
    """
    if isinstance(time_UTC, (str, datetime)):
        return np.array([pd.to_datetime(time_UTC)])
    
    # If already array-like
    arr = np.array(time_UTC)
    
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr)
    
    return pd.to_datetime(arr)

def _broadcast_time_and_space(times: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Broadcast time and longitude arrays to compatible shapes for element-wise operations.

    Parameters
    ----------
    times : np.ndarray
        Array of times (datetime64).
    lons : np.ndarray
        Array of longitudes (degrees).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Broadcasted arrays of times and longitudes.
    """
    times = np.asarray(times)
    lons = np.asarray(lons)
    
    if times.shape == ():
        times = times[None]

    if lons.shape == ():
        lons = lons[None]

    return np.broadcast_arrays(times[..., None], lons)

def calculate_solar_hour_of_day(
    time_UTC: Union[datetime, str, list, np.ndarray],
    geometry: SpatialGeometry = None,
    lat: Union[np.ndarray, float] = None,
    lon: Union[np.ndarray, float] = None
) -> np.ndarray:
    """
    Calculate the solar hour of day for given UTC time(s) and spatial information.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        UTC time(s) as datetime object(s), string(s), or array-like.
    geometry : SpatialGeometry, optional
        SpatialGeometry or RasterGeometry object with longitude attribute.
    lat : float or np.ndarray, optional
        Latitude(s) in degrees (not used, included for API compatibility).
    lon : float or np.ndarray, optional
        Longitude(s) in degrees. Required if geometry is not provided.

    Returns
    -------
    np.ndarray
        Array of solar hour of day values, same shape as broadcasted input.

    Notes
    -----
    The solar hour of day is the local solar time in hours, accounting for longitude offset.
    """
    times = _parse_time(time_UTC)

    if geometry is not None:
        lon = geometry.lon
    elif lon is not None:
        lon = np.asarray(lon)
    else:
        raise ValueError('Must provide either spatial or lon.')
    
    # Broadcast times and lons
    times_b, lons_b = _broadcast_time_and_space(times, lon)

    # Calculate hour_UTC
    hour_UTC = (
        times_b.astype('datetime64[h]').astype(int) % 24
        + (times_b.astype('datetime64[m]').astype(int) % 60) / 60
        + (times_b.astype('datetime64[s]').astype(int) % 60) / 3600
    )

    offset = np.radians(lons_b) / np.pi * 12
    hour_of_day = hour_UTC + offset
    hour_of_day = np.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = np.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day

def calculate_solar_day_of_year(
    time_UTC: Union[datetime, str, list, np.ndarray],
    geometry: SpatialGeometry = None,
    lat: Union[np.ndarray, float] = None,
    lon: Union[np.ndarray, float] = None
) -> np.ndarray:
    """
    Calculate the solar day of year for given UTC time(s) and spatial information.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        UTC time(s) as datetime object(s), string(s), or array-like.
    geometry : SpatialGeometry, optional
        SpatialGeometry or RasterGeometry object with longitude attribute.
    lat : float or np.ndarray, optional
        Latitude(s) in degrees (not used, included for API compatibility).
    lon : float or np.ndarray, optional
        Longitude(s) in degrees. Required if geometry is not provided.

    Returns
    -------
    np.ndarray
        Array of solar day of year values, same shape as broadcasted input.

    Notes
    -----
    The solar day of year is the day of year at the local solar time, accounting for longitude offset.
    """
    times = _parse_time(time_UTC)

    # If latitude is not provided, try to extract from geometry
    if lat is None and isinstance(geometry, SpatialGeometry):
        lat = geometry.lat
    elif lat is None and isinstance(geometry, GeoSeries):
        lat = geometry.y
    elif lat is None:
        raise ValueError("no latitude provided")

    if lon is None and isinstance(geometry, SpatialGeometry):
        lon = geometry.lon
    elif lon is None and isinstance(geometry, GeoSeries):
        lon = geometry.x
    elif lon is None:
        raise ValueError("no longitude provided")

    # Handle 1D time and lon inputs of the same length: pair element-wise
    times = np.asarray(times)
    lon = np.asarray(lon)
    if times.ndim == 1 and lon.ndim == 1 and times.shape == lon.shape:
        times_b = times
        lons_b = lon
    else:
        # Broadcast to 2D if not matching 1D
        times_b, lons_b = _broadcast_time_and_space(times, lon)

    # Vectorized conversion to pandas datetime and dayofyear extraction
    times_b_flat = times_b.flatten()
    times_b_dt = pd.to_datetime(times_b_flat)
    doy_UTC = times_b_dt.dayofyear.values.reshape(times_b.shape)

    hour_UTC = (
        times_b.astype('datetime64[h]').astype(int) % 24
        + (times_b.astype('datetime64[m]').astype(int) % 60) / 60
        + (times_b.astype('datetime64[s]').astype(int) % 60) / 3600
    )

    offset = np.radians(lons_b) / np.pi * 12
    hour_of_day = hour_UTC + offset
    day_of_year = doy_UTC.copy()
    day_of_year = np.where(hour_of_day < 0, day_of_year - 1, day_of_year)
    day_of_year = np.where(hour_of_day > 24, day_of_year + 1, day_of_year)

    return day_of_year

def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    """
    Convert Coordinated Universal Time (UTC) to solar apparent time at a given longitude.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    lon : float
        The longitude in degrees.

    Returns
    -------
    datetime
        The solar time at the given longitude.
    """
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

def solar_to_UTC(time_solar: datetime, lon: float) -> datetime:
    """
    Convert solar apparent time to Coordinated Universal Time (UTC) at a given longitude.

    Parameters
    ----------
    time_solar : datetime
        The solar time.
    lon : float
        The longitude in degrees.

    Returns
    -------
    datetime
        The UTC time at the given longitude.
    """
    return time_solar - timedelta(hours=(np.radians(lon) / np.pi * 12))

def UTC_offset_hours_for_longitude(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the offset in hours from UTC based on longitude.

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude(s) in degrees.

    Returns
    -------
    float or np.ndarray
        The calculated offset in hours from UTC.
    """
    # Convert longitude to radians and calculate the offset in hours from UTC
    return np.radians(lon) / np.pi * 12

def UTC_offset_hours_for_area(geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the UTC offset in hours for a given raster geometry.

    Parameters
    ----------
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The UTC offset in hours as a raster.
    """
    return rt.Raster(np.radians(geometry.lon) / np.pi * 12, geometry=geometry)

def solar_day_of_year_for_area(time_UTC: datetime, geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the solar day of year for a given UTC time and raster geometry.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The day of the year as a raster.
    """
    doy_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = UTC_offset_hours_for_area(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    doy = doy_UTC
    doy = rt.where(hour_of_day < 0, doy - 1, doy)
    doy = rt.where(hour_of_day > 24, doy + 1, doy)

    return doy

def solar_day_of_year_for_longitude(
    time_UTC: datetime, 
    lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the solar day of year for a given UTC time and longitude(s).

    Parameters
    ----------
    time_UTC : datetime
        The UTC time to calculate the day of year for.
    lon : float or np.ndarray
        Longitude(s) in degrees.

    Returns
    -------
    float or np.ndarray
        The calculated day of year.
    """
    # Calculate the day of year at the given longitude
    DOY_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    offset = UTC_offset_hours_for_longitude(lon)
    hour_of_day = hour_UTC + offset
    DOY = DOY_UTC
    # Adjust the day of year if the hour of day is outside the range [0, 24]
    DOY = np.where(hour_of_day < 0, DOY - 1, DOY)
    DOY = np.where(hour_of_day > 24, DOY + 1, DOY)

    return DOY

def solar_hour_of_day_for_area(time_UTC: datetime, geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the solar hour of day for a given UTC time and raster geometry.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The hour of the day as a raster.
    """
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = UTC_offset_hours_for_area(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    hour_of_day = rt.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = rt.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day
