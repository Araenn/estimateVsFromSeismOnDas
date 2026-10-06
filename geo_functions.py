"""Cable coordinates required for sampling bathymetry at DAS channels."""

import csv
import numpy as np


def read_coordinates(filepath):
    """Read latitude and longitude from a semicolon-separated CSV with a header."""
    latitudes, longitudes = [], []
    with open(filepath, newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream, delimiter=";")
        next(rows, None)
        for row in rows:
            if not row:
                continue
            latitudes.append(float(row[0]))
            longitudes.append(float(row[1]))
    if len(latitudes) < 2:
        raise ValueError("The cable CSV must contain at least two coordinate pairs.")
    return latitudes, longitudes


def distance_from_latlon_to_meters(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance using the original spherical Earth model."""
    earth_radius_m = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    a = np.clip(a, 0, 1)
    return earth_radius_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def interpolate_cable_coordinates(filepath, distances_m, allow_outside=False):
    """Sample mapped cable coordinates at section distances in metres.

    With allow_outside=True, return NaN beyond the mapped cable endpoint.
    """
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 1 or not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Cable sampling distances must be finite, nonnegative and one-dimensional.")
    latitudes, longitudes = read_coordinates(filepath)
    lengths = [distance_from_latlon_to_meters(latitudes[i], longitudes[i],
                                              latitudes[i + 1], longitudes[i + 1])
               for i in range(len(latitudes) - 1)]
    positions = np.r_[0.0, np.cumsum(lengths)]
    unique = np.r_[True, np.diff(positions) > 0]
    outside = distances > positions[-1]
    if np.any(outside) and not allow_outside:
        raise ValueError("The cable geometry is shorter than the distances requested by the DAS metadata.")
    # No extrapolation beyond the mapped cable: keep unavailable bathymetry missing.
    latitude = np.interp(distances, positions[unique], np.asarray(latitudes)[unique])
    longitude = np.interp(distances, positions[unique], np.asarray(longitudes)[unique])
    latitude[outside] = np.nan
    longitude[outside] = np.nan
    return latitude, longitude


def create_cable_map(filepath, channel_numbers, interval, start_offset_m=0):
    """Interpolate coordinates at evenly spaced distances along the cable.

    Return original latitude/longitude lists and channel latitude/longitude
    arrays, matching the original function's four-value return signature.
    """
    if channel_numbers < 1 or interval <= 0 or start_offset_m < 0:
        raise ValueError("Channel count and spacing must be positive; offset must be nonnegative.")
    latitudes, longitudes = read_coordinates(filepath)
    new_latitudes, new_longitudes = [], []
    distance_accumulated = 0.0
    next_target = start_offset_m
    for i in range(len(latitudes) - 1):
        lat1, lat2 = latitudes[i], latitudes[i + 1]
        lon1, lon2 = longitudes[i], longitudes[i + 1]
        segment_length = distance_from_latlon_to_meters(lat1, lon1, lat2, lon2)
        if segment_length == 0:
            continue
        while (distance_accumulated + segment_length >= next_target
               and len(new_latitudes) < channel_numbers):
            ratio = (next_target - distance_accumulated) / segment_length
            new_latitudes.append(lat1 + ratio * (lat2 - lat1))
            new_longitudes.append(lon1 + ratio * (lon2 - lon1))
            next_target += interval
        distance_accumulated += segment_length
    if len(new_latitudes) != channel_numbers:
        raise ValueError(f"The cable geometry covers only {len(new_latitudes)} of "
                         f"{channel_numbers} requested channels.")
    return latitudes, longitudes, np.asarray(new_latitudes), np.asarray(new_longitudes)
