"""
===============
spatial_utils.py
===============

Utility functions for geospatial calculations and polygon operations.
"""

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from podaac.subsetter.utils import coordinate_utils


def create_geospatial_bounding_box(spatial_bounds_array, east, west):
    """
    Generate a Well-Known Text (WKT) POLYGON string representing the geospatial bounds.
    """
    lon_min, lon_max = spatial_bounds_array[0]
    lat_min, lat_max = spatial_bounds_array[1]

    if east:
        lon_min = east
    if west:
        lon_max = west

    wkt_polygon = (
        f"POLYGON (({lon_min:.5f} {lat_min:.5f}, {lon_max:.5f} {lat_min:.5f}, "
        f"{lon_max:.5f} {lat_max:.5f}, {lon_min:.5f} {lat_max:.5f}, {lon_min:.5f} {lat_min:.5f}))"
    )
    return wkt_polygon


def create_geospatial_bounds(dataset, lon_var_names, lat_var_names):
    """Create geospatial bounds from 4 corners of 2d array"""
    for lon_var_name, lat_var_name in zip(lon_var_names, lat_var_names):
        lon = dataset[lon_var_name]
        lat = dataset[lat_var_name]
        lon_fill_value = lon.attrs.get('_FillValue', None)
        lat_fill_value = lat.attrs.get('_FillValue', None)
        break
    lon_scale = lon.attrs.get('scale_factor', 1.0)
    lon_offset = lon.attrs.get('add_offset', 0.0)
    lat_scale = lat.attrs.get('scale_factor', 1.0)
    lat_offset = lat.attrs.get('add_offset', 0.0)
    if lon.ndim != 2 or lat.ndim != 2:
        return None
    nrows, ncols = lon.shape
    points = [
        (float(coordinate_utils.remove_scale_offset(lon[0, 0], lon_scale, lon_offset)), float(coordinate_utils.remove_scale_offset(lat[0, 0], lat_scale, lat_offset))),
        (float(coordinate_utils.remove_scale_offset(lon[nrows - 1, 0], lon_scale, lon_offset)), float(coordinate_utils.remove_scale_offset(lat[nrows - 1, 0], lat_scale, lat_offset))),
        (float(coordinate_utils.remove_scale_offset(lon[nrows - 1, ncols - 1], lon_scale, lon_offset)), float(coordinate_utils.remove_scale_offset(lat[nrows - 1, ncols - 1], lat_scale, lat_offset))),
        (float(coordinate_utils.remove_scale_offset(lon[0, ncols - 1], lon_scale, lon_offset)), float(coordinate_utils.remove_scale_offset(lat[0, ncols - 1], lat_scale, lat_offset)))
    ]
    if any(np.isnan(point[0]) or np.isnan(point[1]) or point[0] == lon_fill_value or point[1] == lat_fill_value for point in points):
        return None
    sorted_points = _ensure_counter_clockwise(points)
    wkt_polygon = (
        f"POLYGON(({sorted_points[0][0]:.5f} {sorted_points[0][1]:.5f}, "
        f"{sorted_points[1][0]:.5f} {sorted_points[1][1]:.5f}, "
        f"{sorted_points[2][0]:.5f} {sorted_points[2][1]:.5f}, "
        f"{sorted_points[3][0]:.5f} {sorted_points[3][1]:.5f}, "
        f"{sorted_points[0][0]:.5f} {sorted_points[0][1]:.5f}))"
    )
    return wkt_polygon


def _shoelace_area(points):
    """Computes the signed area of a polygon."""
    x, y = np.array(points)[:, 0], np.array(points)[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _ensure_counter_clockwise(points):
    """Ensures the points are ordered counterclockwise."""
    area = _shoelace_area(points)
    if area > 0:
        return points[::-1]
    return points


def get_east_west_lon(dataset, lon_var_name):
    """
    Determines the easternmost and westernmost longitudes from a dataset,
    correctly handling cases where the data crosses the antimeridian.
    """
    lon_2d = dataset[lon_var_name]
    if lon_2d is None:
        return None, None
    fill_value = lon_2d.attrs.get('_FillValue', None)
    lon_flat = lon_2d.values.flatten()
    if fill_value is not None:
        lon_flat = lon_flat[lon_flat != fill_value]
    lon_flat = lon_flat[~np.isnan(lon_flat)]
    if lon_flat.size == 0:
        return None, None
    crosses_antimeridian = np.any((lon_flat[:-1] > 150) & (lon_flat[1:] < -150))
    lon_360 = np.where(lon_flat < 0, lon_flat + 360, lon_flat)
    lon_sorted = np.sort(lon_360)
    gaps = np.diff(lon_sorted)
    wrap_gap = lon_sorted[0] + 360 - lon_sorted[-1]
    gaps = np.append(gaps, wrap_gap)
    max_gap_index = np.argmax(gaps)
    if crosses_antimeridian:
        eastmost_360 = lon_sorted[max_gap_index]
        westmost_360 = lon_sorted[(max_gap_index + 1) % len(lon_sorted)]
    else:
        eastmost_360 = np.max(lon_flat)
        westmost_360 = np.min(lon_flat)

    def convert_to_standard(lon):
        return lon - 360 if lon > 180 else lon

    eastmost = round(convert_to_standard(eastmost_360), 5)
    westmost = round(convert_to_standard(westmost_360), 5)
    return eastmost, westmost


def translate_longitude(geometry):
    """
    Translates the longitude values of a Shapely geometry from the range [-180, 180) to [0, 360).

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        The input shape geometry to be translated

    Returns
    -------
    geometry
        The translated shape geometry
    """

    def translate_point(point):
        # Translate the point's x-coordinate (longitude) by adding 360
        return Point((point.x + 360) % 360, point.y)

    def translate_polygon(polygon):
        def translate_coordinates(coords):
            if len(coords[0]) == 2:
                return [((x + 360) % 360, y) for x, y in coords]
            if len(coords[0]) == 3:
                return [((x + 360) % 360, y, z) for x, y, z in coords]
            return coords

        exterior = translate_coordinates(polygon.exterior.coords)

        interiors = [
            translate_coordinates(ring.coords)
            for ring in polygon.interiors
        ]

        return Polygon(exterior, interiors)

    if isinstance(geometry, (Point, Polygon)):  # pylint: disable=no-else-return
        return translate_point(geometry) if isinstance(geometry, Point) else translate_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        # Translate each polygon in the MultiPolygon
        translated_polygons = [translate_longitude(subgeometry) for subgeometry in geometry.geoms]
        return MultiPolygon(translated_polygons)
    else:
        # Handle other geometry types as needed
        return geometry
