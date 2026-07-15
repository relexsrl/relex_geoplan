import numpy as np
from osgeo import gdal

from .core.models import Geotransform, RasterShape

gdal.UseExceptions()


def read_tiff(path: str) -> tuple[np.ndarray, Geotransform, str]:
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open: {path}")
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    gt = ds.GetGeoTransform()
    geotransform = Geotransform(*gt)
    crs_wkt = ds.GetProjection()
    ds = None

    return array, geotransform, crs_wkt
