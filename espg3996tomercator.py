import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

def extract_bathy_for_map(tiff_path, save_as=None):
    lon_min, lat_min = 9, 78
    lon_max, lat_max = 22, 79

    to_native = Transformer.from_crs("EPSG:4326", "EPSG:3996", always_xy=True)
    x_min, y_min = to_native.transform(lon_min, lat_min)
    x_max, y_max = to_native.transform(lon_max, lat_max)

    with rasterio.open(tiff_path) as src:
        window = from_bounds(x_min, y_min, x_max, y_max, transform=src.transform)
        data = src.read(1, window=window)
        src_transform = src.window_transform(window)

        dst_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, data.shape[1], data.shape[0],
            left=x_min, bottom=y_min, right=x_max, top=y_max
        )

        dst_data = np.empty((height, width), dtype=data.dtype)

        reproject(
            source=data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

    plt.imshow(np.ma.masked_equal(dst_data, 0), extent=(
        transform.c, transform.c + transform.a * width,
        transform.f + transform.e * height, transform.f),
        cmap='bone', origin='upper')
    
    #plt.colorbar(label="Depth (m)")
    plt.tight_layout()

    if save_as:
        with rasterio.open(
            save_as,
            'w',
            driver='GTiff',
            height=dst_data.shape[0],
            width=dst_data.shape[1],
            count=1,
            dtype=dst_data.dtype,
            crs="EPSG:3857",
            transform=transform
        ) as dst:
            dst.write(dst_data, 1)
        print(f"Extracted bathy saved : {save_as}")
