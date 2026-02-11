import xarray as xr
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.tri as tri
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import warnings


def xtime(a, b):
    return datetime.strftime(
        a + pd.Timedelta(hours=int(b)), format="%Y-%m-%dT%H:00:00.000000000"
    )


def plot_basemap(ax):
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in create_collection",
        category=RuntimeWarning,
    )

    ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#D4DFED", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=2)


def plot_data(ax, ds, field, time, norm, cbar_title):

    data = ds[field].sel(time=time).values

    try:
        geom = gpd.points_from_xy(ds.longitude, ds.latitude, crs=ccrs.PlateCarree())
        gdf = gpd.GeoDataFrame({"data": data}, geometry=geom)

    except ValueError:
        geom = gpd.points_from_xy(
            ds.longitude.values.ravel(),
            ds.latitude.values.ravel(),
            crs=ccrs.PlateCarree(),
        )
        gdf = gpd.GeoDataFrame({"data": data.ravel()}, geometry=geom)

    gdf.plot(
        ax=ax,
        column="data",
        markersize=10,
        cmap="coolwarm",
        legend=True,
        legend_kwds={"label": cbar_title, "shrink": 0.4},
        transform=ccrs.PlateCarree(),
        zorder=1,
        norm=norm,
    )


def get_norm(all_ds, initial_date, field, lead_times):
    vmins = []
    vmaxs = []

    for ds in all_ds:
        for lt in lead_times:
            t = xtime(initial_date, lt)
            data = ds[field].sel(time=t).values
            vmins.append(np.min(data))
            vmaxs.append(np.max(data))

    vmin, vmax = np.nanmin(vmins), np.nanmax(vmaxs)
    norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation againts Climatex")
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="Start date of evaluation period, format: YYYY-mm-dd",
    )
    args = parser.parse_args()

    gt_ds = xr.open_dataset(
        "/scratch/juchar/data/cleaned_data/preproc-climatex-training-6h-20190601-20231231.nc"
    )
    bris_ds = xr.open_dataset(
        f"/scratch/juchar/val/bris_20221228T00.nc", engine="netcdf4"
    )
    climatex_ds = xr.open_dataset(
        f"/scratch/juchar/val/climatex_20221228T00.nc", engine="netcdf4"
    )

    field = args.field
    cbar_title = {
        "2t": "2t [Kelvin]",
        "10u": "10u [m/s]",
        "10v": "10v [m/s]",
        "sp": "sp [HPa]",
    }
    all_ds = [gt_ds, bris_ds, climatex_ds]  # [global_ds, bris_ds, climatex_ds]
    models = ["bris", "climatex"]
    lead_times = [0, 6, 12, 24]
    initial_date = datetime(2022, 12, 28, 0)
    projection = ccrs.Stereographic(
        central_latitude=90.0,  # North Pole
        central_longitude=-90.0,  # STAND_LON
        true_scale_latitude=60.0,  # Usually TRUELAT2 for polar stereo
    )
    norm = get_norm(all_ds, initial_date, field, lead_times)

    fig, ax = plt.subplots(
        len(all_ds),
        len(lead_times),
        figsize=(15, 12),
        subplot_kw={"projection": projection},
    )
    ax = ax.ravel()

    c = 0
    m = 0
    norm = get_norm(all_ds, initial_date, field, lead_times)
    for ds in all_ds:
        for lt in lead_times:
            print(c)
            t = xtime(initial_date, lt)
            colorbar_title = cbar_title.get(field)
            plot_basemap(ax[c])
            plot_data(ax[c], ds, field, t, norm, colorbar_title)

            if c < len(lead_times):
                ax[c].set_title(f"Lead time : {str(lt)} h.")
            if c % len(lead_times) == 0:
                ax[c].set_ylabel(f"{models[m]}")
                m + 1
            c += 1

    plt.savefig(
        f"../reports/plots/prediction/bris-climatex-{field}-20221228.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
