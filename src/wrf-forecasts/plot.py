import xarray as xr
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

projection = ccrs.Stereographic(
    central_latitude=90.0,  # North Pole
    central_longitude=-90.0,  # STAND_LON
    true_scale_latitude=60.0,  # Usually TRUELAT2 for polar stereo
)
WRF_SFC_FIELDS = ["2t", "10u", "10v", "10ff", "msl", "tp"]
DPI = 200


def plot_triangulation(
    coords: np.array,
    data: np.array | xr.DataArray,
    colorbar_label: str,
    title: str,
):
    with warnings.catch_warnings():
        # Ignore the nan warnings
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in create_collection",
            category=RuntimeWarning,
        )

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": projection})

        # Add base map features
        ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

        triangulation = tri.Triangulation(coords[:, 0], coords[:, 1])
        contour = ax.tricontourf(
            triangulation, data, levels=20, cmap="RdBu"
        )  # , transform=ccrs.PlateCarree()
        fig.colorbar(contour, label=colorbar_label)
        ax.set_title(title)
        field_value = title.split("field: '")[1].split("',")[0]
        plt.savefig(
            f"reports/plots/wrf/tricontourf_{field_value}.png",
            bbox_inches="tight",
            dpi=DPI,
        )


def plot_scatter(
    coords: np.array,
    data: np.array | xr.DataArray,
    colorbar_label: str,
    title: str,
    save_path: str,
):
    with warnings.catch_warnings():
        # Ignore the nan warnings
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in create_collection",
            category=RuntimeWarning,
        )

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": projection})

        # Add base map features
        ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

        sc = ax.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=data,
            cmap="RdBu",
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(sc, label=colorbar_label)
        ax.set_title(title)
        field_value = title.split("field: '")[1].split("',")[0]
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=DPI,
        )


def plot_geopandas(
    coords: np.array,
    data: np.array | xr.DataArray,
    colorbar_label: str,
    title: str,
    save_path: str,
):
    with warnings.catch_warnings():
        # Ignore the nan warnings
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in create_collection",
            category=RuntimeWarning,
        )

        # Create GeoDataFrame with point geometries
        geom = gpd.points_from_xy(coords[:, 0], coords[:, 1], crs=projection)
        gdf = gpd.GeoDataFrame({"data": data}, geometry=geom)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": projection})

        # Add base map features
        ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

        # Scatter plot the data
        gdf.plot(
            ax=ax,
            column="data",
            cmap="RdBu",
            markersize=0.5,
            legend=True,
            legend_kwds={"label": colorbar_label, "shrink": 0.6},
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.set_title(title)
        field_value = title.split("field: '")[1].split("',")[0]
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=DPI,
        )


if __name__ == "__main__":
    file_name = "wrfout_d02_processed_23071000.nc"
    wrf_ds = xr.open_dataset(f"data/wrf_data/{file_name}")
    lead_time = 12
    date_short_str = file_name[-11:-3]
    date = datetime.strptime(date_short_str, "%y%m%d%H")
    coords = np.column_stack(
        (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
    )
    for field in WRF_SFC_FIELDS:
        print(field)
        data = (
            wrf_ds[field]
            .sel(
                XTIME=f"{(date + timedelta(hours=lead_time)).strftime('%Y-%m-%dT%H:%M:%S')}.000000000"
            )
            .values.flatten()
        )

        # plot
        title = f"field: '{field}', initial datetime: '{date.strftime('%Y-%m-%dT%H:%M:%S')}',\n lead time: '{lead_time}h'"
        try:
            colorbar_label = wrf_ds[field].attrs["units"]
        except KeyError:
            colorbar_label = "Pa"

        # plot_triangulation(
        #     coords=coords, data=data, colorbar_label=colorbar_label, title=title
        # )
        # plot_scatter(coords=coords, data=data, colorbar_label=colorbar_label, title=title)
        plot_geopandas(
            coords=coords,
            data=data,
            colorbar_label=colorbar_label,
            title=title,
            save_path=f"reports/plots/wrf/gpd_{field}_{date_short_str}_{str(lead_time)}.png",
        )
