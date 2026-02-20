import pickle
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

projection = ccrs.Stereographic(
    central_latitude=90.0,  # North Pole
    central_longitude=-90.0,  # STAND_LON
    true_scale_latitude=60.0,  # Usually TRUELAT2 for polar stereo
)

# DOMAIN BOUNDS (Climatex bounds)
DOMAIN_MINX = -146.74888611
DOMAIN_MINY = 43.16402817
DOMAIN_MAXX = -108.88935089
DOMAIN_MAXY = 66.57196045
CMAP = "GnBu_r"


def clip_global(global_coords, climatex_coords, tolerance=0.03):
    tree = cKDTree(climatex_coords)
    indices = tree.query_ball_point(global_coords, r=tolerance)
    mask = np.array([len(idx) > 0 for idx in indices])

    return mask


def basemap(ax):
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in create_collection",
        category=RuntimeWarning,
    )

    # ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
    # ax.add_feature(cfeature.LAND, facecolor="#d4dfed", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)


def plot_triangulation(ax, longitudes, latitudes, data, mask, norm):

    triangulation = tri.Triangulation(longitudes, latitudes)
    triangle_mask = ~np.all(mask[triangulation.triangles], axis=1)
    triangulation.set_mask(triangle_mask)

    data = np.ma.masked_where(~mask, data)
    # print(np.count_nonzero(~np.isnan(data)))

    contour = ax.tricontourf(
        triangulation,
        data,
        levels=20,
        cmap=CMAP,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )


def plot_data(ax, longitudes, latitudes, data, norm):

    geom = gpd.points_from_xy(longitudes, latitudes, crs=ccrs.PlateCarree())
    gdf = gpd.GeoDataFrame({"data": data}, geometry=geom)

    gdf.plot(
        ax=ax,
        column="data",
        markersize=10,
        cmap=CMAP,
        legend=True,
        legend_kwds={"label": "Terrain height [m]", "shrink": 0.4},
        transform=ccrs.PlateCarree(),
        zorder=1,
        norm=norm,
    )


if __name__ == "__main__":
    """
    Plot IFS vs Climatex topography for the regional domain (similar to Tim's)
    Steps :
        1. Read climatex for one date, select topography
        2. IFS
            2.1 Read for one date, select topography
            2.2 Cut to regional domain or use ax.set_extent

    """

    with open("data/plots_data/orog_regional_global.pkl", "rb") as f:
        data = pickle.load(f)

    print(data.keys())

    global_coords = np.column_stack(
        (data["IFS"]["longitudes"], data["IFS"]["latitudes"])
    )
    climatex_coords = np.column_stack(
        (data["climatex"]["longitudes"], data["climatex"]["latitudes"])
    )
    mask = clip_global(global_coords=global_coords, climatex_coords=climatex_coords)

    # -- norm --
    vmin = np.minimum(
        np.min(np.ma.masked_where(~mask, data["IFS"]["orography"])),
        np.min(data["climatex"]["orography"]),
    )
    vmax = np.maximum(
        np.max(np.ma.masked_where(~mask, data["IFS"]["orography"])),
        np.max(data["climatex"]["orography"]),
    )

    norm = Normalize(vmin=vmin, vmax=vmax)
    # -- plot 2 figures --

    # CLIMATEX

    fig, ax = plt.subplots(figsize=(6.5, 8), subplot_kw={"projection": projection})
    basemap(ax)
    plot_data(
        ax=ax,
        longitudes=data["climatex"]["longitudes"],
        latitudes=data["climatex"]["latitudes"],
        data=data["climatex"]["orography"],
        norm=norm,
    )
    valid_lon = data["climatex"]["longitudes"]
    valid_lat = data["climatex"]["latitudes"]
    ax.set_extent(
        [
            valid_lon.min() + 16.8,
            valid_lon.max() - 13.5,
            valid_lat.min() + 2.7,
            valid_lat.max() - 1.9,
        ],
        crs=ccrs.PlateCarree(),
    )  # works !!
    plt.savefig(
        "reports/plots/datasets/climatex_orography.png", dpi=400, bbox_inches="tight"
    )
    plt.savefig(
        "reports/plots/datasets/climatex_orography.svg",
        format="svg",
        bbox_inches="tight",
    )

    # IFS (using triangulation)

    fig, ax = plt.subplots(figsize=(6, 8), subplot_kw={"projection": projection})

    valid_lon = data["IFS"]["longitudes"][mask]
    valid_lat = data["IFS"]["latitudes"][mask]

    basemap(ax)
    plot_triangulation(
        ax,
        data["IFS"]["longitudes"],
        data["IFS"]["latitudes"],
        data["IFS"]["orography"],
        mask,
        norm,
    )

    ax.set_extent(
        [
            valid_lon.min() + 16.9,
            valid_lon.max() - 13.7,
            valid_lat.min() + 2.6,
            valid_lat.max() - 1.7,
        ],
        crs=ccrs.PlateCarree(),
    )  # works !!
    plt.savefig(
        "reports/plots/datasets/IFS_orography.png", dpi=400, bbox_inches="tight"
    )
    plt.savefig(
        "reports/plots/datasets/IFS_orography.svg", format="svg", bbox_inches="tight"
    )
