import pickle
import numpy as np
import geopandas as gpd
import warnings

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


def basemap(ax):
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in create_collection",
        category=RuntimeWarning,
    )

    # ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
    # ax.add_feature(cfeature.LAND, facecolor="#d4dfed", zorder=0)
    ax.add_feature(cfeature.COASTLINE, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)


def plot_triangulation(
    ax: plt.Axes,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    data: np.ndarray,
    vmin: float,
    vmax: float,
):
    triangulation = tri.Triangulation(longitudes, latitudes)
    tpc = ax.tripcolor(triangulation, data, vmin=vmin, vmax=vmax)


def plot_geopandas(
    ax: plt.Axes,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    data: np.ndarray,
    vmin: float,
    vmax: float,
    transform_crs: bool = False,
):
    if transform_crs:
        geom = gpd.points_from_xy(longitudes, latitudes, crs=projection)
    else:
        geom = gpd.points_from_xy(longitudes, latitudes)

    gdf = gpd.GeoDataFrame({"data": data}, geometry=geom)

    if transform_crs:
        gdf.plot(
            ax=ax,
            column="data",
            cmap="RdBu",
            markersize=0.5,
            zorder=1,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
    else:
        gdf.plot(
            ax=ax,
            column="data",
            cmap="RdBu",
            markersize=0.5,
            zorder=1,
            vmin=vmin,
            vmax=vmax,
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

    vmin = np.minimum(
        np.min(data["IFS"]["orography"]), np.min(data["climatex"]["orography"])
    )
    vmax = np.maximum(
        np.max(data["IFS"]["orography"]), np.max(data["climatex"]["orography"])
    )

    # -- plot --

    # TODO: mask IFS data using kd neighbors and not classical mask

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    axs = [ax1, ax2]

    for ax in axs:
        ax.set_aspect("auto")
        # ax.set_extent(
        #     [DOMAIN_MINX, DOMAIN_MAXX, DOMAIN_MINY, DOMAIN_MAXY], crs=projection
        # )
        ax.set_xlim(DOMAIN_MINX, DOMAIN_MAXX)
        ax.set_ylim(DOMAIN_MINY, DOMAIN_MAXY)
        basemap(ax)

    print("plotted basemap")

    plot_geopandas(
        ax=axs[0],
        longitudes=data["IFS"]["longitudes"],
        latitudes=data["IFS"]["latitudes"],
        data=data["IFS"]["orography"],
        vmin=vmin,
        vmax=vmax,
    )

    # plot_triangulation(
    #     ax=axs[0],
    #     longitudes=data["IFS"]["longitudes"],
    #     latitudes=data["IFS"]["latitudes"],
    #     data=data["IFS"]["orography"],
    #     vmin=vmin,
    #     vmax=vmax,
    # )
    axs[0].set_title("IFS orography")

    axs[0].set_title("IFS orography")

    print("plotted IFS")

    plot_geopandas(
        ax=axs[1],
        longitudes=data["climatex"]["longitudes"],
        latitudes=data["climatex"]["latitudes"],
        data=data["climatex"]["orography"],
        vmin=vmin,
        vmax=vmax,
        transform_crs=True,
    )

    axs[1].set_title("Climatex orography")

    print("plotted climatex")

    plt.savefig(
        "reports/plots/datasets/orog_regional_global.png", dpi=100, bbox_inches="tight"
    )
    print("saved fig")
