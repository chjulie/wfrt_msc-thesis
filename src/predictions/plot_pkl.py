import numpy as np
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
import sys

sys.path.append("../")

from data_constants import (
    DOMAIN_MINX,
    DOMAIN_MAXX,
    DOMAIN_MINY,
    DOMAIN_MAXY,
    PRED_DATA_DIR,
    PRED_PLOT_DIR,
)


def read_pkl(path):

    with open(pred_data_path, "rb") as f:
        domain_state = pickle.load(f)
        date = domain_state["date"]
        latitudes = domain_state["latitudes"]
        longitudes = domain_state["longitudes"]
        fields = domain_state["fields"]

    return date, latitudes, longitudes, fields


if __name__ == "__main__":
    EXPERIENCE = "aifs-single-v1"
    date_str = "20251118"
    domain = "regional"  # 'regional', 'global'

    # Open pkl file
    pred_data_path = f"../../{PRED_DATA_DIR}/{date_str}_{domain}_state.pkl"
    with open(pred_data_path, "rb") as f:
        domain_state = pickle.load(f)

    date, latitudes, longitudes, fields = read_pkl(path=pred_data_path)

    # print("\nState: ")
    # for k,v in domain_state.items():
    #     print(f" - {k}: {type(v)}")

    # print("\nDate: ")
    # print(f" date: {date}")

    print("\nLatitudes: ")
    print(f" lat: {latitudes.shape}")

    print("\nLongitudes: ")
    print(f" lon: {longitudes.shape}")

    # print("\nFields: ")
    # for k,v in fields.items():
    #     print(f" - {k}: {v.shape}")

    # Plot domain data
    DISP_VAR = "2t"
    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    print(" ** lons: ", longitudes.shape)
    print(" ** lats: ", latitudes.shape)

    triangulation = tri.Triangulation(longitudes, latitudes)

    contour = ax.tricontourf(
        triangulation, fields[DISP_VAR], levels=20, cmap="RdBu"
    )  # transform=ccrs.PlateCarree()
    cbar = fig.colorbar(
        contour, ax=ax, orientation="vertical", shrink=0.7, label=f"{DISP_VAR}"
    )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    # remove labels on top and right
    # gl.top_labels = False
    # gl.right_labels = False

    fig.suptitle(f"{DISP_VAR} at {date.strftime(format='%Y%m%d_%H:%M:%S')}")
    plt.savefig(
        f"../../{PRED_PLOT_DIR}/{date.strftime(format='%Y%m%d')}_{EXPERIENCE}_{domain}_{DISP_VAR}"
    )

    ###
    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    triangulation = tri.Triangulation(longitudes, latitudes)

    contour = ax.tricontourf(
        triangulation, fields[DISP_VAR], levels=20, cmap="RdBu"
    )  # transform=ccrs.PlateCarree()
    cbar = fig.colorbar(
        contour, ax=ax, orientation="vertical", shrink=0.7, label=f"{DISP_VAR}"
    )
    ax.triplot(
        triangulation,
        color="black",
        linewidth=0.3,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    # remove labels on top and right
    # gl.top_labels = False
    # gl.right_labels = False

    fig.suptitle(
        f"{DISP_VAR} at {date.strftime(format='%Y%m%d_%H:%M:%S')} and triangulation"
    )
    plt.savefig(
        f"../../{PRED_PLOT_DIR}/{date.strftime(format='%Y%m%d')}_{EXPERIENCE}_{domain}_triangulation_{DISP_VAR}",
        dpi=750,
    )

    ###
    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    triangulation = tri.Triangulation(longitudes, latitudes)

    contour = ax.tricontourf(
        triangulation, fields[DISP_VAR], levels=20, cmap="RdBu"
    )  # transform=ccrs.PlateCarree()
    cbar = fig.colorbar(
        contour, ax=ax, orientation="vertical", shrink=0.7, label=f"{DISP_VAR}"
    )
    ax.scatter(
        longitudes,
        latitudes,
        s=0.8,
        color="black",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    # remove labels on top and right
    # gl.top_labels = False
    # gl.right_labels = False

    fig.suptitle(
        f"{DISP_VAR} at {date.strftime(format='%Y%m%d_%H:%M:%S')} and data points"
    )
    plt.savefig(
        f"../../{PRED_PLOT_DIR}/{date.strftime(format='%Y%m%d')}_{EXPERIENCE}_{domain}_points_{DISP_VAR}",
        dpi=750,
    )

    print(" > Program finished successfully !")
