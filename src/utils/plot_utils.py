import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_interpolation(
    src_coords: np.array,
    src_data: np.array,
    tgt_coords: np.array,
    tgt_pred: np.array,
    specs: dict,
):

    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.scatter(tgt_coords[:, 0], tgt_coords[:, 1], c=tgt_pred, cmap="RdBu")

    triangulation = tri.Triangulation(src_coords[:, 0], src_coords[:, 1])
    contour = ax.tricontourf(
        triangulation, src_data, levels=20, cmap="RdBu", alpha=0.5
    )  # transform=ccrs.PlateCarree()
    ax.set_title(f"field: {specs["field"]}, resampling method: {specs["resampling"]}")

    plt.show()