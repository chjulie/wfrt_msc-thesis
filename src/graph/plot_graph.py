import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

projection = ccrs.NearsidePerspective(
    central_longitude=-110, central_latitude=50, satellite_height=3578583
)


def plot_cutout(
    nodes: np.ndarray,
):
    fig, ax = plt.subplots(
        figsize=(20, 10),
        subplot_kw={"projection": projection},
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in create_collection",
        category=RuntimeWarning,
    )
    ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#D4DFED", zorder=0)
    ax.add_feature(cfeature.COASTLINE, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

    ax.scatter(
        x=nodes[:, 1],
        y=nodes[:, 0],
        s=0.1,
        transform=ccrs.PlateCarree(),
        color="#CA1634",
    )

    plt.savefig(
        f"../../reports/plots/graph/cutout.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(f"../../reports/plots/graph/cutout.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":

    with open("../../data/graphs/source_nodes.npy", "rb") as f:
        nodes = np.load(f)

    print(nodes.shape)
    plot_cutout(nodes=nodes)
