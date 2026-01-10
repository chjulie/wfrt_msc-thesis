import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from shapely.geometry import Polygon, MultiPolygon
import kaleido

import sys

sys.path.append("../")

from data_constants import DOMAIN_MINX, DOMAIN_MAXX, DOMAIN_MINY, DOMAIN_MAXY


def plot_catchments(gdf):

    print(" - starting figure ... ")
    fig = go.Figure()

    with tqdm(total=len(gdf), desc="Plotting geometries") as pbar:
        for geom in gdf["geometry"]:

            if isinstance(geom, MultiPolygon):
                polygons = geom.geoms
            else:
                polygons = [geom]

            for poly in polygons:
                xs, ys = poly.exterior.coords.xy

                # add polygon of catchement area
                fig.add_trace(
                    go.Scatter(
                        x=xs.tolist(),
                        y=ys.tolist(),
                        mode="lines",
                        line=dict(width=2),
                        name="polygon boundary",
                        showlegend=False,
                    )
                )
                pbar.update()

    # add climatex domain
    rect_x = [
        DOMAIN_MINX,
        DOMAIN_MAXX,
        DOMAIN_MAXX,
        DOMAIN_MINX,
        DOMAIN_MINX,
    ]  # rectangle x-coords
    rect_y = [
        DOMAIN_MINY,
        DOMAIN_MINY,
        DOMAIN_MAXY,
        DOMAIN_MAXY,
        DOMAIN_MINY,
    ]  # rectangle y-coords

    fig.add_trace(
        go.Scatter(
            x=rect_x,
            y=rect_y,
            mode="lines",
            line=dict(width=2, color="#219ebc"),
            name="rectangle",
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Polygon Boundaries",
        xaxis=dict(scaleanchor="y"),  # preserve aspect ratio
        yaxis=dict(),
        width=40,
        height=50,
    )
    print("- figure created!")
    # fig.show()
    save_path = "../../reports/plots/geometry/catchments_boundaries.png"
    # fig.write_image(save_path)
    kaleido.write_fig_sync(fig, path=save_path)
    print(
        f"- figure saved to {'../../reports/plots/geometry/catchments_boundaries.png'}!"
    )


if __name__ == "__main__":
    """
    Create on dataset containing boundary of catchments area for BC
    """

    shx_folder = "../../data/geometric_data"
    shx_names = [
        "MDA_ADP_07_DrainageBasin_BassinDeDrainage.shx",
        "MDA_ADP_08_DrainageBasin_BassinDeDrainage.shx",
    ]

    gdf = gpd.GeoDataFrame(columns=["geometry"])

    # concatenate files
    for shx_file in shx_names:
        print(f" - reading file {shx_file} ... ")
        shx_data = gpd.read_file(f"{shx_folder}/{shx_file}")
        gdf = pd.concat([gdf, shx_data])

    print(" - read all files! ")

    plot_catchments(gdf)

    # filter catchments in climatex domain
