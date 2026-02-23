import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import warnings
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    return ccrs, cfeature, mo, pd, plt, rc, warnings


@app.cell
def _(plt, rc):
    rc("font", **{"family": "serif", "serif": ["Times New Roman"], "size": "14"})
    rc("text", usetex=False)
    rc("lines", linewidth=2)
    plt.rcParams["axes.facecolor"] = "w"
    plt.rcParams['axes.grid'] = True 
    plt.rcParams["grid.linewidth"] = 0.2 

    temp_cmap = "Reds"
    precip_cmap = "Blues"
    wind_cmap = "Wistia"
    return precip_cmap, temp_cmap, wind_cmap


@app.cell
def _(pd):
    observations_df = pd.read_csv('data/processed_eccc_bch_obs.csv', converters={
                'UTC_DATE': lambda x : x[:-6]
            }).drop(columns=["Unnamed: 0"])
    observations_df
    return (observations_df,)


@app.cell
def _(observations_df):
    stations_statistics = observations_df.groupby(by=['STN_ID','x','y'])[['TEMP', 'WIND_SPEED', 'PRECIP_AMOUNT']].agg(['size'])
    stations_statistics = stations_statistics.reset_index(level=['x','y'])
    stations_statistics.columns = ['_'.join(col).strip('_') for col in stations_statistics.columns]
    stations_statistics
    return (stations_statistics,)


@app.cell
def _(ccrs):
    projection = ccrs.Stereographic(
        central_latitude=90.0,          # North Pole
        central_longitude=-90.0,        # STAND_LON
        true_scale_latitude=60.0        # Usually TRUELAT2 for polar stereo
    )
    scatter_size=24
    edge_lw = 0.2
    return edge_lw, projection, scatter_size


@app.cell
def _(
    ccrs,
    cfeature,
    edge_lw,
    plt,
    precip_cmap,
    projection,
    scatter_size,
    stations_statistics,
    temp_cmap,
    warnings,
    wind_cmap,
):
    fig, axs = plt.subplots(1,3,figsize=(20,10), subplot_kw={'projection': projection})
    axs = axs.ravel()

    with warnings.catch_warnings():
            # Ignore the nan warnings
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in create_collection",
                category=RuntimeWarning,
            )
            for ax in axs:
                # Add base map features
                ax.add_feature(cfeature.OCEAN, facecolor="#D4DFED", zorder=0)
                ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
                ax.add_feature(cfeature.COASTLINE, zorder=2)
                ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

    sc0 = axs[0].scatter(x=stations_statistics['x'], y=stations_statistics['y'], s=scatter_size, c=stations_statistics['TEMP_size'], cmap=temp_cmap, edgecolors='black', linewidth=edge_lw, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc0, ax=axs[0], orientation="vertical", shrink=0.4, label="Number of observations")
    axs[0].set_title('Temperature observations')

    sc1 = axs[1].scatter(x=stations_statistics['x'], y=stations_statistics['y'], s=scatter_size, c=stations_statistics['PRECIP_AMOUNT_size'], edgecolors='black', linewidth=edge_lw, cmap=precip_cmap, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc1, ax=axs[1], orientation="vertical", shrink=0.4, label="Number of observations")
    axs[1].set_title('Precipitations observations')

    sc2 = axs[2].scatter(x=stations_statistics['x'], y=stations_statistics['y'], s=scatter_size, c=stations_statistics['WIND_SPEED_size'], cmap=wind_cmap, edgecolors='black', linewidth=edge_lw, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc2, ax=axs[2], orientation="vertical", shrink=0.4, label="Number of observations")
    axs[2].set_title('Wind observations')

    plt.savefig('reports/plots/observations/stations_statistics.svg', dpi=600, format='svg', bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Compute total number of observations
    """)
    return


@app.cell
def _(observations_df, pd):
    eccc_df = observations_df[pd.to_numeric(observations_df["STN_ID"], errors="coerce").notna()]
    print(' - unique stn : ', eccc_df['STN_ID'].nunique())
    eccc_df.describe()
    return


@app.cell
def _(observations_df, pd):
    bch_obs = observations_df[~pd.to_numeric(observations_df["STN_ID"], errors="coerce").notna()]
    print(' - unique stn : ', bch_obs['STN_ID'].nunique())
    bch_obs.describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
