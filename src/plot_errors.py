import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import warnings

    from utils.data_constants import ERROR_DATA_DIR
    return ERROR_DATA_DIR, ccrs, cfeature, mo, np, pd, plt, rc, warnings


@app.cell
def _(plt, rc):
    rc("font", **{"family": "serif", "serif": ["Times New Roman"], "size": "14"})
    rc("text", usetex=True)
    rc("lines", linewidth=2)
    plt.rcParams["axes.facecolor"] = "w"
    plt.rcParams['axes.grid'] = True 
    plt.rcParams["grid.linewidth"] = 0.2 

    temp_color = "#CA1634"
    precip_color = "#0B84AD"
    wind_color = "#FFC247"#"#CABC53"

    color_dict = {
        '2t': "#CA1634",
        'tp': "#0B84AD",
        '10ff': "#FFC247"
    }
    return color_dict, precip_color, temp_color, wind_color


@app.cell
def _(ERROR_DATA_DIR, pd):
    file_name = 'errors-nwp_reg-20230101_20231231.csv'
    error_df = pd.read_csv(f"../{ERROR_DATA_DIR}/{file_name}").drop(columns=['Unnamed: 0'])
    return (error_df,)


@app.cell
def _(error_df):
    error_df
    return


@app.cell
def _(error_df):
    lead_times = error_df.lead_time.unique()
    lead_times
    return (lead_times,)


@app.cell
def _(error_df):
    temp = error_df[error_df.field == '2t'].groupby('lead_time')['rmse'].agg('mean')
    temp
    return


@app.cell
def _(error_df):
    precip_df = error_df[(error_df.field == 'tp') & (error_df.lead_time == 12)]
    precip_df
    precip_df.isnull().sum()
    #precip = error_df[error_df.field == 'tp'].groupby('lead_time')['rmse'].agg('mean')
    #precip
    return


@app.cell
def _(error_df):
    wind = error_df[error_df.field == '10ff'].groupby('lead_time')['rmse'].agg('mean')
    wind
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot RMSE vs lead time
    ## Compare fields, model set
    """)
    return


@app.cell
def _(error_df, lead_times, plt, precip_color, temp_color, wind_color):
    model = 'nwp_reg'

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(lead_times, error_df[(error_df.field == '2t') & (error_df.model == model)].groupby('lead_time')['rmse'].agg('mean'), '.-', color=temp_color, label='2t')
    ax.plot(lead_times, error_df[(error_df.field == 'tp') & (error_df.model == model)].groupby('lead_time')['rmse'].agg('mean'), '.-', color=precip_color, label='tp')
    ax.plot(lead_times, error_df[(error_df.field == '10ff') & (error_df.model == model)].groupby('lead_time')['rmse'].agg('mean'), '.-',color=wind_color, label='10ff')

    ax.legend(loc='upper right')
    ax.set_xlabel('Lead time [h]')
    ax.set_ylabel('RMSE')

    ax.set_xlim(0, lead_times[-1]+6)
    ax.set_xticks(lead_times)

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot RMSE vs lead time
    ## Compares models, field is set
    """)
    return


@app.cell
def _(color_dict, error_df, lead_times, plt):
    def _():
        fields = ['2t', 'tp', '10ff']
        models = ['nwp_reg']

        for field in fields:
            fig, ax = plt.subplots(figsize=(6,4))

            for model in models:
                ax.plot(lead_times, error_df[(error_df.field == field) & (error_df.model == model)].groupby('lead_time')['rmse'].agg('mean'), '.-', color=color_dict.get(field), label=model)
        
            # TODO
            ax.legend(loc='upper right')
            ax.set_xlabel('Lead time [h]')
            ax.set_ylabel('RMSE')
            ax.set_title(field)
        
            ax.set_xlim(0, lead_times[-1]+6)
            ax.set_xticks(lead_times)
        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot RMSE map
    """)
    return


@app.cell
def _(ccrs):
    projection = ccrs.Stereographic(
        central_latitude=90.0,          # North Pole
        central_longitude=-90.0,        # STAND_LON
        true_scale_latitude=60.0        # Usually TRUELAT2 for polar stereo
    )
    scatter_size=24
    edge_lw = 0.2
    return (projection,)


@app.cell
def _(cfeature, warnings):
    def set_basemap(ax):
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in create_collection",
            category=RuntimeWarning,
        )
        # Add base map features
        ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)
    return (set_basemap,)


@app.cell
def _(lead_times):
    print(lead_times)
    print(len(lead_times))
    return


@app.cell
def _(ccrs, error_df, lead_times, np, plt, projection, set_basemap):
    def _():
        rmse_map_field = '2t'

        fig, axs = plt.subplots(2,int(len(lead_times)/2), figsize=(3*len(lead_times), 20), subplot_kw={'projection': projection})
        axs = axs.ravel()

        vmin = np.nanmin(error_df[(error_df.field == rmse_map_field)]['rmse'].values)
        vmax = np.nanmax(error_df[(error_df.field == rmse_map_field)]['rmse'].values)

        for i, lt in enumerate(lead_times):
            set_basemap(axs[i])
            df = error_df[(error_df.field == rmse_map_field) & (error_df.lead_time == lt)]

            sc = axs[i].scatter(df.x, df.y, c=df.rmse, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            axs[i].set_title(f"Lead time : {str(lt)} h")

        plt.show()
        return None

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
