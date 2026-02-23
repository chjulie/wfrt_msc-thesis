import marimo

__generated_with = "0.18.0"
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

    return mo, pd, plt, rc


@app.cell
def _(plt, rc):
    rc("font", **{"family": "serif", "serif": ["Times New Roman"], "size": "14"})
    rc("text", usetex=False)
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
        '10ff': "#FFC247",
        'dl_reg': 'darkslategrey', #"#A61166",
        'dl_glob': 'olive',
    }


    return (color_dict,)


@app.cell
def _(pd):
    folder = 'data/error_data/'
    #dl_glob = pd.read_csv(f"{folder}errors-global-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    dl_reg = pd.read_csv(f"{folder}errors-stage-c-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    return (dl_reg,)


@app.cell
def _(dl_reg):
    dl_reg
    return


@app.cell
def _(mo):
    mo.md(r"""
    # RMSE vs lead time for temperature and wind
    """)
    return


@app.cell
def _(dl_reg):
    temp_reg_df = dl_reg[dl_reg.field == '2t']
    temp_reg_df['nrmse'] = temp_reg_df['rmse'] / temp_reg_df['obs_value'].mean()
    temp_reg_df
    return (temp_reg_df,)


@app.cell
def _(temp_reg_df):
    grouped_temp_reg_df = temp_reg_df.groupby('lead_time')['nrmse'].agg('mean')
    grouped_temp_reg_df
    return (grouped_temp_reg_df,)


@app.cell
def _(color_dict, dl_reg):
    lead_times = dl_reg.lead_time.unique()
    rmse_fields = ['2t', '10ff']
    models = {'dl_reg': 
                 {
                    'color': color_dict['dl_reg'],
                    'name': 'Regional GNN',
                    'data_2t': dl_reg[dl_reg.field == '2t'].groupby('lead_time')['rmse'].agg('mean'),
                    'data_10ff': dl_reg[dl_reg.field == '10ff'].groupby('lead_time')['rmse'].agg('mean'),
                 },
    #        'dl_glob':
     #            {
      #              'color': color_dict['dl_glob'] ,
      #              'name': 'Global GNN',
       #             'data_2t': dl_glob[dl_glob.field == '2t'].groupby('lead_time')['rmse'].agg('mean'),
       #             'data_10ff': dl_glob[dl_glob.field == '10ff'].groupby('lead_time')['rmse'].agg('mean'),
        #         },
             }
    return lead_times, models, rmse_fields


@app.cell
def _(grouped_temp_reg_df, lead_times, models, plt, rmse_fields):
    for field in rmse_fields:
        fig, ax = plt.subplots(figsize=(6,4))

        for model, mdict in models.items():
            ax.plot(lead_times, grouped_temp_reg_df, '.-', color=mdict['color'], label=mdict['name'])

        # TODO
        ax.legend(loc='upper right')
        ax.set_xlabel('Lead time [h]')
        ax.set_ylabel('nRMSE')
        ax.set_title(field)

        ax.set_xlim(0, lead_times[-1]+6)
        ax.set_xticks(lead_times)

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Precipitation analysis
    1. Equitable threat score
    2. Q-Q plot
    """)
    return


@app.cell
def _(dl_reg):
    rain_threshold = 1 # 1mm accumulated over 6 hours

    precip_reg = dl_reg[dl_reg.field == 'tp']
    precip_reg['binary_obs'] = precip_reg['obs_value'] > rain_threshold
    precip_reg['binary_pred'] = precip_reg['pred_value'] > rain_threshold
    print(precip_reg.pred_value.values[:20])
    precip_reg
    return (precip_reg,)


@app.cell
def _(precip_reg):
    precip_reg.describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
