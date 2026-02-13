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
    return ERROR_DATA_DIR, pd, plt, rc


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

    nwp_reg_color = "#A61166"
    dl_reg_color = "#F26419"
    dl_glob_color = "#54A085" #A5E972
    return


@app.cell
def _(ERROR_DATA_DIR, pd):
    file_name = 'scorecard-nwp_reg-20230306_20230306.csv'
    error_df = pd.read_csv(f"../{ERROR_DATA_DIR}/{file_name}").drop(columns=['Unnamed: 0'])
    error_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
