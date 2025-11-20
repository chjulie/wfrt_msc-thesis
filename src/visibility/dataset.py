import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import plotly.graph_objects as go
    import os
    import sys
    sys.path.append('../')
    return go, mo, np, os, pd, plt


@app.cell
def _():
    OBS_DATA_DIR = "../../data/eccc_data"
    start_date = '20220101'
    file_dir = f"{OBS_DATA_DIR}/{start_date}_visibility_eccc_obs"
    clean_data_path = f"{OBS_DATA_DIR}/clean/{start_date}_df_clean.csv"
    return clean_data_path, file_dir


@app.cell
def _(file_dir, os, pd):
    li = []

    for f in os.listdir(file_dir):
        temp = pd.read_csv(f"{file_dir}/{f}")
        li.append(temp)

    df_raw = pd.concat(li, axis=0, ignore_index=True)
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    df = df_raw.dropna(how='any',axis=0) 
    df = df[df.VISIBILITY > 0.01]
    df
    return (df,)


@app.cell
def _(df):
    print(df.UTC_DATE)
    return


@app.cell
def _(df, np):
    station_ids = np.unique(df.STN_ID)
    station_ids_w_precip = [574, 8040, 8045, 29733, 43004, 43963, 54338]
    station_ids_w_visibility = [51317, 51419, 52198, 52798,]
    print(station_ids)
    return (station_ids,)


@app.cell
def _():
    var_1 = 'TEMP'
    var_2 = 'DEW_POINT_TEMP'
    #var_1 = 'VISIBILITY'
    #var_2 = 'PRECIP_AMOUNT'
    return


@app.cell
def _(plt, station_ids):
    n = len(station_ids)
    fig, ax = plt.subplots(n,2,figsize=(15,3*n))

    #for i in range(n):
    #    ax[i,0].scatter(df[df.STN_ID == station_ids[i]].UTC_DATE, df[df.STN_ID == station_ids[i]][var_1], s=4)
    #    ax[i,1].scatter(df[df.STN_ID == station_ids[i]].UTC_DATE, df[df.STN_ID == station_ids[i]][var_2], s=4)
    #    ax[i,0].set_ylabel(f"STN_ID={station_ids[i]}")

    #    if i == n-1:
    #        for j in [0,1]:
    #            ax[i,j].xaxis.set_major_locator(plt.MaxNLocator(4))
    #            ax[i,j].xaxis.set_tick_params(rotation=45)

    #    if i == 0:
    #        ax[i,0].set_title(f"{var_1}")
    #        ax[i,1].set_title(f"{var_2}")

    #plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # To-do
    1. Remove precip_amount outliers (>100mm)
    2. Check visibility unit in eccc technical documentation: 0.1km
        - Relative humidity: %
        - Dewpoint temp, temp: °C
        -
    3. Plot stations localization and select most appropriates
    """)
    return


@app.cell
def _(df):
    df_stn = df.groupby(by='STN_ID')[['x','y']].agg(['mean','size'])
    df_stn
    return


@app.cell
def _():
    return


@app.cell
def _(go, pd):
    def plot_values(df: pd.DataFrame):

        fig = go.Figure(go.Scattermap(
            lat=df['y', 'mean'],
            lon=df['x', 'mean'],
            mode='markers',
            marker=go.scattermap.Marker(
                size=17,
                color=df['x','size'],
                opacity=0.7,
                colorbar=dict(
                    title="Nb of observations",
                    thickness=20,
                    x=1.02,
                    y=0.5,
                    len=0.7,
                    outlinewidth=0,
                    tickfont=dict(size=12)
                )
            ),
        ))

        fig.update_layout(
        title=dict(text="ECCC stations observation stations"),
        height=600,
        width=800,
        showlegend=False,
        map=dict(
            bearing=0,
            center=dict(
                lat=55,
                lon=-125
            ),
            pitch=0,
            zoom=2,
            style='light'
        ),
        )

        fig.show()
    return (plot_values,)


@app.cell
def _(df, plot_values):
    plot_values(df.groupby(by='STN_ID')[['x','y']].agg(['mean','size']))
    return


@app.cell
def _(df_raw, plot_values):
    plot_values(df_raw.groupby(by='STN_ID')[['x','y']].agg(['mean','size']))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Linear regression
    """)
    return


@app.cell
def _(df, pd):
    print(type(df['UTC_DATE'][76]))
    df.loc[:,'dt_UTC_DATE'] = pd.to_datetime(df['UTC_DATE'], format='%Y-%m-%dT%H:%M:%S').copy()
    print(type(df['dt_UTC_DATE'][76]))
    return


@app.cell
def _(df, np):
    # Add additional predictants
    # from anemoi-datasets: The Julian day is the number of days since the 1st of January at 00:00 of the current year. cos(julian_day/365.25 ∗ 2 ∗ π)
    df.loc[:,'julian_day'] = (df["dt_UTC_DATE"] - df["dt_UTC_DATE"].dt.to_period("Y").dt.start_time).dt.days
    df.loc[:,'cos_day'] = np.cos(df['julian_day'] / (365.25 * 2 * np.pi))
    df.loc[:,'sin_day'] = np.sin(df['julian_day'] / (365.25 * 2 * np.pi))

    df.loc[:,'time'] = df['dt_UTC_DATE'].dt.hour # Using local time!!
    df.loc[:,'cos_time'] =  np.cos(df['time'] / (24 * 2 * np.pi))
    df.loc[:,'sin_time'] =  np.sin(df['time'] / (24 * 2 * np.pi))
    df.head(10)
    #df['sine_time'] = 
    return


@app.cell
def _(df):
    # transform target
    # see notes for computations
    beta = 3.912 / (100 * df['VISIBILITY'])
    df.loc[:,'target_y'] = beta
    return


@app.cell
def _(df):
    print(df.columns)
    return


@app.cell
def _(clean_data_path, df):
    save_cols = ['ID', 'julian_day', 'target_y', 'TEMP', 'PRECIP_AMOUNT', 'RELATIVE_HUMIDITY', 'VISIBILITY', 'cos_day', 'sin_day', 'cos_time', 'sin_time']
    df.to_csv(clean_data_path, columns=save_cols, index=False)
    return


@app.cell
def _(df, np):
    print(df['target_y'].apply(np.isinf).sum())
    print(df['target_y'].max())
    return


@app.cell
def _(df):
    print(len(df[df.VISIBILITY < 0.001]))
    return


@app.cell
def _(df, plt):
    def _():
        fig, ax = plt.subplots()

        ax.scatter(df.RELATIVE_HUMIDITY, df.target_y)
        ax.set_xlabel('prognostic variable')
        ax.set_ylabel('target variable')
        return plt.show()


    _()
    return


@app.cell
def _(df):
    df['target_y'].hist(bins=100)
    return


@app.cell
def _(df, plt):
    plt.scatter(df.dt_UTC_DATE, df.target_y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
