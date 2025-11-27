import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sys
    sys.path.append('../')

    from data_constants import OBS_DATA_DIR
    return OBS_DATA_DIR, go, mo, pd


@app.function
def show_serie_stats(serie):
    mean = serie.mean()
    std = serie.std()
    min = serie.min()
    max = serie.max()
    return mean, std, min, max


@app.cell
def _(OBS_DATA_DIR):
    file_name = f"../../{OBS_DATA_DIR}/20251126_verif_eccc_obs.csv"
    return (file_name,)


@app.cell
def _(file_name, pd):
    df = pd.read_csv(file_name)
    df.TEMP = df.TEMP.apply(lambda x: x + 273.15)
    df.head(5)
    return (df,)


@app.cell
def _(df):
    print(df.dtypes)
    return


@app.cell
def _(df):
    # show statistics for each column
    df.describe(include='all')
    return


@app.cell
def _(df):
    df.TEMP
    return


@app.cell
def _(df):
    df.x
    return


@app.cell
def _(go, pd):
    def plot_values(df: pd.DataFrame, field: str):

        fig = go.Figure(go.Scattermap(
            lat=df.y,
            lon=df.x,
            mode='markers',
            marker=go.scattermap.Marker(
                size=17,
                color=df[field],
                opacity=0.7,
                colorbar=dict(
                    title=f"{field}",
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
        title=dict(text=f"ECCC stations observation for {field}"),
        height=600,
        width=500,
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
    date = "2025-08-01T08:00:00"
    print(date)
    daily_df = df[(df["UTC_DATE"]=="2025-08-01T16:00:00")]
    daily_df.describe()
    plot_values(daily_df, "TEMP")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pre-processing brainstorming
    - Exclude outliers (double than 90th percentile)
    - Exclude missing values, but only for that field (exceot for wind: requires 2 non-missing values)
    """)
    return


if __name__ == "__main__":
    app.run()
