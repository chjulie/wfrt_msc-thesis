import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import pandas as pd
    import os
    from datetime import datetime

    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree

    import matplotlib.pyplot as plt
    return cKDTree, datetime, mo, np, os, pd, plt, xr


@app.cell
def _():
    obs_dir = "/scratch/juchar/eccc_data/"
    local_dir = "../../data/eccc_data/"
    return local_dir, obs_dir


@app.cell
def _(xr):
    climatex_ds = xr.open_dataset(
        "/scratch/juchar/prediction_data/bris-lam-inference-20230101T12-20230102T12.nc"
    )
    wrf_ds = xr.open_dataset("data/wrf_data/wrfout_d02_processed_23012000.nc")
    return climatex_ds, wrf_ds


@app.cell
def _(climatex_ds, np, wrf_ds):
    src_coords = np.column_stack(
        (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
    )  # source coords is WAC00WG-01 XLONG/XLAT
    tgt_coords = np.column_stack(
        (climatex_ds.longitude.values, climatex_ds.latitude.values)
    )
    return src_coords, tgt_coords


@app.cell
def _(cKDTree, np):
    def clip_stations(station_coords, src_coords, tgt_coords, tolerance=0.03):
        tree_src = cKDTree(src_coords)
        tree_tgt = cKDTree(tgt_coords)

        src_near = np.array(
            [
                len(idx) > 0
                for idx in tree_src.query_ball_point(station_coords, r=tolerance)
            ]
        )
        tgt_near = np.array(
            [
                len(idx) > 0
                for idx in tree_tgt.query_ball_point(station_coords, r=tolerance)
            ]
        )

        return src_near & tgt_near
    return (clip_stations,)


@app.cell
def _(clip_stations, obs_dir, os, pd, src_coords, tgt_coords):
    for obs_file in os.listdir(obs_dir):
        if obs_file[:5] == "eccc_":
            print(f" ⚡️ file : {obs_file}")
            obs_df = pd.read_csv(
                f"{obs_dir}/{obs_file}",
                converters={
                    "UTC_DATE": lambda x: pd.to_datetime(
                        x, format="%Y-%m-%dT%H:%M:%S"
                    ).tz_localize("utc")
                },
            )
            station_coords = obs_df[["x", "y"]].to_numpy()
            mask = clip_stations(
                station_coords=station_coords,
                src_coords=src_coords,
                tgt_coords=tgt_coords,
            )

            clipped_stations = obs_df[mask].copy()
            clipped_stations.to_csv(f"{obs_dir}/clipped_{obs_file}")
    return mask, obs_file


@app.cell
def _(obs_dir, obs_file, os):
    for clipped_file in os.listdir(obs_dir):
        if clipped_file.startswith("clipped_eccc"):
            print(f" ⚡️ file : {obs_file}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Merge into a single DataFrame
    """)
    return


@app.cell
def _(local_dir, os, pd):
    dfs = []

    for file in os.listdir(local_dir):
        if file.startswith("clipped_eccc"):
            # print(f" ⚡️ file : {file}")
            df_temp = pd.read_csv(f"{local_dir}/{file}").drop(columns=["Unnamed: 0"])
            dfs.append(df_temp)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(f"{local_dir}/all_eccc_data.csv", index=False)
    final_df
    return


@app.cell
def _(local_dir, pd):
    df = pd.read_csv(f"{local_dir}/all_eccc_data.csv")
    df
    return (df,)


@app.cell
def _(df):
    print(df.UTC_DATE)
    print(type(df.UTC_DATE[0]))
    print(df.UTC_DATE[0])
    return


@app.cell
def _(datetime, df):
    df['datetime'] = df['UTC_DATE'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))
    df['datetime']
    return


@app.cell
def _(df):
    print(df.datetime.values)
    return


@app.cell
def _(df):
    init_hours = [00, 6, 12, 18]
    mask = df['datetime'].apply(lambda x: x.hour in init_hours)
    print(mask)
    return (mask,)


@app.cell
def _(df, mask):
    df_init = df[mask]
    df_init
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Exploratory data analysis
    """)
    return


@app.cell
def _(df, np, plt):
    columns = ["TEMP", "PRECIP_AMOUNT", "WIND_SPEED"]

    for col in columns:
        print(f"{col}")
        print(" - max : ", np.nanmax(df[col]))
        print(" - min : ", np.nanmin(df[col]))
        print(" - mean : ", np.nanmean(df[col]))
        print(" - median : ", np.nanmedian(df[col]))
        print(" - std : ", np.nanstd(df[col]))

        if col == "PRECIP_AMOUNT":
            df[col].hist(bins=200, log=True)
        else:
            df[col].hist(bins=200)
        plt.title(f"ECCC data distribution - {col}")
        plt.show()
    return


@app.cell
def _(df, plt):
    # Cut precip outliers
    precip_amount_capped = df[df['PRECIP_AMOUNT'] < 70]['PRECIP_AMOUNT']
    precip_amount_capped.hist(bins=200, log=True)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
