import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from datetime import datetime
    return (pd,)


@app.cell
def _(pd):
    def get_datetime_range(start_date: str, end_date: str):

        daily_range = pd.date_range(
            start=start_date, end=end_date
        )
        offsets = [
            pd.Timedelta(hours=0),
            pd.Timedelta(hours=6),
            pd.Timedelta(hours=12),
            pd.Timedelta(hours=18),
        ]

        date_range = [day + off for day in daily_range for off in offsets]

        return date_range

    return (get_datetime_range,)


@app.cell
def _(get_datetime_range, pd):
    date_range = get_datetime_range(start_date='2023-01-01', end_date='2024-01-04')
    # stations_statistics = pd.DataFrame(columns=['STN_ID', 'x', 'y', 'size_TEMP', 'size_PRECIP'])
    stations_dict = {}

    for date in date_range:
        print(f" ⚡️ date : {date}")
        obs_file_path = f"../scratch/eccc_data/eccc_{date.strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        obs_df = pd.read_csv(
                obs_file_path,
                converters={
                    "UTC_DATE": lambda x: pd.to_datetime(
                        x, format="%Y-%m-%dT%H:%M:%S"
                    ).tz_localize("utc")
                },
            )
        obs_df["WIND_VALID"] = (obs_df["WIND_SPEED"].notna() & obs_df["WIND_DIRECTION"].notna())
        grouped = (
            obs_df
            .groupby("STN_ID")
            .agg(
                x=("x", "first"),
                y=("y", "first"),
                size_TEMP=("TEMP", "count"),
                size_PRECIP=("PRECIP_AMOUNT", "count"),
                size_WIND=("WIND_VALID", "sum"),
            )
        )
        for stn_id, row in grouped.iterrows():
            if stn_id not in stations_dict:
                print(f"new stn_id : {stn_id}")
                stations_dict[stn_id] = {
                    "STN_ID": stn_id,
                    "x": row["x"],
                    "y": row["y"],
                    "size_TEMP": int(row["size_TEMP"]),
                    "size_PRECIP": int(row["size_PRECIP"]),
                    "size_WIND": int(row["size_WIND"]),
                }
            else:
                stations_dict[stn_id]["size_TEMP"] += int(row["size_TEMP"])
                stations_dict[stn_id]["size_PRECIP"] += int(row["size_PRECIP"])
                stations_dict[stn_id]["size_WIND"] += int(row["size_WIND"])

            
        print(f"len(stations_dict) : {len(stations_dict)}")

    stations_statistics = pd.DataFrame.from_dict(stations_dict, orient="index").reset_index(drop=True)
    return (stations_statistics,)


@app.cell
def _(stations_statistics):
    stations_statistics
    return


@app.cell
def _(stations_statistics):
    stations_statistics.to_csv('../scratch/eccc_data/stations_statistics.csv', index=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
