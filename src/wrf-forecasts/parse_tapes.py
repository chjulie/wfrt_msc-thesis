import csv
import re
import pandas as pd


def get_unique_volume_tags(lines, model_name):

    tags = []

    for line in lines:
        m = re.search(r"VolumeTag = (\w+),File number=(\d+),.*/(\d{8})", line)
        if m:
            tag, _, _ = m.groups()
            tags.append(tag)

    unique_volume_tags = list(set(tags))
    outfile = f"src/wrf-forecasts/{model_name}-unique_volume_tags.csv"

    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["VolumeTag"])
        for tag in sorted(unique_volume_tags):
            w.writerow([tag])


def get_daily_infos(lines, model_name):

    df = pd.DataFrame(columns=["Date", "Volume tag", "File number"])

    for line in lines:
        m = re.search(r"VolumeTag = (\w+),File number=(\d+),.*/(\d{8})", line)
        if m:
            tag, filenum, dt = m.groups()
            yy = int(dt[:2])
            yyyy = 2000 + yy
            mm = dt[2:4]
            dd = dt[4:6]
            hh = dt[6:8]
            iso_date = f"{yyyy}-{mm}-{dd} {hh}:00"
            df.loc[len(df)] = [iso_date, tag, filenum]

    df.to_csv(f"src/wrf-forecasts/{model_name}-tapes_daily_infos.csv")


if __name__ == "__main__":

    model_name = "WR"
    file_path = f"src/wrf-forecasts/{model_name}.txt"

    with open(file_path, "r") as f:
        lines = f.readlines()

    get_unique_volume_tags(lines, model_name)
    get_daily_infos(lines, model_name)
