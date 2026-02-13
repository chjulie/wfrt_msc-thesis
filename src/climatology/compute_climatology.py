import os
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import metpy.calc as mpcalc
from metpy.units import units
import subprocess


from climatology_process_utils import (
    process_wrf,
    get_previous_datetime,
    DATA_DIR,
)
from climatology_download_utils import (
    get_rclone_source,
    get_subfolders,
    find_water_year_subfolder,
)

def get_date_range(start,end):
    daily_range = pd.date_range(start=start, end=end, freq='D')

    offsets = [
        pd.Timedelta(hours=0),
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12),
        pd.Timedelta(hours=18),
    ]

    date_range = [day + off for day in daily_range for off in offsets]
    return date_range


def download(datetime, valid_subfolders, file_type):
    print(' ⚡️ datetime: ', datetime)
    rclone_source = get_rclone_source(datetime.year)

    # year folder 
    wyear_folder = find_water_year_subfolder(datetime, valid_subfolders)
    if wyear_folder[4:8] == '2020':
        wyear_folder += '_NONUDGE'

    # month folder
    if not ((datetime.day == 1) and (datetime.hour == 0)):
        month_subfolder = "metgrid_" + str(datetime.year) + "_" + datetime.strftime("%m")
    else:
        # For first hour of water year, go to previous month folder
        wyear_date = datetime - pd.DateOffset(months=1)
        month_subfolder = (
            "metgrid_" + str(wyear_date.year) + "_" + wyear_date.strftime("%m")
        )

    if file_type == 'WRFOUT':
        file_name = f"wrfout_d03_{datetime.strftime('%Y-%m-%d_%H:%M:%S')}_compressed"
        source = f"{rclone_source}:/Share_Data/{wyear_folder}/{month_subfolder}/WRFOUT/{file_name}"
    elif file_type == 'WRFUVIC':
        file_name = f"wrfuvic_d03_{datetime.strftime('%Y-%m-%d_%H:%M:%S')}"
        source = f"{rclone_source}:/Share_Data/{wyear_folder}/{month_subfolder}/WRFUVIC/{file_name}"
    
    destination = f"/scratch/juchar/climatology/"

    # download file
    cmd = f"rclone copy '{source}' {destination}"
    print(f" ⚙️ Executing rclone cmd: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

    return file_name



if __name__ == "__main__":
    print('Starting script')
    os.makedirs("/scratch/juchar/climatology/data/", exist_ok=True)
    # TODO : extend to every year
    # TODO : extend to every day of year

    daily_range = get_date_range(start='2023-01-01', end='2023-01-01')
    offsets = [
        pd.Timedelta(hours=0),
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12),
        pd.Timedelta(hours=18),
    ]

    date_range = [day + off for day in daily_range for off in offsets]

    valid_subfolders = get_subfolders()
  
    
    # download wrfout file t-1
    prev = date_range[0]-pd.Timedelta(hours=6)
    for yr in [1989, 1990]:
        previous_datetime = datetime(yr, prev.month, prev.day, prev.hour)
        _ = download(datetime=previous_datetime, valid_subfolders=valid_subfolders, file_type='WRFOUT')
        _ = download(datetime=previous_datetime, valid_subfolders=valid_subfolders, file_type='WRFUVIC')

    for date in daily_range:
        print(" [INFO] ⚡️ date")
        for year in range(1990,1992):
            print(f" [INFO] 🌍 year : {year}")

            try:
                current_datetime = datetime(year, date.month, date.day, date.hour)
            except ValueError:
                continue

            print(f" [INFO] current datetime : {current_datetime}")

            # download files
            try:
                wrfout_f = download(datetime=current_datetime, valid_subfolders=valid_subfolders, file_type='WRFOUT')
                wrfuvic_file = download(datetime=current_datetime, valid_subfolders=valid_subfolders, file_type='WRFUVIC')
            except subprocess.CalledProcessError as e:
                print(f" ⚠️ [WARNING] File download for {current_datetime} : {e}")
                continue

            # process file and save
            try: 
                ds = process_wrf(wrfout_f, wrfuvic_file)
            except subprocess.CalledProcessError as e:
                print(f" ⚠️ [WARNING] Dataset processing failed for {current_datetime}")
            ds['time'] = [current_datetime]
            # date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
            # ds["dates"] = xr.DataArray(date, dims=("date",))

            ds_folder = f"/scratch/juchar/climatology/{date.strftime('%m-%dT%H:%M:%S')}"
            ds_path = f"{ds_folder}/{year}.nc"
            os.makedirs(ds_folder, exist_ok=True)
            ds.to_netcdf(ds_path, mode="w", unlimited_dims=["time"])
            print(f" [INFO] temp ds saved to {ds_path}")

        clim_ds = xr.open_mfdataset(f"{ds_folder}/*.nc", combine="by_coords", data_vars="all")
        # print(' - combined dataset : ', clim_ds)

        # take mean and save dataset
        clim_ds.mean(dim="time").to_netcdf(f"/scratch/juchar/climatology/data/{date.strftime('%m-%dT%H:%M:%S')}.nc")
        print(f" [INFO] ✅ Saved climatology for day {date.strftime('%m-%dT%H:%M:%S')} ")

        # remove files from previous datetime
        dt_minus_6 = get_previous_datetime(current_datetime, time_delta=6)
        file_minus_6 = f"wrfout_d03_{dt_minus_6.strftime('%Y-%m-%d_%H:%M:%S')}"

        if os.path.exists(os.path.join(DATA_DIR, file_minus_6)):
            os.remove(os.path.join(DATA_DIR, file_minus_6))



        # take the mean across time dimension

        # save combined dataset



