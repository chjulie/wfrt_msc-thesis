import argparse
import subprocess
import sys
from datetime import datetime
import pandas as pd

from src.utils.data_constants import FIR_SCRATCH, DOMAIN_MINX, DOMAIN_MAXX, DOMAIN_MINY, DOMAIN_MAXY

ECCC_FIELDS = "ID%2CSTN_ID%2CUTC_DATE%2CTEMP%2CPRECIP_AMOUNT%2CWIND_SPEED%2CWIND_DIRECTION"

if __name__ == "__main__":
    # run with uv run python -m src.observations.download --start_date 2023-01-01 --end_date 2024-01-04
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--start_date", type=str, required=True, help="Date in YYYY-mm-dd format"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date in YYYY-mm-dd format"
    )

    args = parser.parse_args()
    daily_range = pd.date_range(start=args.start_date, end=args.end_date)
    offsets = [
        pd.Timedelta(hours=0),
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12),
        pd.Timedelta(hours=18),
    ]
    date_range = [(day + off).tz_localize('utc') for day in daily_range for off in offsets]

    for utc_date in date_range:

        # download .csv file 
        print(f"⚡️ utc_date={utc_date.strftime(format="%Y-%m-%dT%H:%M:%S")}")
        download_cmd = f'wget -O {FIR_SCRATCH}/eccc_data/eccc_{utc_date.strftime(format="%Y-%m-%dT%H:%M:%S")}.csv "https://api.weather.gc.ca/collections/climate-hourly/items?bbox={DOMAIN_MINX},{DOMAIN_MINY},{DOMAIN_MAXX},{DOMAIN_MAXY}&UTC_DATE={utc_date.strftime(format="%Y-%m-%dT%H:%M:%S")}Z&properties={ECCC_FIELDS}&sortby=UTC_DATE&f=csv"' # Date is specified using local time!!
        print(' > executing command: ', download_cmd)
        subprocess.run(download_cmd, shell=True, check=True)



