import argparse
import pandas as pd
import subprocess
import sys
import os

sys.path.append('../')
from data_constants import *

if __name__ == "__main__":
    fields_type = 'visibility'
    limit = '10000' # max number of lines do download at once

    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--start_date", type=str, required=True, help="Date in YYYY-mm-dd format"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date in YYYY-mm-dd format"
    )
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    fields = {'verif': VERIF_FIELDS,
              'visibility': VISIBILITY_FIELDS,
              }.get(fields_type)
    
    monthly_date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    file_dir = f"{OBS_DATA_DIR}/{start_date.replace('-','')}_{fields_type}_eccc_obs"
    os.makedirs(file_dir, exist_ok=True)

    for i in range(len(monthly_date_range)-1): 
        ms_date = monthly_date_range[i].strftime('%Y-%m-%d')
        me_date = monthly_date_range[i+1].strftime('%Y-%m-%d')
        # download_cmd = f'wget -O {file_dir}/{ms_date.replace('-','')}_monthly_obs.csv "https://api.weather.gc.ca/collections/climate-hourly/items?bbox={DOMAIN_MINX},{DOMAIN_MINY},{DOMAIN_MAXX},{DOMAIN_MAXY}&datetime={ms_date}T08:00:00Z/{me_date}T07:00:00Z&properties={fields}&limit={limit}&sortby=UTC_DATE&f=csv"'
        download_cmd = f'wget -O {file_dir}/{ms_date.replace('-','')}_monthly_obs.csv "https://api.weather.gc.ca/collections/climate-hourly/items?datetime={ms_date}T08:00:00Z/{me_date}T07:00:00Z&properties={fields}&limit={limit}&sortby=UTC_DATE&f=csv"'
        subprocess.run(download_cmd, shell=True, check=True)