import argparse
import subprocess
import sys
from datetime import datetime

sys.path.append('../')
from data_constants import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--date", type=str, required=True, help="Date in YYYY-mm-dd format"
    )
    # parser.add_argument(
    #     "--end_date", type=str, required=True, help="End date in YYYY-mm-dd format"
    # )
    parser.add_argument(
        "--fields_type", type=str, required=True, help="'verif' or 'visibility'"
    )
    args = parser.parse_args()

    date = args.date
    # end_date = args.end_date
    fields = {'verif': VERIF_FIELDS,
              'visibility': VISIBILITY_FIELDS,
              }.get(args.fields_type)

    download_cmd = f'wget -O {OBS_DATA_DIR}/{date.replace('-','')}_{args.fields_type}_eccc_obs.csv "https://api.weather.gc.ca/collections/climate-hourly/items?bbox={DOMAIN_MINX},{DOMAIN_MINY},{DOMAIN_MAXX},{DOMAIN_MAXY}&datetime={date}T06:00:00Z&properties={fields}&sortby=UTC_DATE&f=csv"'
    print(' > executing command: ', download_cmd)
    subprocess.run(download_cmd, shell=True, check=True)

