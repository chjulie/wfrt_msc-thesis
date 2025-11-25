import argparse
import subprocess
import sys
import datetime

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

    date_str = args.date + "T06:00:00"
    date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    utc_date = date - datetime.timedelta(hours=8)   # conversion from PST to UTC dates

    # end_date = args.end_date
    fields = {'verif': VERIF_FIELDS,
              'visibility': VISIBILITY_FIELDS,
              }.get(args.fields_type)

    download_cmd = f'wget -O {OBS_DATA_DIR}/{args.date.replace('-','')}_{args.fields_type}_eccc_obs.csv "https://api.weather.gc.ca/collections/climate-hourly/items?bbox={DOMAIN_MINX},{DOMAIN_MINY},{DOMAIN_MAXX},{DOMAIN_MAXY}&datetime={utc_date.strftime(format="%Y-%m-%dT%H:%M:%S")}Z&properties={fields}&sortby=UTC_DATE&f=csv"' # Date is specified using local time!!
    print(' > executing command: ', download_cmd)
    subprocess.run(download_cmd, shell=True, check=True)

