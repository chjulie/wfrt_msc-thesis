import argparse
import datetime

import earthkit.data as ekd
import earthkit.regrid as ekr
from ecmwf.opendata import Client as OpendataClient

def get_mars_data(param, date_str):
    data = ekd.from_source("mars", request={
        "param": param,
        "levtype": "sfc",
        "date": date_str,
        "time": "12:00:00",
        "stream": "oper",
        "expver": "1",
        "type": "fc",
        "target": "output",
        })
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--date", type=str, required=True, help="Date in YYYY-mm-dd format"
    )
    args = parser.parse_args()

    date_str = args.date
    param = "20.3"  # 20.3: visibility

    get_mars_data(param, date_str)
    print(" > Program finished successfully !")