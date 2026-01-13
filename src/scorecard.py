import argparse
import datetime
import pandas as pd
import numpy as np

from data_constants import P_LEVELS, EVAL_LEAD_TIMES

SCORECARD_VARS = [""]


if __name__ == "__main__":

    """
    1. CLIMATEX INFERENCE
    1.1 Ground truth
    1.2 Prediction for given lead time
    1.3 Evaluation of prediction

    2. WRF FORECASTS
    2.1 Prediction for given lead time
    2.2 Evaluation of prediction

    3. SAVING ERROR DATA
    3.1 Compute metrics diff (average in space)
    3.2 Average in time
    3.3 Save to scorecard.csv
    """
    parser = argparse.ArgumentParser(description="Evaluation againts Climatex")
    parser.add_argument(
        "--start_date",
        type="str",
        required=True,
        help="Start date of evaluation period, format: YYYY-mm-dd",
    )
    parser.add_argument(
        "--end_date",
        type="str",
        required=True,
        help="End date of evaluation period, format: YYYY-mm-dd",
    )
    args = parser.parse_args()
    date_range = pd.date_range(
        start=args.start_date, end=args.end_date
    )  # should be at 00 everyday

    scorecard_df = pd.DataFrame(
        columns=["initial_date", "lead_time", "fields", "rmse_wf", "rmse_climatex"]
    )

    for date in date_range:
        print(f"⚡️ date: {date}")
        # 2.1 WAC00WG-01 pred for date
        # rclone download for date

        for lead_time in EVAL_LEAD_TIMES:
            print(f"  lead time: {lead_time}")

            # 1.1 CLIMATEX ground truth
            # open file select initial_date=date+lead_time, lead_time=0

            # 1.2 CLIMATEX pred for date
            # open file, select initial_date=date, lead_time=0

            # 2.1 WAC00WG-01 pred for date
            # open file, select XTIME=date+lead_time
            # grid interpolation

            # 1.3 Evaluate CLIMATEX pred

            # 1.4 Evaluate WAC00WG-01 pred

            # 3.1 Performance difference and spatial average

    # 3.2 Average in time

    # 3.3 Save error data
