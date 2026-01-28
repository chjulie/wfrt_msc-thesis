import argparse
import datetime
import pandas as pd
import numpy as np

from utils.data_reading_utils import read_pkl
from utils.resampling_utils import pyresample_resampling, scipy_resampling
from utils.model_forecast_evaluator import model_forecast_evaluator_factory
from utils.data_constants import EVAL_LEAD_TIMES

if __name__ == "__main__":
    """
    Evaluates model prediction against observations

    """
    # ARGS PARSING -----------------------------------------------------
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of evaluation period, format: YYYY-mm-dd",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of evaluation period, format: YYYY-mm-dd",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="'dl_reg', 'nwp_reg' or 'dl_glob'."
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------

    date_range = pd.date_range(
        start=args.start_date, end=args.end_date
    )  # should be at 00 everyday

    evaluator = model_forecast_evaluator_factory(
        model_name=args.model,
        date_range=date_range,
        lead_times=EVAL_LEAD_TIMES,
    )
    evaluator.evaluate()
    print(" > Program finished successfully !")
