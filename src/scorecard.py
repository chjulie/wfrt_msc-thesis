import argparse
import subprocess
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np

from utils.resampling_utils import pyresample_resampling
from utils.scorecard_evaluator import scorecard_evaluator_factory
from utils.data_constants import P_LEVELS, EVAL_LEAD_TIMES, MODEL_ID, FIR_SCRATCH

SCORECARD_FIELDS = ["2t"]


def rclone_copy(run_id: str):

    source = f"wfrt-nextcloud:Documents/WRF-forecasts/{MODEL_ID}/wrfout_d02_processed_{run_id}.nc"
    destination = f"{FIR_SCRATCH}/wrf_data"
    cmd = f"rclone copy '{source}' '{destination}' --progress"
    # print(' - rclone cmd: ', cmd)
    subprocess.run(cmd, shell=True, check=True)


xtime = lambda a, b: datetime.strftime(
    a + pd.Timedelta(hours=int(b)), format="%Y-%m-%dT%H:00:00.000000000"
)


def clip_coords(src_coords, tgt_coords, tolerance=0.03):

    tree_src = cKDTree(src_coords)
    indices = tree_src.query_ball_point(tgt_coords, r=tolerance)
    tgt_mask = np.array([len(idx) > 0 for idx in indices])

    tree_tgt = cKDTree(tgt_coords)
    indices = tree_tgt.query_ball_point(src_coords, r=tolerance)
    src_mask = np.array([len(idx) > 0 for idx in indices])

    return tgt_mask, src_mask


def get_domain_coords(wrf_ds, climatex_ds):

    src_coords = np.column_stack(
        (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
    )
    tgt_coords = np.column_stack(
        (climatex_ds.longitude.values, climatex_ds.latitude.values)
    )

    tgt_mask, src_mask = clip_coords(src_coords, tgt_coords)
    clipped_src_coords = src_coords[src_mask]
    clipped_tgt_coords = tgt_coords[tgt_mask]

    return clipped_src_coords, clipped_tgt_coords, src_mask, tgt_mask


def rmse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):

    assert array1.shape == array2.shape, "Both arrays must have the same shape"
    assert array1.ndim == 1 and array2.ndim == 1, "Both arrays must be one-dimesional"

    rmse = np.sqrt(np.power(array1 - array2, 2)).mean()  # spatial average

    return rmse


if __name__ == "__main__":

    """
    1. CLIMATEX INFERENCE
    1.1 Ground truth
    1.2 Prediction for given lead time
    1.3 Evaluation of prediction

    2. WRF FORECASTS
    2.1 Download and open file for given initial date
    2.2 Select XTIME=initial_date+lead_time
    2.3 Grid interpolation
    2.4 Evaluation of prediction, open file using given lead time

    3. SAVING ERROR DATA
    3.1 Compute metrics diff (average in space)
    3.2 Average in time
    3.3 Save to scorecard.csv
    """

    
    parser = argparse.ArgumentParser(description="Evaluation againts Climatex")
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
        "--model", type=str, required=True, help="'dl_reg' or 'nwp_reg'."
    )
    args = parser.parse_args()
    
    # --------------------------------------------------------------
    # climatex_ds = xr.open_dataset(f"{FIR_SCRATCH}/data/cleaned_data/anemoi-climatex-training-6h-20230101-20231231.zarr", engine='zarr')
    # for i,a in enumerate(climatex_ds.attrs['variables']):
    #     print(i,a)
    # print(climatex_ds)

    # --------------------------------------------------------------

    date_range = pd.date_range(
        start=args.start_date, end=args.end_date
    )  # should be at 00 everyday

    # print(" - daterange: ", date_range)

    print(" > Instantiating evaluator", flush=True)
    evaluator = scorecard_evaluator_factory(
        model_name=args.model,
        date_range=date_range,
        lead_times=EVAL_LEAD_TIMES,
    )

    print(" > Starting evaluation", flush=True)
    evaluator.evaluate()

    print(" > Program finished successfully !", flush=True)

    # --------------------------------------------------------------