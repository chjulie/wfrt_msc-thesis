import argparse
import pandas as pd 
from data_constants import OBS_DATA_DIR, PRED_DATA_DIR

if __name__ == "__main__":
    '''
    Evaluates model prediction against observations

    '''
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--date", type=str, required=True, help="Date in YYYY-mm-dd format"
    )
    parser.add_argument(
        "--field", type=str, required=True, help="'temp'"
    )
    args = parser.parse_args()

    date = args.date

    obs_path = f"{OBS_DATA_DIR}/{date.replace('-','')}_verif_eccc_obs.cvs"
    pred_path = f"{PRED_DATA_DIR}/{date.replace('-','')}_regional_state.pkl"

    # TODO:
    # - read both files
    # - get station localisation.
    # - y_pred: Interpolate prediction to station localisation
    # - y_obs: get observation value
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS
    # - compare y_pred and y_obs
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS, ERR_0, ..., ERR_N
    # - save error metric to data/error_data

    # for now: 
    # timestamps: only 1
    # stations: all 
    # error metric: MSE
