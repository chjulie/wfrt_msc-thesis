import argparse
import datetime
import pandas as pd 

from utils import read_pkl
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

    date_str = args.date + "T06:00:00"
    input_date = pd.to_datetime(date_str, format="%Y-%m-%dT%H:%M:%S")
    print(f'date: {type(input_date)}, {input_date}')

    obs_path = f"{OBS_DATA_DIR}/{args.date.replace('-','')}_verif_eccc_obs.csv"
    pred_path = f"{PRED_DATA_DIR}/{args.date.replace('-','')}_regional_state.pkl"

    # TODO:
    # - read both files
    # TODO: add checks that path exist
    date, latitudes, longitudes, fields = read_pkl(pred_path)   # predictions
    # obs_df = pd.read_csv(obs_path, converters={'UTC_DATE': lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")})
    obs_df = pd.read_csv(obs_path, converters={'UTC_DATE': lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S").tz_localize('utc')})
    print(obs_df.shape)
    print(f'date: {type(obs_df.UTC_DATE[200])}, {obs_df.UTC_DATE[200]}')
    print(' ** obs date: ', obs_df.UTC_DATE[200].tzinfo)
    print(' ** pred date: ', date.tzinfo)
    print(obs_df.UTC_DATE[200] - date)
    obs_df = obs_df[obs_df.UTC_DATE == date]
    print(obs_df.shape)

    # - get station localisation.

    # - y_pred: Interpolate prediction to station localisation
    # scipy'y LinearNDInterpolator: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator

    # - y_obs: get observation value
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS
    # - compare y_pred and y_obs
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS, ERR_0, ..., ERR_N
    # - save error metric to data/error_data

    # for now: 
    # timestamps: only 1
    # stations: all 
    # error metric: MSE
    print(" > Program finished successfully !")