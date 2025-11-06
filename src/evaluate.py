import argparse
from data_constants import OBS_DATA_DIR

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
    pred_path = f""
