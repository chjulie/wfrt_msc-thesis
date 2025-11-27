import pickle
import pandas as pd


def read_pkl(path):

    with open(path, "rb") as f:
        domain_state = pickle.load(f)
        print(f" PRED DATE: {type(domain_state["date"])}, {domain_state["date"]}")
        date = pd.to_datetime(domain_state["date"])  # .tz_localize('utc')
        print(f" PD PRED DATE: {type(date)}, {date}")
        latitudes = domain_state["latitudes"]
        longitudes = domain_state["longitudes"]
        fields = domain_state["fields"]

    return date, latitudes, longitudes, fields
