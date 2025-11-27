import pickle
import pandas as pd


def read_pkl(path):

    with open(path, "rb") as f:
        domain_state = pickle.load(f)
        date = pd.to_datetime(domain_state["date"]).tz_localize("utc")
        latitudes = domain_state["latitudes"]
        longitudes = domain_state["longitudes"]
        fields = domain_state["fields"]

    return date, latitudes, longitudes, fields
