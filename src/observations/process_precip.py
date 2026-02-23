import numpy as np
import xarray as xr
import pandas as pd
import os
from datetime import datetime

from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

def get_6h_precip(x,y):
        offsets = [pd.Timedelta(hours=i) for i in range(0,6)]
        previous_dates = [x - off for off in  offsets]

        total = 0
        for prev in previous_dates:
            prev_val = precip_df[(precip_df['datetime']==prev.strftime('%Y-%m-%d %H:%M:%S')) & (precip_df['STN_ID']==y)]['PRECIP_AMOUNT'].values 
            print(f" - {prev} : {prev_val}")
            if len(prev_val) != 1:
                return np.nan
            total+=prev_val[0]

        return total

if __name__ == "__main__":
    obs_dir = obs_dir = "/scratch/juchar/eccc_precip_data/"
    precip_df = pd.read_csv(f"{obs_dir}/all_eccc_precip_data.csv")
    precip_df['datetime'] = precip_df['UTC_DATE'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))
    init_hours = [00, 6, 12, 18]
    init_mask = precip_df['datetime'].apply(lambda x: x.hour in init_hours)

    # df_init = precip_df[init_mask]
    # print(' > df_init.head(10) : ', df_init.head(10))

    # df_init['6h_precip'] = df_init.apply(
    #     lambda row: get_6h_precip(row['datetime'], row['STN_ID']),
    #     axis=1
    # )

    # print(' > sum :', df_init['6h_precip'].sum())
    # print(df_init)

    precip_df = precip_df.sort_values(['STN_ID', 'datetime'])
    
    rolled = (
        precip_df
        .set_index('datetime')
        .groupby('STN_ID')['PRECIP_AMOUNT']
        .rolling('6H', closed='right')   # include current hour
    )
    
    sum_vals = rolled.sum().reset_index(level=0, drop=True)
    count_vals = rolled.count().reset_index(level=0, drop=True)

    precip_df['6h_precip'] = sum_vals.to_numpy()

    # Require all 6 hourly observations
    precip_df.loc[count_vals.to_numpy() != 6, '6h_precip'] = np.nan
    print(' > sum :', precip_df['6h_precip'].sum())
    print(precip_df)

    init_hours = [00, 6, 12, 18]
    init_mask = precip_df['datetime'].apply(lambda x: x.hour in init_hours)
    df_init = precip_df[init_mask]

    print(' > sum :', df_init['6h_precip'].sum())
    print(df_init)

    df_init.to_csv(f"{obs_dir}/eccc_6h_precip_data.csv")
