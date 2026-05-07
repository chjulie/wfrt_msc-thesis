import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import warnings
    from datetime import datetime, timedelta
    import calendar
    return calendar, datetime, mo, np, pd, plt, rc, timedelta


@app.cell
def _(plt, rc):
    rc("font", **{"family": "serif", "serif": ["Times New Roman"], "size": "14"})
    rc("text", usetex=False)
    rc("lines", linewidth=2)
    plt.rcParams["axes.facecolor"] = "w"
    plt.rcParams['axes.grid'] = True 
    plt.rcParams["grid.linewidth"] = 0.2 

    temp_color = "#CA1634"
    precip_color = "#0B84AD"
    wind_color = "#FFC247"#"#CABC53"

    color_dict = {
        '2t': "#CA1634",
        'tp': "#0B84AD",
        '10ff': "#FFC247",
        'dl_reg': '#466EC3', #'#219ebc', #"#A61166",
        'dl_transfer': '#FFBA49', #'#8ecae6',
        'nwp_reg': '#CF4526', #'#023047',
        'dl_reg_gauss': '#219ebc',
    }

    figure_folder = 'reports/plots/evaluation/'
    return color_dict, figure_folder


@app.cell
def _(pd):
    folder = 'data/error_data/'
    #dl_glob_raw = pd.read_csv(f"{folder}errors-global-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    dl_reg_raw = pd.read_csv(f"{folder}errors-stage-c-nn_scipy_resampling-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    #dl_reg_gauss = pd.read_csv(f"{folder}errors-stage-c-gauss_resampling-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    nwp_reg_raw = pd.read_csv(f"{folder}errors-nwp_reg-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    bris_raw = pd.read_csv(f"{folder}errors-bris-20230101_20231231.csv").drop(columns=['Unnamed: 0'])
    return bris_raw, dl_reg_raw, nwp_reg_raw


@app.cell
def _(nwp_reg_raw):
    nwp_reg_raw 

    return


@app.cell
def _(nwp_reg_raw):
    print(len(nwp_reg_raw) / 52 / 3)
    return


@app.cell
def _(dl_reg_raw, nwp_reg_raw):
    nwp_reg_raw['initial_date'] = nwp_reg_raw['initial_date'].apply(lambda x : x + ' 00:00:00')
    nwp_reg_raw['valid_time'] = nwp_reg_raw.apply(lambda row : row['initial_date'] + ' ' + str(row['lead_time']), axis=1)
    print('nwp')
    dl_reg_raw['initial_date'] = dl_reg_raw['initial_date'].apply(lambda x : x + ' 00:00:00' if len(x) < 11 else x)
    dl_reg_raw['valid_time'] = dl_reg_raw.apply(lambda row : row['initial_date'] + ' ' + str(row['lead_time']), axis=1)
    print('dl')
    return


@app.cell
def _(bris_raw):
    bris_raw['initial_date'] = bris_raw['initial_date'].apply(lambda x : x + ' 00:00:00' if len(x) < 11 else x)
    bris_raw['valid_time'] = bris_raw.apply(lambda row : row['initial_date'] + ' ' + str(row['lead_time']), axis=1)
    return


@app.cell
def _(dl_reg_raw, nwp_reg_raw):
    print(len(dl_reg_raw.valid_time.unique()))
    print(len(nwp_reg_raw.valid_time.unique()))
    return


@app.cell
def _():
    #unique_valid_time_slow = [x for x in nwp_reg_raw.valid_time.unique() if  x in dl_reg_raw.valid_time.unique()]
    #print(len(unique_valid_time_slow))
    return


@app.cell
def _(dl_reg_raw, np, nwp_reg_raw):
    #unique_valid_time = [x for x in dl_reg_raw.valid_time.unique() if  x in nwp_reg_raw.valid_time.unique()]
    unique_valid_time = np.intersect1d(
        dl_reg_raw.valid_time.unique(),
        nwp_reg_raw.valid_time.unique(),
        assume_unique=True
    )
    print(len(unique_valid_time))
    return (unique_valid_time,)


@app.cell
def _(unique_valid_time):
    print(sorted(unique_valid_time))
    return


@app.cell
def _(bris_raw, dl_reg_raw, nwp_reg_raw, unique_valid_time):
    dl_reg_mask = dl_reg_raw.valid_time.isin(unique_valid_time)
    nwp_reg_mask = nwp_reg_raw.valid_time.isin(unique_valid_time)
    bris_mask = bris_raw.valid_time.isin(unique_valid_time)
    return bris_mask, dl_reg_mask, nwp_reg_mask


@app.cell
def _(bris_mask, bris_raw, dl_reg_mask, dl_reg_raw, nwp_reg_mask, nwp_reg_raw):
    dl_reg = dl_reg_raw[dl_reg_mask]
    nwp_reg = nwp_reg_raw[nwp_reg_mask]
    bris = bris_raw[bris_mask]

    print(len(dl_reg))
    print(len(nwp_reg))
    print(len(bris))
    return bris, dl_reg, nwp_reg


@app.cell
def _():
    #dl_glob_bch = dl_glob[~pd.to_numeric(dl_glob["station_id"], errors="coerce").notna()]
    #dl_reg_bch = dl_reg[~pd.to_numeric(dl_reg["station_id"], errors="coerce").notna()]
    #nwp_reg_bch = nwp_reg[~pd.to_numeric(nwp_reg["station_id"], errors="coerce").notna()]
    return


@app.cell
def _():
    #dl_glob_eccc = dl_glob[pd.to_numeric(dl_glob["station_id"], errors="coerce").notna()]
    #dl_reg_eccc = dl_reg[pd.to_numeric(dl_reg["station_id"], errors="coerce").notna()]
    #nwp_reg_eccc = nwp_reg[pd.to_numeric(nwp_reg["station_id"], errors="coerce").notna()]
    return


@app.cell
def _(mo):
    mo.md(r"""
    # RMSE vs lead time for temperature and wind
    """)
    return


@app.cell
def _(np):
    def process_precip_df(model, dl_df, rain_threshold=1.2):
        #init_hour_mask = dl_df.initial_date.apply(lambda x: x[-8:] == "00:00:00")
        precip = dl_df[(dl_df.field == 'tp')] # & init_hour_mask]
        if 'dl_' in model:
            print('dl model!')
            precip.loc[:,'pred_value'] = 1000*precip.loc[:,'pred_value'] # m to mm conversion
        precip.loc[:,'rmse'] = np.sqrt(np.power(precip.loc[:,'pred_value'] - precip.loc[:,'obs_value'], 2)) # correct rmse

        precip['binary_obs'] = precip.loc[:,'obs_value'] > rain_threshold
        precip['binary_pred'] = precip.loc[:,'pred_value'] > rain_threshold

        return precip
    return (process_precip_df,)


@app.function
def process_temp_and_wind(df, field):
    df = df[(df.field==field)]
    #df['nrmse'] = df.loc[:,'rmse'] / df.loc[:,'pred_value'].mean()
    grouped_df = df.groupby('lead_time')['rmse'].agg('mean')
    grouped_df = grouped_df[grouped_df.index!=0]
    return grouped_df


@app.cell
def _(dl_reg):
    dl_reg_temp = dl_reg[dl_reg.field=='2t'].groupby('lead_time')['rmse'].agg(['size', 'mean'])
    dl_reg_temp
    return


@app.cell
def _(nwp_reg):
    nwp_reg_temp = nwp_reg[nwp_reg.field=='2t'].groupby('lead_time')['rmse'].agg(['size', 'mean'])
    nwp_reg_temp
    return


@app.cell
def _(bris, color_dict, dl_reg, nwp_reg, process_precip_df):
    lead_times = nwp_reg.lead_time.unique()
    rmse_fields = ['2t', '10ff']
    field_units = {
        '2t': "°C",
        '10ff': 'm/s'
    }
    field_title = {
        '2t': '2 m temperature',
        '10ff': '10 m wind speed',
    }
    models = {'nwp_reg': 
                 {
                    'color': color_dict['nwp_reg'],
                    'name': 'nwp_reg',
                    'data_2t': process_temp_and_wind(nwp_reg, '2t'),
                    'data_10ff': process_temp_and_wind(nwp_reg, '10ff'),
                    'data_tp': process_precip_df('nwp_reg', nwp_reg),
                 },
              'dl_reg':    
                 {
                    'color': color_dict['dl_reg'],
                    'name': 'dl_reg',
                    'data_2t': process_temp_and_wind(dl_reg, '2t'),
                    'data_10ff': process_temp_and_wind(dl_reg, '10ff'),
                    'data_tp': process_precip_df('dl_reg', dl_reg),
                 },
              'dl_transfer':    
                 {
                    'color': color_dict['dl_transfer'],
                    'name': 'dl_transfer',
                    'data_2t': process_temp_and_wind(bris, '2t'),
                    'data_10ff': process_temp_and_wind(bris, '10ff'),
                    'data_tp': process_precip_df('dl_transfer', bris),
                 },

            #'dl_glob':
            #     {
            #        'color': color_dict['dl_glob'] ,
            #        'name': 'dl_glob',
            #        'data_2t': process_temp_and_wind(dl_glob, '2t'),
            #        'data_10ff': process_temp_and_wind(dl_glob, '10ff'),
            #        'data_tp': process_precip_df('dl_glob', dl_glob),
            #     },
             }
    return field_title, field_units, lead_times, models, rmse_fields


@app.cell
def _(lead_times):
    print(lead_times)
    return


@app.cell
def _(field_title, field_units, lead_times, models, plt, rmse_fields):
    for field in rmse_fields:
        fig, ax = plt.subplots(figsize=(8,5))

        for model, mdict in models.items():
            #if model=='dl_glob':
            #    continue
            d = mdict[f"data_{field}"]
            ax.plot(d.index, d, '.-', color=mdict['color'], label=mdict['name'])


        # TODO
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel('Lead time [h]')
        ax.set_ylabel(f"RMSE [{field_units[field]}]")
        ax.set_title(field_title[field])

        ax.set_xlim(0, lead_times[-1]+6)
        ax.set_xticks(lead_times)

        #plt.savefig(f"{figure_folder}rmse_vs_lt-test-{field}.png", bbox_inches='tight')
        #plt.savefig(f"{figure_folder}rmse_vs_lt-test-{field}-bch.svg", format='svg', bbox_inches='tight')

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Precipitation analysis
    ⚠️ Also for NWP
    1. Equitable threat score
    2. Q-Q plot
    """)
    return


@app.function
def compute_ETS(df, lt):
    df = df[df.lead_time==lt]
    a = df[(df.binary_obs==True) & (df.binary_pred==True)].size
    b = df[(df.binary_obs==False) & (df.binary_pred==True)].size
    c = df[(df.binary_obs==True) & (df.binary_pred==False)].size
    d = df[(df.binary_obs==False) & (df.binary_pred==False)].size
    n = df.size

    print(' - a : ', a)
    print(' - b : ', b)
    print(' - c : ', c)
    print(' - d : ', d)

    a_r = (a+b) * (a + c) / n

    ETS = (a - a_r) / (a - a_r + b + c)
    return ETS


@app.cell
def _(nwp_reg):
    lead_times_ETS = nwp_reg.lead_time.unique()[1:]
    print(lead_times_ETS)
    return (lead_times_ETS,)


@app.cell
def _(lead_times_ETS, models, plt):
    def _():
        # 1. ETS vs lead_time

        fig, ax = plt.subplots(figsize=(8,5))

        for model, mdict in models.items():
            ETS_list = [compute_ETS(mdict['data_tp'], lt) for lt in lead_times_ETS]
            ax.plot(lead_times_ETS, ETS_list, '.-', color=mdict['color'], label=mdict['name'])

        # TODO
        ax.legend(fontsize=10)
        ax.set_xlabel('Lead time [h]')
        ax.set_ylabel('ETS')
        ax.set_title('6h accumulated precipitation')

        ax.set_xlim(0, lead_times_ETS[-1]+6)
        ax.set_xticks(lead_times_ETS)
        #plt.savefig(f"{figure_folder}ETS_vs_lt-test-nwp-tp.svg", format='svg', bbox_inches='tight')
        return plt.show()


    _()
    return


@app.cell
def _(models, plt):
    def _():
        # 2. Q-Q Plot

        fig, ax = plt.subplots(figsize=(8,5))
        count = 0

        for model, mdict in models.items():
            # drop na
            qq_df = mdict['data_tp'].dropna(axis=0)
            obs = sorted(qq_df.obs_value)
            if count==0:
                ax.plot(obs, obs, '--', color='grey')

            forecasts = sorted(qq_df.pred_value)
            print(' -max forecast : ', forecasts[-1])
            ax.plot(obs, forecasts, color=mdict['color'], label=mdict['name'])
            count += 1

        # TODO
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylabel('Sorted forecasts [mm]')
        ax.set_xlabel('Sorted observations [mm]')
        ax.set_title('6h accumulated precipitation')
        #plt.savefig(f"{figure_folder}qq_plot-test-nwp-tp.svg", format='svg', bbox_inches='tight')

        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot time series for 1 station
        - observations
        - prediction
    """)
    return


@app.cell
def _(bris, datetime, dl_reg, nwp_reg, timedelta):
    def get_station_df(model_name, station_id, month):
        if model_name == 'nwp_reg':
            station_df = nwp_reg[(nwp_reg.station_id==station_id) & (nwp_reg.field=='2t')]
        elif model_name == 'dl_reg':
            station_df = dl_reg[(dl_reg.station_id==station_id) & (dl_reg.field=='2t') & (dl_reg.lead_time!=0)]
        elif model_name == 'dl_transfer':
            station_df = bris[(bris.station_id==station_id) & (bris.field=='2t') & (bris.lead_time!=0)]
        else:
            raise NotImplementedError

        month_mask = station_df.initial_date.apply(lambda x : x[5:7]==month)
        station_df = station_df[month_mask]
        station_df['initial_date'] = station_df['initial_date'].apply(lambda x: x[:10])
        station_df['xtime'] = station_df.apply(lambda row: datetime.strptime(row.initial_date, '%Y-%m-%d') + timedelta(hours=int(row['lead_time'])), axis=1)

        return station_df
    return (get_station_df,)


@app.cell
def _(get_station_df):
    station_id = 'YRV'
    field_ts = '2t'
    month = '12'
    model_ts = 'nwp_reg'

    station_df = get_station_df(model_ts, station_id, month)
    station_df
    return field_ts, model_ts, month, station_df, station_id


@app.cell
def _(calendar, color_dict, model_ts, month, plt, station_df, station_id):
    stn_obs_df = station_df.sort_values(by='xtime')

    fig_ts, ax_ts = plt.subplots(figsize=(20,5))

    # plot obs
    ax_ts.plot(stn_obs_df.xtime.values, stn_obs_df.obs_value - 273.15,'.-', color='#B8B8B8', linewidth=3, label='Observations')

    # plot forecasts
    c=0
    for init_date in station_df.initial_date.unique() : 
        current_df = station_df[station_df.initial_date==init_date] 
        #print(current_df.pred_value.size)
        if (current_df.pred_value.size==14):
            if c==0:
                ax_ts.plot(current_df.xtime, current_df['pred_value']-273.15, '.-', color=color_dict[model_ts], linewidth=1, label=f"Forecasts, {model_ts}")
            else:
                ax_ts.plot(current_df.xtime, current_df['pred_value']-273.15, '.-', color=color_dict[model_ts], linewidth=1)
            c+=1

    ax_ts.set_title(f"Temperature observations and forecasts for station {station_id}, {calendar.month_name[int(month)]}")
    ax_ts.set_ylabel(" Temperature [°C]")
    ax_ts.legend(loc='upper right')
    plt.xticks(fontsize=8, rotation=45)
    plt.savefig(f"reports/plots/obs_time_series/{model_ts}_{station_id}-month_{month}.png")

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot mean rmse for each station
    """)
    return


@app.cell
def _(dl_reg_gauss):
    print(len(dl_reg_gauss.station_id.unique()))
    print(dl_reg_gauss.station_id.unique())
    return


@app.cell
def _(dl_reg_gauss, pd):
    dl_reg_gauss_bch = dl_reg_gauss[~pd.to_numeric(dl_reg_gauss["station_id"], errors="coerce").notna()]
    return


@app.cell
def _(dl_reg_gauss, field_ts, figure_folder, lead_times, plt):
    cnt = 0
    for stn_id in dl_reg_gauss.station_id.unique():
        print(cnt, stn_id)
        df = dl_reg_gauss[(dl_reg_gauss.field==field_ts) & (dl_reg_gauss.station_id==stn_id)]
        print(df.size)
        grouped_df = df.groupby('lead_time')['rmse'].agg('mean')

        fig_stn, ax_stn = plt.subplots()
        ax_stn.plot(grouped_df.index, grouped_df, '.-')
        ax_stn.set_title(f"Station {stn_id}")

        ax_stn.set_xlabel('Lead time [h]')
        ax_stn.set_ylabel(f"RMSE [°C]")

        ax_stn.set_xlim(0, lead_times[-1]+6)
        ax_stn.set_xticks(lead_times)
        plt.savefig(f"{figure_folder}/stn-{stn_id}-rmse.png")
        plt.close()

        cnt += 1
    return


@app.cell
def _(mo):
    mo.md(r"""
    # RMSE vs lead time per season
    """)
    return


@app.cell
def _(bris, dl_reg, nwp_reg):
    def get_season_df(model_name, field, season):
        if model_name == 'nwp_reg':
            season_df = nwp_reg[(nwp_reg.field==field)]
        elif model_name == 'dl_reg':
            season_df = dl_reg[(dl_reg.field==field) & (dl_reg.lead_time!=0)]
        elif model_name == 'dl_transfer':
            season_df = bris[(bris.field==field) & (bris.lead_time!=0)]
        else:
            raise NotImplementedError
        season_months = {
            'Spring' : ['03', '04', '05'],
            'Summer': ['06', '07', '08'],
            'Fall': ['09', '10', '11'],
            'Winter': ['12', '01', '02']  
        }
        season_mask = season_df.initial_date.apply(lambda x : x[5:7] in season_months[season])
        season_df = season_df[season_mask]
        grouped_season_df = season_df.groupby('lead_time')['rmse'].agg('mean')

        return grouped_season_df
    return (get_season_df,)


@app.cell
def _():
    season_field = '2t'
    season = 'Winter'
    return season, season_field


@app.cell
def _(
    field_title,
    field_units,
    figure_folder,
    get_season_df,
    lead_times,
    models,
    plt,
    rmse_fields,
    season,
    season_field,
):
    def _():
        for field in rmse_fields:
            fig, ax = plt.subplots(figsize=(8,5))

            for model, mdict in models.items():
                if model=='dl_transfer':
                    continue
                d = get_season_df(model, season_field, season)
                ax.plot(d.index, d, '.-', color=mdict['color'], label=mdict['name'])


            # TODO
            ax.legend(loc='upper left', fontsize=10)
            ax.set_xlabel('Lead time [h]')
            ax.set_ylabel(f"RMSE [{field_units[field]}]")
            ax.set_title(f"RMSE vs lead time for {field_title[field]} \n averaged over {season}", fontsize=12)

            ax.set_xlim(0, lead_times[-1]+6)
            ax.set_xticks(lead_times)

            plt.savefig(f"{figure_folder}rmse_vs_lt-{season}-test-{field}.png", bbox_inches='tight')
            #plt.savefig(f"{figure_folder}rmse_vs_lt-test-{field}-bch.svg", format='svg', bbox_inches='tight')
        return plt.show()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
