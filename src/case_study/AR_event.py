import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from datetime import datetime, timedelta

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    import matplotlib.tri as tri
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    import warnings
    return (
        Normalize,
        cKDTree,
        ccrs,
        cfeature,
        datetime,
        gpd,
        mo,
        np,
        pd,
        plt,
        tri,
        warnings,
        xr,
    )


@app.cell
def _(plt):
    from matplotlib import rc
    rc("font", **{"family": "serif", "serif": ["Times New Roman"], "size": "12"})
    rc("text", usetex=False)
    rc("lines", linewidth=2)
    plt.rcParams["axes.facecolor"] = "w"
    return


@app.cell
def _(xr):
    data_folder = 'data/AR_data'
    nwp_reg_03 = xr.open_dataset(f"{data_folder}/nwp_reg/AR_wrfout_d02_processed_23120300.nc")
    dl_reg_03 = xr.open_dataset(f"{data_folder}/dl_reg/20231203T00.nc")
    dl_glob_03 = xr.open_dataset(f"{data_folder}/dl_glob/20231203T00.nc")
    return dl_glob_03, dl_reg_03, nwp_reg_03


@app.cell
def _(cKDTree, np):
    def clip_coords(wrf_coords, prediction_coords, tolerance=0.03):
            '''
            prediction_coords = climatex_coords with trim_edges (=> domain contained in climatex coords!)

            '''
            # Compute intersection between prediction_coords and wrf_coords : this yields the evaluation domain
            tree_wrf = cKDTree(wrf_coords)
            indices = tree_wrf.query_ball_point(prediction_coords, r=tolerance)
            prediction_mask = np.array([len(idx) > 0 for idx in indices])

            tree_prediction = cKDTree(prediction_coords)
            indices = tree_prediction.query_ball_point(wrf_coords, r=tolerance)
            wrf_mask = np.array([len(idx) > 0 for idx in indices])

            return prediction_mask, wrf_mask

    def get_domain_coords(wrf_ds, prediction_ds):
            wrf_coords = np.column_stack(
                (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
            )
            prediction_coords = np.column_stack(
                (prediction_ds["longitude"].values, prediction_ds["latitude"].values)
            )
        
            prediction_mask, wrf_mask = clip_coords(wrf_coords, prediction_coords)
            return prediction_mask, wrf_mask 
    return (get_domain_coords,)


@app.cell
def _(ccrs):
    CMAP={
        'tp': 'coolwarm',
        'sp': 'coolwarm',
        'msl': 'coolwarm',
        'z_500': 'coolwarm',
        'z_850': 'coolwarm',
    }
    PROJECTION = ccrs.Stereographic(
        central_latitude=90.0,          # North Pole
        central_longitude=-90.0,        # STAND_LON
        true_scale_latitude=60.0        # Usually TRUELAT2 for polar stereo
    )
    results_folder="reports/plots/case_study"
    return CMAP, PROJECTION, results_folder


@app.cell
def _(pd):
    def utc_to_pst(datetime_str):
        dt_utc = pd.to_datetime(datetime_str, utc=True)
        dt_van = dt_utc.tz_convert("America/Vancouver")
        dt_van = dt_van.floor("H")

        return dt_van.strftime("%Y-%m-%d %H:00:00")
    return (utc_to_pst,)


@app.cell
def _(
    CMAP,
    Normalize,
    PROJECTION,
    cKDTree,
    ccrs,
    cfeature,
    datetime,
    gpd,
    np,
    pd,
    tri,
    warnings,
):
    def fix(lons):
        # Shift the longitudes from 0-360 to -180-180
        return np.where(lons > 180, lons - 360, lons)

    def clip_global(global_coords, climatex_coords, tolerance=0.03):
        tree = cKDTree(climatex_coords)
        indices = tree.query_ball_point(global_coords, r=tolerance)
        mask = np.array([len(idx) > 0 for idx in indices])

        return mask

    def xtime(a, b):
        return datetime.strftime(
            a + pd.Timedelta(hours=int(b)), format="%Y-%m-%dT%H:00:00.000000000"
        )

    def plot_basemap(ax):
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in create_collection",
            category=RuntimeWarning,
        )

        ax.add_feature(cfeature.OCEAN, facecolor="#a6bddb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#D4DFED", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=2)


    def plot_data(ax, ds, field, time, norm, cbar_title, mask):

        if 'time' in ds.dims:
            if field=='tp':
                data = 1000*ds[field].sel(time=time).values # m to mm conversion
            elif field=='msl':
                data = 1e-3*ds[field].sel(time=time).values # Pa to HPa conversion
            else:
                data = ds[field].sel(time=time).values
            lons=ds.longitude[mask]
            lats=ds.latitude[mask]
            proj=ccrs.PlateCarree()
        elif 'XTIME' in ds.dims:
            if "_" in field:
                var, level = field.split("_")
                var = "geopotential" if var == "z" else var
                data = ds[var].sel(XTIME=time, air_pressure=int(level)).values
            elif field=='msl':
                data = 1e-3*ds[field].sel(XTIME=time).values
            else:
                data = ds[field].sel(XTIME=time).values
            lons=ds.XLONG.values.flatten()[mask]
            lats=ds.XLAT.values.flatten()[mask]
            proj=PROJECTION
            data = data.flatten()

        data = data.squeeze()[...,mask]
    
        try:
            geom = gpd.points_from_xy(lons, lats, crs=proj)
            gdf = gpd.GeoDataFrame({"data": data}, geometry=geom)

        except ValueError:
            geom = gpd.points_from_xy(
                lons.values.ravel(),
                lats.values.ravel(),
                crs=ccrs.PlateCarree(),
            )
            gdf = gpd.GeoDataFrame({"data": data.ravel()}, geometry=geom)

        gdf.plot(
            ax=ax,
            column='data',
            markersize=10,
            cmap=CMAP[field],
            legend=True,
            legend_kwds={"shrink": 0.4},
            transform=ccrs.PlateCarree(),
            zorder=1,
            norm=norm,
            )

        cbar = ax.get_figure().axes[-1]  # last axis is usually the colorbar
        cbar.set_ylabel(cbar_title, fontsize=9)
        cbar.tick_params(labelsize=8)

    def plot_global_data(fig, ax, ds, mask, field, time, norm, cbar_title):

        triangulation = tri.Triangulation(ds.longitude, ds.latitude)

        triangle_mask = ~np.all(mask[triangulation.triangles], axis=1)
        triangulation.set_mask(triangle_mask)

        data = ds[field].sel(time=time).values
        data = np.ma.masked_where(~mask, data)
        data = 1000 * data # m to mm conversion

        contour = ax.tricontourf(
            triangulation,
            data,
            levels=20,
            cmap="coolwarm",
            norm=norm,
            transform=ccrs.PlateCarree()
        )
        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.4, label=cbar_title)

        # restrict map extent to valid region
        valid_lon = ds.longitude.values[mask]
        valid_lat = ds.latitude.values[mask]

        ax.set_extent([
            valid_lon.min() + 15, 
            valid_lon.max() - 12,
            valid_lat.min() + 2.2, 
            valid_lat.max() - 1.4,
            ],
            crs=ccrs.PlateCarree()
        ) # works !!


    def get_norm(all_ds, initial_date, field, lead_times):
        vmins = []
        vmaxs = []

        for ds in all_ds:
            for lt in lead_times:
                t = xtime(initial_date, lt)
                try:
                    if 'time' in ds.dims:
                        if field=='tp':
                            data = 1000*ds[field].sel(time=t).values # m to mm conversion
                        elif field=='msl':
                            data = 1e-3*ds[field].sel(time=t).values # Pa to HPa conversion
                        else:
                            data = ds[field].sel(time=t).values
                    elif 'XTIME' in ds.dims:
                        if "_" in field:
                            var, level = field.split("_")
                            var = "geopotential" if var == "z" else var
                            data = ds[var].sel(XTIME=t, air_pressure=int(level)).values
                        elif field=='msl':
                            data = 1e-3*ds[field].sel(XTIME=t).values
                        else:
                            data = ds[field].sel(XTIME=t).values
                except KeyError as e:
                    print(f" [Warning] lead time {lt} not found: {e}")
                    continue
                vmins.append(np.min(data))
                vmaxs.append(np.max(data))

        vmin, vmax = np.nanmin(vmins), np.nanmax(vmaxs)
        print(' - vmin : ', vmin)
        print(' - vmax : ', vmax)

        norm = Normalize(vmin=vmin, vmax=vmax)

        return norm
    return (
        clip_global,
        fix,
        get_norm,
        plot_basemap,
        plot_data,
        plot_global_data,
        xtime,
    )


@app.cell
def _(clip_global, dl_glob_03, dl_reg_03, fix, np):
    global_coords = np.column_stack((fix(dl_glob_03.longitude.values), dl_glob_03.latitude.values))
    climatex_coords = np.column_stack((dl_reg_03.longitude.values, dl_reg_03.latitude.values))
    mask = clip_global(global_coords, climatex_coords)
    return (mask,)


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Plot tp from dec. 03 to dec. 06 for all models
    """)
    return


@app.cell
def _(dl_reg_03, get_domain_coords, nwp_reg_03):
    prediction_mask, wrf_mask = get_domain_coords(nwp_reg_03, dl_reg_03)
    return prediction_mask, wrf_mask


@app.cell
def _(
    PROJECTION,
    datetime,
    dl_reg_03,
    get_norm,
    nwp_reg_03,
    prediction_mask,
    wrf_mask,
):
    field = 'z_850'
    cbar_title = {
        '2t': '2t [Kelvin]',
        '10u': '10u [m/s]',
        '10v': '10v [m/s]',
        'msl': 'msl [HPa]',
        'z_50': 'Geopotential [HPa]',
        'z_500': 'Geopotential [HPa]',
        'z_850': 'Geopotential [HPa]',
        'tp': 'tp [mm]',

    }
    ds_all = {
        'nwp_reg': {
            'ds': nwp_reg_03, 
            'projection': PROJECTION,
            'mask': wrf_mask,
        },
        'dl_reg': {
            'ds': dl_reg_03, 
            'projection': PROJECTION,
            'mask': prediction_mask,
        },
        #'dl_glob' : {
        #    'ds': dl_glob_03,
        #    'projection': PROJECTION,
        #},
    }

    lead_times = [36, 42, 48, 54, 60, 66, 72] # 12, 18, 24, 30, 36, 
    initial_date = datetime(2023,12,3,00)
    norm = get_norm([ds_all[m]['ds'] for m in ds_all.keys()], initial_date, field, lead_times)
    return cbar_title, ds_all, field, initial_date, lead_times, norm


@app.cell
def _(
    cbar_title,
    datetime,
    ds_all,
    field,
    initial_date,
    lead_times,
    mask,
    norm,
    plot_basemap,
    plot_data,
    plot_global_data,
    plt,
    results_folder,
    utc_to_pst,
    xtime,
):
    for model_name, ds_dict in ds_all.items():
        n = len(lead_times)
        fig, ax = plt.subplots(1, n, figsize=(3*n,5), subplot_kw={'projection': ds_all[model_name]['projection']})
        ax = ax.ravel()

        for i, lt in enumerate(lead_times):
            print(model_name, lt)
            t = xtime(initial_date, lt)
            colorbar_title = cbar_title.get(field)
            plot_basemap(ax[i])

            if 'glob' in model_name:
                plot_global_data(
                    fig=fig, 
                    ax=ax[i],
                    ds=ds_dict['ds'],
                    mask=mask,
                    field=field,
                    time=t,
                    norm=norm,
                    cbar_title=colorbar_title,
                )
            else:
                # TODO
                plot_data(ax[i], ds_dict['ds'], field, t, norm, colorbar_title,mask=ds_dict['mask'])

            ax[i].set_title(f"{utc_to_pst(t)}", fontsize=10) # TODO: change UTC to PST
        plt.savefig(f"{results_folder}/{field}-{model_name}-{datetime.strftime(initial_date, '%Y%m%d%H')}.png", bbox_inches='tight')
    plt.show()
    return (t,)


@app.cell
def _(nwp_reg_03):
    nwp_reg_03
    return


@app.cell
def _(dl_glob_03, field, t):
    glob_data = 1000 * dl_glob_03[field].sel(time=t).values
    glob_data.max()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot distribution of precipitation amount
    """)
    return


@app.cell
def _(
    datetime,
    dl_reg_03,
    initial_date,
    nwp_reg_03,
    plt,
    prediction_mask,
    results_folder,
    utc_to_pst,
    wrf_mask,
    xtime,
):
    # 1. Choose one lead time
    # 2. Plot histogram

    fig_tp, ax_tp = plt.subplots(figsize=(10,5))

    data_nwp = nwp_reg_03['tp'].sel(XTIME=xtime(initial_date,54)).values.flatten()
    data_nwp = data_nwp.squeeze()[wrf_mask]
    data_dl = 1000*dl_reg_03['tp'].sel(time=xtime(initial_date,54)).values
    data_dl = data_dl.squeeze()[prediction_mask]

    ax_tp.hist(data_nwp, bins=200, alpha=0.5, color='blue', log=True, label='nwp_reg')
    ax_tp.hist(data_dl, bins=200, alpha=0.5, color='darkslategrey', log=True, label='dl_reg')

    ax_tp.legend(loc='upper right')
    ax_tp.set_ylabel('Number of forecasts')
    ax_tp.set_xlabel('6h accumulated precipitation amount [mm]')
    ax_tp.set_title(f"6h accumulated precipitation amount distribution\n for valid time {utc_to_pst(xtime(initial_date,54))} PDT")
    plt.savefig(f"{results_folder}/distribution-tp-{datetime.strftime(initial_date, '%Y%m%d%H')}-lt54.png", bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(
    dl_reg_03,
    initial_date,
    nwp_reg_03,
    plt,
    prediction_mask,
    utc_to_pst,
    wrf_mask,
    xtime,
):
    def _():
        # 1. Choose one lead time
        # 2. Plot histogram

        fig_tp, ax_tp = plt.subplots(1, 2, figsize=(12,4))

        for j, lt_tp in enumerate([54,60]):
            data_nwp = nwp_reg_03['tp'].sel(XTIME=xtime(initial_date,lt_tp)).values.flatten()
            data_nwp = data_nwp.squeeze()[wrf_mask]
            data_dl = 1000*dl_reg_03['tp'].sel(time=xtime(initial_date,lt_tp)).values
            data_dl = data_dl.squeeze()[prediction_mask]
        
            ax_tp[j].hist(data_nwp, bins=200, alpha=0.5, color='blue', log=True, label='nwp_reg')
            ax_tp[j].hist(data_dl, bins=200, alpha=0.5, color='darkslategrey', log=True, label='dl_reg')
        
            ax_tp[j].legend(loc='upper right')
            ax_tp[j].set_ylabel('Number of forecasts')
            ax_tp[j].set_xlabel('6h accumulated precipitation amount [mm]')
            ax_tp[j].set_title(f"6h accumulated precipitation amount distribution\n for valid time {utc_to_pst(xtime(initial_date,lt_tp))} PDT")
        return plt.show()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
