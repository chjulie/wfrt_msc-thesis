from abc import ABC, abstractmethod
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime, timedelta
from pyresample import geometry, bilinear, kd_tree
import os

from utils.data_constants import PRED_DATA_DIR, ERROR_DATA_DIR, OBS_EVAL_FIELDS, FIR_SCRATCH, EVAL_LEAD_TIMES, OLIVIA_GROUP_SCRATCH

def model_evaluator_factory(
    model_name: str,
    date_range,
    lead_times: list[str],
    system: str,
):  

    if model_name == "dl_reg":
        evaluator = DLModelEvaluator(
            date_range=date_range,
            lead_times=lead_times,
            prediction_data_path='/cluster/projects/nn10090k/results/juchar/climatex-lam-inference-20220701T00-20220715T18.nc',
            system=system,
        )
    elif model_name=="nwp_reg":
        evaluator = RegNWPModelEvaluator(
            date_range=date_range,
            lead_times=lead_times,
            system=system,
        )
    elif model_name=="dl_glob":
        evaluator = DLModelEvaluator(
            date_range=date_range,
            lead_times=lead_times,
            prediction_data_path="/cluster/projects/nn10090k/results/juchar/global-lam-inference-20230101T00-20230115T12.nc",
            system=system,
        )
    else:
        raise NotImplementedError

    return evaluator

class ModelEvaluator(ABC):
    def __init__(self, 
                date_range,
                lead_times: list[str],
                model_name: str,
                system: str
                ):
        self.system = system

        self.date_range = date_range
        self.lead_times = lead_times
        self.model_name = model_name
        self.error_df = pd.DataFrame(columns=["model", "initial_date", "lead_time", "station_id", "field", "rmse"])
        
        self.counter = 0
        self.coords = None
        self.stations_coords = None
        self.current_initial_date = None
        self.current_lead_time = None
        self.current_field = None
        self.model_resampled_data = None
        self.current_obs_data_df = None
    
    @property
    def current_datetime(self):
        return self.current_initial_date + timedelta(hours=int(self.current_lead_time))

    @abstractmethod
    def compute_coordinates(self):
        '''
        Processe the model coordinates to get them in 
        '''
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        '''
        Loop over the date range and the lead time to populate self.error_df
        '''
        raise NotImplementedError

    def get_station_observation(self, obs_folder_path):
        obs_file_path = f"{obs_folder_path}/clipped_eccc_{self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        obs_df = pd.read_csv(
            obs_file_path,
            converters={
                "UTC_DATE": lambda x: pd.to_datetime(
                    x, format="%Y-%m-%d %H:%M:%S+00:00"
                ).tz_localize("utc")
            },
        ).drop(columns=['Unnamed: 0'])

        # Get station localisation
        s_longitudes = obs_df.x.values 
        s_latitudes = obs_df.y.values
        self.stations_coords = np.column_stack((s_longitudes, s_latitudes))

        # Get station data
        self.current_obs_data_df = obs_df[obs_df.UTC_DATE == self.current_datetime.tz_localize("utc")]   # TODO: check that this work (or add localization)
    
    def rmse(self, field: str):
        field_obs = OBS_EVAL_FIELDS.get(field)

        if field == '2t':
            # TODO : convert obs to kelvin 
            observations = self.current_obs_data_df[field_obs] + 273.15     # convert to Kelvin
        else :
            observations = self.current_obs_data_df[field_obs]

        assert self.model_resampled_data.shape == observations.shape, "self.model_resampled_data and self.obs_data must have the same shape"
        rmse = np.sqrt(np.power(self.model_resampled_data-observations, 2)) 
    
        df = pd.DataFrame(data={
            "model": self.model_name,
            "initial_date": self.current_initial_date,
            "lead_time": self.current_lead_time,
            "station_id": self.current_obs_data_df['STN_ID'],
            "x": self.current_obs_data_df['x'],
            "y": self.current_obs_data_df['y'],
            "field": field,
            "rmse": rmse,
        })

        # append to existing error df
        self.error_df = pd.concat((self.error_df, df), axis=0)

    def get_prediction_at_station_loc(self, data: np.ndarray | xr.DataArray):
        '''
        Get the forecast value at the stations location
        (equivalent to utils.resampling_utils.pyresample_resampling())
        '''
        src_grid = geometry.SwathDefinition(lons=self.coords[:, 0], lats=self.coords[:, 1])
        tgt_grid = geometry.SwathDefinition(lons=self.stations_coords[:, 0], lats=self.stations_coords[:, 1])
        # TODO: convert data to data.values only if data : xr.DataArray
        if hasattr(data, 'values'):
            data = data.values

        resampled_data = kd_tree.resample_nearest(
            source_geo_def=src_grid,
            data=data,
            target_geo_def=tgt_grid,
            radius_of_influence=50000,
        )
        self.model_resampled_data = resampled_data

    def save_error_df(self):
        if self.system == 'Fir':
            self.error_df.to_csv(f"{ERROR_DATA_DIR}/errors-{self.model_name}-{self.date_range[0].strftime('%Y%m%d')}_{self.date_range[-1].strftime('%Y%m%d')}.csv")
        elif self.system == 'Olivia':
            self.error_df.to_csv(f"{OLIVIA_GROUP_SCRATCH}/results/juchar/errors-{self.model_name}-{self.date_range[0].strftime('%Y%m%d')}_{self.date_range[-1].strftime('%Y%m%d')}.csv")
        else:
            raise NotImplementedError


class RegNWPModelEvaluator(ModelEvaluator):
    def __init__(self, 
                date_range,
                lead_times: list[str],
                system: str,
                ):
        super().__init__( 
                date_range=date_range, 
                lead_times=lead_times,
                model_name='nwp_reg',
                system=system,
                )

        self.run_id = None
        self.model_id = 'WAC00WG-01'
        self.local_folder_path = f"{FIR_SCRATCH}/wrf_data"

        self.ds = None

    @property
    def file_name(self):
        return f"wrfout_d02_processed_{self.run_id}.nc"

    def compute_coordinates(self):
        self.coords = np.column_stack((self.ds.XLONG.values.flatten(), self.ds.XLAT.values.flatten()))

    def rclone_copy(self):
        source = f"wfrt-nextcloud:Documents/WRF-forecasts/{self.model_id}/wrfout_d02_processed_{self.run_id}.nc"
        # cmd = f"rclone copy '{source}' '{self.local_folder_path}'"# --progress"

        cmd = [
            "rclone", "copy",
            source,
            self.local_folder_path,
            "--checksum"
        ]
        subprocess.run(cmd, capture_output=True, check=True, text=True)

    @staticmethod
    def xtime(a, b):
        return datetime.strftime(a + pd.Timedelta(hours=int(b)), format='%Y-%m-%dT%H:00:00.000000000')

    def evaluate(self):

        for initial_date in self.date_range:
            print(f"⚡️ date: {initial_date}", flush=True)
            self.current_initial_date = initial_date
            self.run_id = initial_date.strftime('%y%m%d%H')

            # 2.1 Download file for given initial date
            try: 
                self.rclone_copy()
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if "corrupted on transfer" in stderr and "hash differ" in stderr:
                    print(
                        f"⚠️ [WARNING] Corrupted transfer for file {self.file_name} "
                        f"(checksum mismatch)",
                        flush=True
                    )
                elif "not found" in stderr or "directory not found" in stderr:
                    print(
                        f"⚠️ [WARNING] File {self.file_name} not found on Nextcloud",
                        flush=True
                    )
                else:
                    print(
                        f"❌ [ERROR] rclone failed for file {self.file_name}\n{stderr}",
                        flush=True
                    )
                continue
                
            self.ds = xr.open_dataset(f"{self.local_folder_path}/{self.file_name}")

            for lead_time in EVAL_LEAD_TIMES:
                self.current_lead_time = lead_time
                self.get_station_observation(obs_folder_path=f"{FIR_SCRATCH}/eccc_data/")  # one DF for one lead time

                for field_mod, field_obs in OBS_EVAL_FIELDS.items():
                    try:
                        xtime = self.xtime(initial_date,lead_time)
                        print(f" - xtime: {xtime}")
                        field_data = self.ds[field_mod].sel(XTIME=self.xtime(initial_date, lead_time))
                    except KeyError as e:
                        print(f"⚠️ [WARNING] Lead time {lead_time} not found in file {self.file_name}", flush=True)
                        continue

                    if self.counter == 0:
                        self.compute_coordinates()

                    self.get_prediction_at_station_loc(field_data)
                    self.rmse(field_mod)

                self.counter += 1

            self.ds.close()
            os.remove(f"{self.local_folder_path}/wrfout_d02_processed_{self.run_id}.nc") # remove wrfout file to save scratch space

        self.save_error_df()

class DLModelEvaluator(ModelEvaluator):
    def __init__(self,
                date_range,
                lead_times: list[str],
                prediction_data_path: str,
                system: str,
                ):
        super().__init__(
                date_range=date_range,
                lead_times=lead_times,
                model_name='dl_glob',
                system=system,
            )

        self.predictions_data_path = prediction_data_path
        self.forecast_ds = None

    def compute_coordinates(self):
        self.coords = np.column_stack((self.forecast_ds['longitude'].values, self.forecast_ds['latitude'].values))

    def evaluate(self):
        for initial_date in self.date_range:
            print(f"⚡️ date: {initial_date}", flush=True)
            self.current_initial_date = initial_date

            self.forecast_ds = xr.open_dataset(f"{self.predictions_data_path}")

            for lead_time in EVAL_LEAD_TIMES:
                self.current_lead_time = lead_time
                self.get_station_observation(obs_folder_path=f"{OLIVIA_GROUP_SCRATCH}/datasets/ECCC_OBS")

                for field_mod, field_obs in OBS_EVAL_FIELDS.items():

                    if field_mod == '10ff':
                        try: 
                            u10 = self.forecast_ds['10u'].sel(initial_date=initial_date, lead_time=int(lead_time))
                            v10 = self.forecast_ds['10v'].sel(initial_date=initial_date, lead_time=int(lead_time))
                            field_data = mpcalc.wind_speed(u10.values*units('m/s'), v10.values*units('m/s'))
                            field_data = field_data.magnitude
                        except KeyError as e:
                            print(f"⚠️ [WARNING] Lead time {lead_time} not found in file {self.predictions_data_path.split('/')[-1]}", flush=True)
                            continue
                    else: 
                        try:
                            field_data = self.forecast_ds[field_mod].sel(initial_date=initial_date, lead_time=int(lead_time))
                        except KeyError as e:
                            print(f"⚠️ [WARNING] Lead time {lead_time} not found in file {self.predictions_data_path.split('/')[-1]}", flush=True)
                            continue

                    if self.counter == 0:
                        self.compute_coordinates()
                
                    self.get_prediction_at_station_loc(field_data)
                    self.rmse(field_mod)

                    self.counter += 1

            self.forecast_ds.close()
            self.save_error_df()
