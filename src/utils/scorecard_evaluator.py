from abc import ABC, abstractmethod
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pyresample import geometry, bilinear, kd_tree
from scipy.spatial import cKDTree
from utils.resampling_utils import pyresample_resampling
import os
import warnings
warnings.filterwarnings("ignore", message="Engine 'cfgrib'")

import matplotlib.pyplot as plt

from utils.data_constants import (
    PRED_DATA_DIR,
    ERROR_DATA_DIR,
    OBS_EVAL_FIELDS,
    FIR_SCRATCH,
    EVAL_LEAD_TIMES,
)


def scorecard_evaluator_factory(
    model_name: str,
    date_range,
    lead_times: list[str],
    system: str,
):
    print(f" [INFO] model : {model_name}")
    print(f" [INFO] system : {system}")
    if ('stage-' in model_name) | ('bris' in model_name):
        evaluator = RegDLScorecardEvaluator(
            date_range=date_range,
            lead_times=lead_times,
            model_name=model_name,
            system=system,
        )

    elif model_name == "nwp_reg":
        evaluator = RegNWPScorecardEvaluator(
            date_range=date_range,
            lead_times=lead_times,
            system=system
        )
    else:
        raise NotImplementedError

    return evaluator


class ScorecardEvaluator(ABC):
    def __init__(
        self,
        date_range,
        lead_times: list[str],
        model_name: str,
        system: str,
    ):

        self.date_range = date_range
        self.lead_times = lead_times
        self.model_name = model_name
        print(f" [INFO] Evaluating for lead times : {lead_times}")

        self.current_initial_date = None
        self.current_lead_time = None
        self.scorecard_df = pd.DataFrame(
            columns=["initial_date", "lead_time", "field", "model", "rmse", "mse"]
        )

        self.system = system
        if system == 'fir':
            self.ground_truth_ds = xr.open_dataset(
                f"{FIR_SCRATCH}/data/cleaned_data/anemoi-climatex-training-6h-20190601-20231231.zarr",
                engine="zarr",
            )
            print(f" [INFO] Found ground truth dataset at path : {FIR_SCRATCH}/data/cleaned_data/anemoi-climatex-training-6h-20230101-20231231.zarr")
        elif system == 'olivia':
            self.ground_truth_ds = xr.open_dataset(
                f"/nird/datapeak/NS10090K/datasets/climatex/anemoi-climatex-training-6h-20190601-20231231.zarr",
                engine="zarr",
            )
            print(f" [INFO] Found ground truth dataset at path : /nird/datapeak/NS10090K/datasets/climatex/anemoi-climatex-training-6h-20190101-20231231.zarr")
        else: 
            raise NotImplementedError

        self.climatex_var_map = {
            "2t": 3,
            "10u": 0,
            "10v": 1,
            "tp": 61,
            "msl": 25,
            "z_50": 15,
            "z_100": 8,
            "z_250": 12,
            "z_500": 16,
            "z_850": 18,
            "t_50": 54,
            "t_100": 47,
            "t_250": 51,
            "t_500": 55,
            "t_850": 57,
            "u_50": 69,
            "u_100": 62,
            "u_250": 66,
            "u_500": 70,
            "u_850": 72,
            "v_50": 81,
            "v_100": 86,
            "v_250": 78,
            "v_500": 82,
            "v_850": 84,
        }

        print(" [INFO] Evaluating fields ", list(self.climatex_var_map.keys()))

    @property
    def climatex_missing(self):
        return [
            datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
            for d in self.ground_truth_ds.attrs["missing_dates"]
        ]

    @property
    def gt_time(self):
        start = self.ground_truth_ds.attrs["start_date"]
        end = self.ground_truth_ds.attrs["end_date"]
        frequency = self.ground_truth_ds.attrs["frequency"]
        time = pd.date_range(start=start, end=end, freq=frequency)
        return time

    @staticmethod
    def xtime(a, b):
        return datetime.strftime(a + pd.Timedelta(hours=int(b)), format="%Y-%m-%d")

    @staticmethod
    def clip_coords(wrf_coords, prediction_coords, climatex_coords, tolerance=0.03):
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

        # Propagate intersection to the climatex domain
        tree_climatex = cKDTree(climatex_coords)
        dist, idx_map = tree_climatex.query(prediction_coords, k=1)

        if not np.all(dist < 1e-12):
            raise ValueError("D2 is not an exact subset of D1.")

        climatex_mask = np.zeros(len(climatex_coords), dtype=bool)
        climatex_mask[idx_map[prediction_mask]] = True

        return prediction_mask, wrf_mask, climatex_mask


    def get_domain_coords(self, wrf_ds, prediction_ds, climatex_ds):
        wrf_coords = np.column_stack(
            (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
        )
        prediction_coords = np.column_stack(
            (prediction_ds["longitude"].values, prediction_ds["latitude"].values)
        )
        climatex_coords = np.column_stack(
            (climatex_ds["longitudes"].values, climatex_ds["latitudes"].values)
        )

        prediction_mask, wrf_mask, climatex_mask = self.clip_coords(wrf_coords, prediction_coords, climatex_coords)

        clipped_climatex_coords = climatex_coords[climatex_mask]
        clipped_prediction_coords = prediction_coords[prediction_mask]

        return prediction_mask, wrf_mask, climatex_mask 

    def get_ground_truth(self, xtime, field):
        field_index = self.climatex_var_map.get(field)
        time_index = np.where(self.gt_time == xtime)
        # print(f" ** xtime : {xtime}")
        # print(f" ** time index : {len(time_index)} {time_index}")
        if len(time_index) > 0:
            data = self.ground_truth_ds["data"][time_index[0], field_index, 0, :]
        else:
            print(" NO MATCHING DATE")
            raise ValueError(f"No matching date in ground truth dataset for datetime {xtime}")

        return data

    def save_scorecard_df(self):
        self.scorecard_df.to_csv(
            f"{ERROR_DATA_DIR}/scorecard-{self.model_name}-{self.date_range[0].strftime('%Y%m%d')}_{self.date_range[-1].strftime('%Y%m%d')}.csv"
        )
        print(f" [INFO] ✅ Saved Dataframe to csv at path : {ERROR_DATA_DIR}/scorecard-{self.model_name}-{self.date_range[0].strftime('%Y%m%d')}_{self.date_range[-1].strftime('%Y%m%d')}.csv")


class RegNWPScorecardEvaluator(ScorecardEvaluator):
    def __init__(
        self,
        date_range,
        lead_times: list[str],
        system: str,
    ):
        super().__init__(
            date_range=date_range, 
            lead_times=lead_times, 
            model_name="nwp_reg", 
            system=system,
        )

        self.run_id = None
        self.model_id = "WAC00WG-01"
        self.local_folder_path = f"{FIR_SCRATCH}/wrf_data"

        self.ds = None

    @property
    def file_name(self):
        return f"wrfout_d02_processed_{self.run_id}.nc"

    def rclone_copy(self):
        source = f"wfrt-nextcloud:Documents/WRF-forecasts/{self.model_id}/wrfout_d02_processed_{self.run_id}.nc"
        # cmd = f"rclone copy '{source}' '{self.local_folder_path}'"# --progress"

        # cmd = ["rclone", "copy", source, self.local_folder_path, "--checksum"]
        cmd = ["rclone", "copy", source, self.local_folder_path, "--progress"]
        print(f" [INFO] running rclone command : {cmd}")
        # print(' - rclone cmd: ', cmd)
        subprocess.run(cmd, capture_output=True, check=True, text=True)

    @staticmethod
    def xtime(a, b):
        return datetime.strftime(
            a + pd.Timedelta(hours=int(b)), format="%Y-%m-%dT%H:00:00.000000000"
        )

    @staticmethod
    def rmse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):
        array1, array2 = np.asarray(array1), np.asarray(array2)
        assert array1.shape == array2.shape, "Both arrays must have the same shape"
        assert (
            array1.ndim == 1 and array2.ndim == 1
        ), "Both arrays must be one-dimensional"

        rmse = np.sqrt(np.power(array1 - array2, 2)).mean()  # spatial average

        return rmse
    @staticmethod

    def mse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):
        array1, array2 = np.asarray(array1), np.asarray(array2)
        assert array1.shape == array2.shape, "Both arrays must have the same shape"
        assert (
            array1.ndim == 1 and array2.ndim == 1
        ), "Both arrays must be one-dimensional"

        mse = np.power(array1 - array2, 2).mean()

        return mse
    
    def clip_domain(self):
        prediction_ds = xr.open_dataset('/scratch/juchar/prediction_data/20220701T00.nc')
        _, wrf_mask, climatex_mask = self.get_domain_coords(self.ds, prediction_ds, self.ground_truth_ds)

        wrf_coords = np.column_stack(
            (self.ds.XLONG.values.flatten(), self.ds.XLAT.values.flatten())
        )
        climatex_coords = np.column_stack(
            (self.ground_truth_ds["longitudes"].values, self.ground_truth_ds["latitudes"].values)
        )
        clipped_wrf_coords = wrf_coords[wrf_mask]
        clipped_climatex_coords = climatex_coords[climatex_mask]

        return clipped_wrf_coords, clipped_climatex_coords, wrf_mask, climatex_mask


    def evaluate(self):
        # print(f" *** GT TIME : {len(self.gt_time)}")
        # print(f" *** gt data : {self.ground_truth_ds.time.shape}")
        # print(f" *** missing date : {self.climatex_missing}")
        counter = 0

        print(f" [INFO] Starting evaluation for date range : {self.date_range[0]} - {self.date_range[-1]}")

        for initial_date in self.date_range:
            print(f" [INFO] ⚡️ date : {initial_date}", flush=True)
            self.current_initial_date = initial_date
            self.run_id = initial_date.strftime("%y%m%d%H")

            # 2.1 Download file for given initial date
            try:
                self.rclone_copy()
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if "corrupted on transfer" in stderr and "hash differ" in stderr:
                    print(
                        f"⚠️ [WARNING] Corrupted transfer for file {self.file_name} "
                        f"(checksum mismatch)",
                        flush=True,
                    )
                elif "not found" in stderr or "directory not found" in stderr:
                    print(
                        f"⚠️ [WARNING] File {self.file_name} not found on Nextcloud",
                        flush=True,
                    )
                else:
                    print(
                        f"❌ [ERROR] rclone failed for file {self.file_name}\n{stderr}",
                        flush=True,
                    )
                continue

            self.ds = xr.open_dataset(f"{self.local_folder_path}/{self.file_name}")
            print(f" [INFO] Openened forecast data at path {self.local_folder_path}/{self.file_name}")

            for lead_time in EVAL_LEAD_TIMES:
                # print(f" - lead time: {lead_time}")
                self.current_lead_time = lead_time

                xtime = initial_date + pd.Timedelta(hours=int(lead_time))

                if xtime.year == 2024:
                    print('break')
                    break

                print(f" [INFO] xtime : {xtime}")

                if xtime in self.climatex_missing:
                    print(f"⚠️ [WARNING] datetime {xtime} missing in ground truth dataset. ")
                    continue

                for field in self.climatex_var_map.keys():
                    raw_truth_field = self.get_ground_truth(xtime, field)

                    try:
                        if "_" in field:
                            var, level = field.split("_")
                            var = "geopotential" if var == "z" else var
                            raw_wrf_field = (
                                self.ds[var]
                                .sel(
                                    XTIME=self.xtime(initial_date, lead_time),
                                    air_pressure=int(level),
                                )
                                .values.flatten()
                            )
                        else:
                            raw_wrf_field = (
                                self.ds[field]
                                .sel(XTIME=self.xtime(initial_date, lead_time))
                                .values.flatten()
                            )
                    except KeyError as e:
                        print(
                            f"⚠️ [WARNING] Lead time {lead_time} or field {field} not found in file {self.file_name}",
                            flush=True,
                        )
                        continue

                    # 2.3 Grid interpolation
                    if counter == 0:
                        (
                            clipped_wrf_coords,
                            clipped_climatex_coords,
                            wrf_mask,
                            climatex_mask,
                        ) = self.clip_domain()

                    wrf_field = pyresample_resampling(
                        src_coords=clipped_wrf_coords,
                        tgt_coords=clipped_climatex_coords,
                        data=raw_wrf_field[wrf_mask],
                    )
                    truth_field = raw_truth_field.squeeze()[climatex_mask]

                    # print(f" [INFO] wrf_field.shape : {wrf_field.shape}")
                    # print(f" [INFO] truth_field.shape : {truth_field.shape}")

                    rmse = self.rmse(wrf_field, truth_field)
                    mse = self.mse(wrf_field, truth_field)
                    print(f" [INFO] rmse for field {field} : {rmse}")
                    self.scorecard_df.loc[len(self.scorecard_df)] = [
                        initial_date.strftime("%Y-%m-%dT%H:00:00"),
                        lead_time,
                        field,
                        self.model_name,
                        rmse,
                        mse,
                    ]
                    # print(f" field : {field}, rmse : {rmse.values}")
                    counter += 1

            self.ds.close()
            os.remove(
                f"{self.local_folder_path}/wrfout_d02_processed_{self.run_id}.nc"
            )  # remove wrfout file to save scratch space

        # save scores to csv
        self.save_scorecard_df()


class RegDLScorecardEvaluator(ScorecardEvaluator):
    def __init__(
        self,
        date_range,
        lead_times: list[str],
        model_name: str,
        system: str,
    ):
        super().__init__(
            date_range=date_range,
            lead_times=lead_times,
            model_name=model_name,
            system=system
        )
        self.forecast_folder = (
            f"/cluster/projects/nn10090k/results/juchar/{model_name}-lam-inference"
        )
        print(f" [INFO] Looking for inference results in {self.forecast_folder}")
        self.forecast_ds = None # holds the prediction (inference) for self.current_initial_date

    @staticmethod
    def rmse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):
        array1, array2 = np.asarray(array1), np.asarray(array2)
        assert array1.shape == array2.shape, "Both arrays must have the same shape"

        rmse = np.sqrt(np.power(array1 - array2, 2)).mean(axis=1)  # spatial average

        return rmse

    @staticmethod
    def mse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):
        array1, array2 = np.asarray(array1), np.asarray(array2)
        assert array1.shape == array2.shape, "Both arrays must have the same shape"

        mse = np.power(array1-array2, 2).mean(axis=1)

        return mse
    

    def clip_domain(self):
        wrf_ds = xr.open_dataset('/cluster/projects/nn10090k/juchar/wrfout_d02_processed_23010100.nc')
        prediction_mask, _, climatex_mask = self.get_domain_coords(wrf_ds, self.forecast_ds, self.ground_truth_ds)

        return prediction_mask, climatex_mask
        
    def evaluate(self):
        # print(f" *** GT TIME : {len(self.gt_time)}")
        # print(f" *** gt data : {self.ground_truth_ds.time.shape}")
        # print(f" *** missing date : {self.climatex_missing}")
        counter = 0

        print(f" [INFO] Starting evaluation for date range : {self.date_range[0]} - {self.date_range[-1]}")
   
        for initial_date in self.date_range:
            print(f" [INFO] ⚡️ current_initial_date : {initial_date}", flush=True)
            self.current_initial_date = initial_date

            try:
                self.forecast_ds = xr.open_dataset(
                    f"{self.forecast_folder}/{initial_date.strftime("%Y%m%dT%H")}.nc",
                    engine="netcdf4"
                )   # inference folder, file for that specific initial date
                print(f" [INFO] Opened inference data at path {self.forecast_folder}/{initial_date.strftime("%Y%m%dT%H")}.nc")
            except FileNotFoundError as e:
                print(
                    f"⚠️ [WARNING] Error when opening inference data : {e}.",
                    flush=True,
                )
                continue


            for lead_time in EVAL_LEAD_TIMES:
                self.current_lead_time = lead_time
                xtime = initial_date + pd.Timedelta(hours=int(lead_time))

                if xtime.year == 2024:
                    print('break')
                    break

                # TODO: try without this check : probs nan propagation ?
                if xtime in self.climatex_missing:
                    print(
                        f"⚠️ [WARNING] datetime {xtime} missing in climatex dataset.",
                        flush=True,
                    )
                    continue

                # go over field in parallel
                fields = list(self.climatex_var_map.keys())
                field_indices = [self.climatex_var_map[f] for f in fields]

                # get ground truth
                time_idx = np.where(self.gt_time == xtime)[0]
                if len(time_idx) == 0:
                    raise ValueError(f"No matching date in ground truth dataset for datetime {xtime}")
                time_idx = time_idx[0]

                raw_truth_fields = self.ground_truth_ds["data"].isel(
                    time=time_idx,
                    variable=field_indices,
                    ensemble=0,
                ) 
                raw_truth_fields = raw_truth_fields.assign_coords(variable=fields)
                try:
                    raw_prediction_fields = self.forecast_ds[fields].sel(time=xtime).to_array(dim="variable")
                except KeyError as e:
                    print(
                        f"⚠️ [WARNING] Lead time {lead_time} missing in inference data {self.forecast_folder}/{initial_date.strftime("%Y%m%dT%H")}.nc.",
                        flush=True,
                    )

                if counter == 0:
                    prediction_mask, climatex_mask = self.clip_domain()

                truth_fields = raw_truth_fields.squeeze()[...,climatex_mask]
                prediction_fields = raw_prediction_fields.squeeze()[...,prediction_mask]    

                rmse = self.rmse(prediction_fields, truth_fields)
                mse = self.mse(prediction_fields, truth_fields)
                rows = pd.DataFrame({
                    "initial_date": initial_date.strftime("%Y-%m-%dT%H:00:00"),
                    "lead_time": lead_time,
                    "field": fields,
                    "model": self.model_name,
                    "rmse": rmse,
                    "mse": mse,
                })

                self.scorecard_df = pd.concat([self.scorecard_df, rows], ignore_index=True)
                counter += 1
        
        # save scores to csv
        self.save_scorecard_df()