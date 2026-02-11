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
):
    if model_name == "dl_reg":
        evaluator = RegDLScorecardEvaluator(
            date_range=date_range,
            lead_times=lead_times,
        )

    elif model_name == "nwp_reg":
        evaluator = RegNWPScorecardEvaluator(
            date_range=date_range,
            lead_times=lead_times,
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
    ):

        self.date_range = date_range
        self.lead_times = lead_times
        self.model_name = model_name

        self.current_initial_date = None
        self.current_lead_time = None
        self.scorecard_df = pd.DataFrame(
            columns=["initial_date", "lead_time", "field", "model", "rmse"]
        )

        self.ground_truth_ds = xr.open_dataset(
            f"{FIR_SCRATCH}/data/cleaned_data/anemoi-climatex-training-6h-20230101-20231231.zarr",
            engine="zarr",
        )
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

    @property
    def climatex_missing(self):
        return [
            datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
            for d in self.ground_truth_ds.attrs["missing_dates"]
        ]

    @staticmethod
    def xtime(a, b):
        return datetime.strftime(a + pd.Timedelta(hours=int(b)), format="%Y-%m-%d")

    @staticmethod
    def rmse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):
        assert array1.shape == array2.shape, "Both arrays must have the same shape"
        assert (
            array1.ndim == 1 and array2.ndim == 1
        ), "Both arrays must be one-dimesional"

        rmse = np.sqrt(np.power(array1 - array2, 2)).mean()  # spatial average

        return rmse.values

    def get_ground_truth(self, t, field):
        field_index = self.climatex_var_map.get(field)
        return self.ground_truth_ds["data"][t, field_index, 0, :]

    def save_scorecard_df(self):
        self.scorecard_df.to_csv(
            f"{ERROR_DATA_DIR}/scorecard-{self.model_name}-{self.date_range[0].strftime('%Y%m%d')}_{self.date_range[-1].strftime('%Y%m%d')}.csv"
        )


class RegNWPScorecardEvaluator(ScorecardEvaluator):
    def __init__(
        self,
        date_range,
        lead_times: list[str],
    ):
        super().__init__(
            date_range=date_range, lead_times=lead_times, model_name="nwp_reg"
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

        cmd = ["rclone", "copy", source, self.local_folder_path, "--checksum"]
        # print(' - rclone cmd: ', cmd)
        subprocess.run(cmd, capture_output=True, check=True, text=True)

    @staticmethod
    def xtime(a, b):
        return datetime.strftime(
            a + pd.Timedelta(hours=int(b)), format="%Y-%m-%dT%H:00:00.000000000"
        )

    @staticmethod
    def clip_coords(wrf_coords, climatex_coords, tolerance=0.03):

        tree_wrf = cKDTree(wrf_coords)
        indices = tree_wrf.query_ball_point(climatex_coords, r=tolerance)
        climatex_mask = np.array([len(idx) > 0 for idx in indices])

        tree_climatex = cKDTree(climatex_coords)
        indices = tree_climatex.query_ball_point(wrf_coords, r=tolerance)
        wrf_mask = np.array([len(idx) > 0 for idx in indices])

        return climatex_mask, wrf_mask

    def get_domain_coords(self, wrf_ds, climatex_ds):

        # TODO : correct
        wrf_coords = np.column_stack(
            (wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())
        )
        climatex_coords = np.column_stack(
            (climatex_ds["longitudes"].values, climatex_ds["latitudes"].values)
        )

        climatex_mask, wrf_mask = self.clip_coords(wrf_coords, climatex_coords)
        clipped_wrf_coords = wrf_coords[wrf_mask]
        clipped_climatex_coords = climatex_coords[climatex_mask]

        return clipped_wrf_coords, clipped_climatex_coords, wrf_mask, climatex_mask

    def evaluate(self):
        datetime_index = 0
        counter = 0

        for initial_date in self.date_range:
            print(f"⚡️ date : {initial_date}", flush=True)
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

            for lead_time in EVAL_LEAD_TIMES:
                # print(f" - lead time: {lead_time}")
                self.current_lead_time = lead_time

                xtime = initial_date + pd.Timedelta(hours=int(lead_time))
                # print(f"⚡️ datetime {datetime_index} : {xtime}", flush=True)

                if xtime in self.climatex_missing:
                    print(f"⚠️ [WARNING] datetime {xtime} missing in climatex dataset. ")
                    continue

                for field in self.climatex_var_map.keys():
                    raw_truth_field = self.get_ground_truth(datetime_index, field)

                    try:
                        xtime = self.xtime(initial_date, lead_time)
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
                            f"⚠️ [WARNING] Lead time {lead_time} not found in file {self.file_name}",
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
                        ) = self.get_domain_coords(self.ds, self.ground_truth_ds)

                    wrf_field = pyresample_resampling(
                        src_coords=clipped_wrf_coords,
                        tgt_coords=clipped_climatex_coords,
                        data=raw_wrf_field[wrf_mask],
                    )
                    truth_field = raw_truth_field[climatex_mask]

                    rmse = self.rmse(wrf_field, truth_field)
                    self.scorecard_df.loc[len(self.scorecard_df)] = [
                        initial_date.strftime("%Y-%m-%dT%H:00:00"),
                        lead_time,
                        field,
                        self.model_name,
                        rmse,
                    ]
                    # print(f" field : {field}, rmse : {rmse.values}")
                    counter += 1

                datetime_index += 1

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
    ):
        super().__init__(
            date_range=date_range,
            lead_times=lead_times,
            model_name="dl_reg",
        )
        self.forecast_folder = (
            f"/cluster/projects/nn10090k/results/juchar/climatex-lam-inference"
        )
        self.forecast_ds = None

    def evaluate(self):
        datetime_index = 0
        counter = 0

        for initial_date in self.date_range:
            print(f"⚡️ date : {initial_date}", flush=True)
            self.current_initial_date = initial_date

            self.forecast_ds = xr.open_dataset(
                f"{self.forecast_folder}/{initial_date.strftime("%Y%m%dT%H")}.nc"
            )

            for lead_time in EVAL_LEAD_TIMES:
                self.current_lead_time = lead_time
                xtime = initial_date + pd.Timedelta(hours=int(lead_time))
                # print(f"⚡️ datetime {datetime_index} : {xtime}", flush=True)

                if xtime in self.climatex_missing:
                    print(
                        f"⚠️ [WARNING] datetime {xtime} missing in climatex dataset.",
                        flush=True,
                    )
                    continue

                for field in self.climatex_var_map.keys():
                    raw_truth_field = self.get_ground_truth(datetime_index, field)

                    try:
                        field_data = self.forecast_ds[field].sel(
                            time=self.xtime(initial_date, lead_time)
                        )
                    except KeyError as e:
                        print(
                            f"⚠️ [WARNING] Lead time {lead_time} not found in file {self.predictions_data_path.split('/')[-1]}",
                            flush=True,
                        )

                    # Clip domain
                    if counter == 0:
                        (
                            clipped_wrf_coords,
                            clipped_climatex_coords,
                            wrf_mask,
                            climatex_mask,
                        ) = self.get_domain_coords(self.ds, self.ground_truth_ds)
