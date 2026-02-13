import os
import xarray as xr
import xwrf
import xgcm
import metpy
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from pint import UnitRegistry
from metpy.units import units
import subprocess

DATA_DIR = '/scratch/juchar/climatology/'
GRAVITY_CONSTANT = 9.81
ANEMOI_PRESSURE_LEVELS = np.array(
    [
        1000.0,
        925.0,
        850.0,
        700.0,
        500.0,
        400.0,
        300.0,
        250.0,
        200.0,
        150.0,
        100.0,
        50.0,
    ]
)  # in hPa
pressure_level_variables = [
        "U",
        "V",
        "W",
        "geopotential",
        "geopotential_height",
        "QVAPOR",
        "air_potential_temperature",
    ]
variables_to_quantify = {
        "U": units("m/s"),
        "V": units("m/s"),
        "geopotential": units("meter squared / second squared"),
        "geopotential_height": units("meter"),
        "QVAPOR": units("kg/kg"),
        "air_potential_temperature": units.kelvin,
        "T2": units.kelvin,
        "Q2": units("kg/kg"),
    }

def process_wrf(wrfout_f, wrfuvic_file):

    # decompress file
    if not os.path.exists(os.path.join(DATA_DIR, wrfout_f[:-11])):
        wrfout_file = decompress_file(wrfout_f)
    else:
        print(f" ** {wrfout_f} already decompressed")
        wrfout_file = wrfout_f[:-11]

    # open files
    raw_wrfout = xr.open_dataset(f"{DATA_DIR}/{wrfout_file}", engine="netcdf4")
    date = np.array([f"{wrfout_file[11:21]}T{wrfout_file[22:24]}:00:00"])

    if not raw_wrfout.dims or not raw_wrfout.data_vars:   # filter out empty files
        raw_wrfout.close()
        raise ValueError(f"Empty or corrupted NetCDF file: {wrfout_file}")

    raw_wrfuvic = xr.open_dataset(f"{DATA_DIR}/{wrfuvic_file}", engine="netcdf4")
    
    # process files
    pp_wrfout = raw_wrfout.xwrf.postprocess()
    ds_wrfout = pp_wrfout.xwrf.destagger()
    grid = xgcm.Grid(ds_wrfout, periodic=False)

    air_pressure_hpa = ds_wrfout.air_pressure / 100.0
    processed_variables = {}
    for var in pressure_level_variables:
        processed_variables[var] = interpolate_variable(
            variable=ds_wrfout[var], air_pressure=air_pressure_hpa, grid=grid
        )
    for var, unit in variables_to_quantify.items():
        if var in processed_variables.keys():
            processed_variables[var] = quantify_variable(
                variable=processed_variables[var], unit=unit
            )
        else:
            processed_variables[var] = quantify_variable(
                variable=ds_wrfout[var], unit=unit
            )

    air_temperature = mpcalc.temperature_from_potential_temperature(
    pressure=processed_variables["air_potential_temperature"].air_pressure * units("hPa"),
    potential_temperature=processed_variables["air_potential_temperature"],
    )

    ds = xr.Dataset({
        "2t": processed_variables["T2"],  # 2m temperature, [K]
        "10u": ds_wrfout.U10,  # should be in [m/s]
        "10v": ds_wrfout.V10,  # should be in [m/s]
        "msl": calculate_sea_level_pressure(
            elevations=ds_wrfout.HGT,
            surface_pressure=ds_wrfout.PSFC * units("Pa"),
            air_temperature_2m=processed_variables["T2"],
            q_2m=processed_variables["Q2"],
        ),
        "sp": ds_wrfout.PSFC * units("Pa"),  # surface pressure, should be in Pa
        "surface_geopotential": ds_wrfout.HGT * GRAVITY_CONSTANT,
        "tp": calculate_accumulated_variable("tp", raw_wrfuvic) / 1000,  # unit: mm to m
        "u": processed_variables["U"],
        "v": processed_variables["V"],
        "t": air_temperature.transpose("Time", ..., "air_pressure"),
        "geopotential": processed_variables[
                "geopotential"
            ],  # geopotential, should be in [m^2/s^2]
    })

    ds = ds.rename(
        {
            "air_pressure": "level",
            "Time": "time",
        }
    )

    ds = ds.drop_vars(["XTIME", "CLAT"])

    ds['x'].attrs['axis'] = 'X'
    ds['y'].attrs['axis'] = 'Y'
    ds['level'].attrs['axis'] = 'Z'
    ds["time"].attrs.update(
        {"standard_name": "time", "axis": "T"}
    )

    return ds

def calculate_sea_level_pressure(
        elevations, surface_pressure, air_temperature_2m, q_2m
    ):
        ideal_gas_constant = 29.3  # m/K

        virtual_temp_2m = mpcalc.virtual_temperature(air_temperature_2m, q_2m)
        virtual_temp_2m_with_lapse_rt = (
            virtual_temp_2m.values + 0.0065 * elevations
        )  # lapse rate of 6.5 K/km

        slp = surface_pressure * np.exp(
            elevations / (ideal_gas_constant * virtual_temp_2m_with_lapse_rt)
        )

        # Add the units
        slp_units = slp * units("Pa")
        slp_units.metpy.quantify()

        # add to our dataset
        return slp_units


def decompress_file(f):
    compressed = "_compressed" in f
    if compressed:
        file_name = f[:-11]
        decompress_cmd = f"nccopy -d0 -s {os.path.join(DATA_DIR,f)} {os.path.join(DATA_DIR,file_name)}"
        result = subprocess.run(decompress_cmd, shell=True, check=True)
    else:
        file_name = f

    return file_name


def interpolate_variable(variable, air_pressure, grid):
    """
    relevant xgcm documentation: https://xgcm.readthedocs.io/en/latest/transform.html

    """
    return grid.transform(
        variable,
        "Z",
        ANEMOI_PRESSURE_LEVELS,
        target_data=air_pressure,
        method="log",
        mask_edges=False,  # if set to 'False' value outside the range of target will be filled with the nearest valid value
    )

def quantify_variable(variable, unit):
    variables_with_units = variable * unit
    return variables_with_units.metpy.quantify()


def get_previous_datetime(input_datetime, time_delta=6):

    if isinstance(input_datetime, str):
        dt = datetime.strptime(input_datetime, "%Y-%m-%d_%H:%M:%S")
    else:
        dt = input_datetime

    previous_dt = dt - timedelta(hours=time_delta)

    return previous_dt


def calculate_total_rain(raw_data, var_name_list):
    """
    RAINNC is remainder, RAINC is cumulative
    tp = rainnc + BUCKET_MM*I_RAINNC + rainc + BUCKET_MM*I_RAINC
    BUCKET_MM :100.0
    """
    try:
        bucket_str = raw_data.attrs["BUCKET_MM"]
        bucket = float(bucket_str)
    except Exception as e:
        # print (raw_data.attrs)
        # print(f"WARNING: could not read BUCKET_MM attribute from dataset: {e}. Set default value to 100.0")
        bucket = 100.0  # default value
        # raise e
    total_rain = (
        raw_data[var_name_list[0]]
        + bucket * raw_data[var_name_list[1]]
        + raw_data[var_name_list[2]]
        + bucket * raw_data[var_name_list[3]]
    )
    return total_rain


def calculate_accumulated_variable(variable, raw_data):
    """
    Separate bins are used because of precision issues.
    So here, define variables and their bin names according to WRF documentation.

    Source: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.1/users_guide_chap5.html
    """
    var_name_list = {
        "tp": ["RAINNC", "I_RAINNC", "RAINC", "I_RAINC"],
    }.get(variable)

    calculate_var_fctn = {
        "tp": calculate_total_rain,
    }.get(variable)

    file_prefix = {"tp": "wrfuvic"}.get(variable)

    # get previous datetime
    d = raw_data.Times.values[0]
    date = d.decode("utf-8")

    previous_dt = get_previous_datetime(date)
    previous_dt_str = previous_dt.strftime("%Y-%m-%d_%H:%M:%S")
    previous_ds_name = f"{file_prefix}_d03_{previous_dt_str}"

    # calculate current total accumulated radiation
    total = calculate_var_fctn(raw_data, var_name_list)

    # TODO: handle case when we don't have the file, download it maybe?
    if os.path.exists(os.path.join(DATA_DIR, previous_ds_name)):
        previous_ds = xr.open_dataset(os.path.join(DATA_DIR, previous_ds_name))

    else:
        print(f" * Decompressing previous file: {previous_ds_name}")
        previous_file_name = decompress_file(previous_ds_name + "_compressed")
        previous_ds = xr.open_dataset(os.path.join(DATA_DIR, previous_file_name))

    # calculate total rain
    previous_total = calculate_var_fctn(previous_ds, var_name_list)

    # substract and return result
    accumulated_variable = total - previous_total

    previous_ds.close()

    accumulated_variable = accumulated_variable.rename(
        {
            "Time": "time",
            "south_north": "y",
            "west_east": "x",
            "XLONG": "longitude",
            "XLAT": "latitude",
        }
    )

    return accumulated_variable

