import xarray as xr
import numpy as np
import xwrf
import xgcm
import metpy.calc as mpcalc
from metpy.units import units
import argparse

P_LEVELS = np.array([50, 100, 250, 500, 850])


def interpolate_variable(variable, air_pressure, grid):
    """
    relevant xgcm documentation: https://xgcm.readthedocs.io/en/latest/transform.html
    :variable   : ds_pp[var]
    :air_pressure   : ds_pp.air_pressure / 100.0  (conversion from Pa to hPa)
    :grid   :   xgcm.Grid(ds_pp, periodic=False)

    """
    return grid.transform(
        variable,
        "Z",
        P_LEVELS,
        target_data=air_pressure,
        method="log",
        mask_edges=False,  # if set to 'False' value outside the range of target will be filled with the nearest valid value
    )


def quantify_variable(variable, unit):
    variables_with_units = variable * unit
    return variables_with_units.metpy.quantify()


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


def calculate_total_rain(ds):
    """
    RAINNC: ACCUMULATED TOTAL GRID SCALE PRECIPITATION
    RAINC: ACCUMULATED TOTAL CUMULUS PRECIPITATION
    RAINSH: ACCUMULATED SHALLOW CUMULUS PRECIPITATION
    """

    tp_6h = ds.RAINNC.diff(dim="XTIME")  # 6h accumulation
    tp_6h = tp_6h.reindex(XTIME=ds.XTIME, fill_value=float("nan"))

    tp_6h.name = "tp_6h"
    tp_6h.attrs["units"] = "mm"
    tp_6h.attrs["description"] = "6-hour accumulated precipitation"

    return tp_6h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process raw wrfout")
    parser.add_argument(
        "--model", type="str", required=True, help="Model folder name, ex: WAC00WG-01"
    )
    parser.add_argument(
        "--id", type="str", required=True, help="Forecast inital dat, ex: 23040300"
    )
    args = parser.parse_args()

    FOLDER_NAME = f"/archives/forecasts/{args.model}/{args.id}"
    raw_wrfout = xr.open_dataset(f"{FOLDER_NAME}/merged.nc")
    pp_wrfout = raw_wrfout.xwrf.postprocess()
    ds_wrfout = pp_wrfout.xwrf.destagger()

    # interpolate variables
    processed_variables = {}
    air_pressure = ds_wrfout.air_pressure / 100.0
    grid = xgcm.Grid(ds_wrfout, periodic=False)
    pressure_level_variables = ["U", "V", "geopotential", "air_potential_temperature"]

    for var in pressure_level_variables:
        processed_variables[var] = interpolate_variable(
            variable=ds_wrfout[var], air_pressure=air_pressure, grid=grid
        )

    air_temperature = mpcalc.temperature_from_potential_temperature(
        pressure=P_LEVELS * units("hPa"),
        potential_temperature=quantify_variable(
            processed_variables["air_potential_temperature"], units.kelvin
        ),
    )
    air_temperature = air_temperature.transpose("XTIME", ..., "air_pressure")

    mean_sea_level_pressure = calculate_sea_level_pressure(
        elevations=ds_wrfout.HGT,
        surface_pressure=ds_wrfout.PSFC * units("Pa"),
        air_temperature_2m=ds_wrfout["T2"],
        q_2m=quantify_variable(ds_wrfout["Q2"], units("kg/kg")),
    )

    acc_precip_6h = calculate_total_rain(ds_wrfout)

    dataset = xr.Dataset(
        {
            "geopotential": processed_variables["geopotential"],
            "u": processed_variables["U"],
            "v": processed_variables["V"],
            "t": air_temperature,
            "10u": ds_wrfout["U10"],
            "10v": ds_wrfout["U10"],
            "2t": ds_wrfout["T2"],
            "msl": mean_sea_level_pressure,
            "tp": acc_precip_6h,
        }
    )
    dataset.to_netcdf(f"{FOLDER_NAME}/wrfout_d02_processed_{args.id}.nc")
