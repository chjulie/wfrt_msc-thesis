import datetime
import os
import sys
import numpy as np
import torch
import argparse
from collections import defaultdict
from netCDF4 import Dataset
import pickle

import earthkit.data as ekd
import earthkit.regrid as ekr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.models.layers.processor import TransformerProcessor
from ecmwf.opendata import Client as OpendataClient

sys.path.append('../')
from data_constants import DOMAIN_MINX, DOMAIN_MAXX, DOMAIN_MINY, DOMAIN_MAXY, PRED_DATA_DIR

# Create dummy flash_attn package and submodule
import sys
import types

# --- Create dummy flash_attn package ---
flash_attn = types.ModuleType('flash_attn')
flash_attn_interface = types.ModuleType('flash_attn_interface')

# Dummy function to satisfy checkpoint
def flash_attn_func(*args, **kwargs):
    raise RuntimeError("This is a dummy flash_attn_func. Should not be called during inference with replaced processor.")

flash_attn_interface.flash_attn_func = flash_attn_func

# Register modules
flash_attn.flash_attn_interface = flash_attn_interface
sys.modules['flash_attn'] = flash_attn
sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface

# --- Definition of constants ---
# SCRIPT CONSTANT
EXPERIENCE = "inference_aifs_single-v1"

# INPUT VARIABLE
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
# msl: Mean sea level pressure
# skt: Skin temperature
# sp: Surface pressure
# tcw: Total column vertically-integrated water vapour
# lsm: Land Sea Mask
# z: Geopotential
# slor: Slope of sub-gridscale orography (step 0)
# sdor: Standard deviation of sub-gridscale orography (step 0)
PARAM_SOIL =["vsw","sot"]
# vsw: Volumetric soil water (layers 1-4)
# sot: Soil temperature (layers 1-4)
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
# q: Specific humidity
# w: vertical velocity
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1,2]

def get_open_data(param, input_date, levelist=[]):
    fields = defaultdict(list)
    # Get data at time t and t-1:
    for date in [input_date - datetime.timedelta(hours=6), input_date]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist) # <class 'earthkit.data.readers.grib.file.GRIBReader'>
        for f in data:  # <class 'earthkit.data.readers.grib.codes.GribField'>
            assert f.to_numpy().shape == (721,1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25°x0.25° (regular lat-lon grid, 2D) to N320 (reduced gaussian grid, 1D, see definition here: https://www.ecmwf.int/en/forecasts/documentation-and-support/gaussian_n320) 
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields

def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)


def save_state(state, path):
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--date", type=str, required=True, help="Date in YYYY-mm-dd format. Must be < 1 month old."
    )
    # parser.add_argument(
    #     "--field", type=str, required=True, help="'temp'"
    # )
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date + 'T06:00:00', '%Y-%m-%dT%H:%M:%S') # time should be 06,12,18 or 00.
    print(" ** IDATE: ", type(date))
    # tdate = OpendataClient().latest()
    # print(" ** DATE: ", type(tdate))

    # Create necessary dir
    os.makedirs(PRED_RES_DIR, exist_ok=True)

    ## Import initial conditions from ECMWF Open Data
    fields = {}
    fields.update(get_open_data(param=PARAM_SFC, input_date=date))

    fields.update(get_open_data(param=PARAM_PL, input_date=date, levelist=LEVELS))
    # Convert geopotential height into geopotential (transform GH to Z)
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665
        
    soil=get_open_data(param=PARAM_SOIL, input_date=date, levelist=SOIL_LEVELS)

    # soil parameters need to be renamed to be consistent with training
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
            'vsw_1': 'swvl1','vsw_2': 'swvl2'}
    for k,v in soil.items():
        fields[mapping[k]]=v

    print(" > data downloaded! ")

    # Create initial state
    input_state = dict(date=date, fields=fields)

    ## Load the model and run the forecast
    checkpoint = {"huggingface":"ecmwf/aifs-single-1.0"}
    print(' > CUDA availability: ', torch.cuda.is_available())

    # Modify model to NOT use flash-attn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../../data/prediction_data/aifs-single-mse-1.0.ckpt", map_location=device, weights_only=False).to(device)
    model.model.processor = TransformerProcessor(
        num_layers=16,
        window_size=1024,
        num_channels=1024,
        num_chunks=2,
        activation='GELU',
        num_heads=16,
        mlp_hidden_ratio=4,
        dropout_p=0.0,
        attention_implementation="scaled_dot_product_attention").to(device)

    print(" > Model modified to use 'scaled_dot_product_attention'.")
    runner = SimpleRunner(checkpoint, device="cuda")
    runner.model = model

    # Run the forecast
    for state in runner.run(input_state=input_state, lead_time=12):
        print(state.keys())
        print_state(state)

    ## Plot generation
    DISP_VAR = "2t"
    latitudes = state["latitudes"]
    longitudes = state["longitudes"]
    fixed_longitudes = fix(longitudes)
    values = state["fields"][DISP_VAR]

    # print(' -- values --')
    # print(' - type: ', type(values))
    # print(' - shape: ', values.shape)
    # print('\n')

    # print(' -- longitudes --')
    # print('- type: ', type(longitudes))
    # print('- shape: ', longitudes.shape)
    # print('- len: ', len(longitudes))
    # print('- [0]: ', longitudes[0])
    # print('- [-1]: ', longitudes[-1])
    # print('- res: ', longitudes[1] - longitudes[0])
    # print('\n')

    # print(' -- fixed longitudes --')
    # print('- type: ', type(fixed_longitudes))
    # print('- shape: ', fixed_longitudes.shape)
    # print('- len: ', len(fixed_longitudes))
    # print('- [0]: ', fixed_longitudes[0])
    # print('- [-1]: ', fixed_longitudes[-1])
    # print('- res: ', fixed_longitudes[1] - fixed_longitudes[0])
    # print('\n')

    # print(' -- latitudes --')
    # print('- type: ', type(latitudes))
    # print('- shape: ', latitudes.shape)
    # print('- len: ', len(latitudes))
    # print('- [0]: ', latitudes[0])
    # print('- [-1]: ', latitudes[-1])
    # print('- res: ', latitudes[1] - latitudes[0])
    # print('\n')

    domain_mask = ((fixed_longitudes >= DOMAIN_MINX) & (fixed_longitudes <= DOMAIN_MAXX) &
        (latitudes >= DOMAIN_MINY) & (latitudes <= DOMAIN_MAXY))
    domain_lon = fixed_longitudes[domain_mask]
    domain_lat = latitudes[domain_mask]
    domain_values = values[domain_mask]
    # global_data = [state["fields"][sub_key] for sub_key in state["fields"].keys()]
    # domain_fields = dict(zip(state["fields"].keys(), [data[domain_mask] for data in global_data]))
    domain_fields = dict(zip(state["fields"].keys(), [data[domain_mask] for data in state["fields"].values()]))
    print(f" ** STATE[FIELDS]: {type(state["fields"])}, keys= {state["fields"].keys()}, val= {state["fields"].values()}") #, shape = {state["fields"].shape}, [0] = {state["fields"][0]}")
    # domain_fields = state["fields"][:,domain_mask]

    domain_state = {
        "date": state["date"],
        "fields": domain_fields,
        "latitudes": domain_lat,
        "longitudes": domain_lon
    }

    ## Save predicted domain_state to .pkl file
    state["longitudes"] = fixed_longitudes
    save_state(domain_state, f"{PRED_DATA_DIR}/{date.strftime(format='%Y%m%d')}_regional_state.pkl")
    save_state(state, f"{PRED_DATA_DIR}/{date.strftime(format='%Y%m%d')}_global_state.pkl")

    # print(' -- domain longitudes --')
    # print('- type: ', type(domain_lon))
    # print('- shape: ', domain_lon.shape)
    # print('- len: ', len(domain_lon))
    # print('- [0]: ', domain_lon[0])
    # print('- [-1]: ', domain_lon[-1])
    # print('- res: ', domain_lon[1] - domain_lon[0])
    # print('\n')

    # print(' -- domain latitudes --')
    # print('- type: ', type(domain_lat))
    # print('- shape: ', domain_lat.shape)
    # print('- len: ', len(domain_lat))
    # print('- [0]: ', domain_lat[0])
    # print('- [-1]: ', domain_lat[-1])
    # print('- res: ', domain_lat[1] - domain_lat[0])
    # print('\n')

    # fig, ax = plt.subplots(1,2,figsize=(11, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    # # global domain
    # ax[0].coastlines()
    # ax[0].add_feature(cfeature.BORDERS, linestyle=":")

    # triangulation = tri.Triangulation(fix(longitudes), latitudes)

    # contour=ax[0].tricontourf(triangulation, values, levels=20, transform=ccrs.PlateCarree(), cmap="RdBu")
    # cbar = fig.colorbar(contour, ax=ax[0], orientation="vertical", shrink=0.7, label=f"{DISP_VAR}")

    # # regional domain
    # ax[1].coastlines()
    # ax[1].add_feature(cfeature.BORDERS, linestyle=":")

    # domain_triangulation = tri.Triangulation(domain_lon, domain_lat)

    # contour=ax[1].tricontourf(domain_triangulation, domain_values, levels=20, transform=ccrs.PlateCarree(), cmap="RdBu")
    # cbar = fig.colorbar(contour, ax=ax[1], orientation="vertical", shrink=0.7, label=f"{DISP_VAR}")

    # fig.suptitle("Temperature at {}".format(state["date"]))
    # plt.savefig(os.path.join(PRED_RES_DIR, f"{EXPERIENCE}_{DISP_VAR}_{date.strftime(format='%Y%m%d_%H:%M:%S')}"), )

    print(" > Program finished successfully!")

