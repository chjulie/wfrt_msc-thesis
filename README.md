# Data-driven weather models for precipitation forecasting

This repository contains the code developed to compare and evaluate the performance of data-driven weather models. 
The aim of this thesis is the assessment of the accuracy and reliability of high-resolution data-driven models. This specific
work focuses on British Columbia, Canada.

For the actual regional data-driven model can be found in this repository: https://github.com/b0ws3r/anemoi-wfrt.

## Models
We compare the following models:
- **aifs-v1** : low-resolution global data-driven model
- **wfrt-anemoi** : high-resolution regional data-driven model (British Columbia)
- ... : high-resolution Numerical Weather Prediction (NWP) model 

## Evaluation methods
### Evaluation against observation
Model performance are first compared againts observation data. Hourly stations observations are publicly released by _Environment and Climate Change Canada_ (ECCC) and provided by _BC Hydro_.

- technical specifications of measurements (quality)
- number and location of stations
- evaluation metrics

### Evaluation against gridded reanalysis
A gridded reanalysis dataset is used as well to assess the models. We use the _Climatex_ dataset, a dynamically downscaled version of ERA5 ! add citation.

- broad technical specs of dataset (details in paper)
- relevance of the evaluation with gridded reanalysis
- evaluation metrics

## How to use this repository
### Evaluation against observations

- ```src/evaluate.py```: Wrapper for evaluation against observations, calls the right evaluator object for a given model.
- ```src/utils/model_forecast_evaluator```: Implements the ```ModelEvaluator``` class and its children. Important class methods are :


| Method    | Description                | Comment         |
|-----------|----------------------------|-----------------|
| `evaluate()`  | Iterates over datetimes and variables, calls other method to compute the loss metric.  |        |
| `get_station_observations()`  | Reads the dataframe containing the observations and return their value at each location.   | Observations path is defined in ```ModelEvaluator.__init__()```. This may need to be changed.  |  
| `get_prediction_at_station_loc()`    | Computes model prediction at the location of the observation (using the nearest neighbor). |          |
| `rmse()` | Implements the chosen loss metric.     |         |
| `compute_coordinates()` | Extract the model's coordinates system to be used in `get_prediction_at_station_loc()`. |         |
|`rclone_copy()`  | Download WRF model prediction from Nextcloud | Only for `RegNWPModelEvaluator`, !! path may need to be updated !! |
|`save_error_df()`| Saves the dataframe containing the error metric as a .csv file | !! Saving path needs to be changed (currently points to my scratch) |


*Additional considerations*
- For deep learning based models, inference must be performed beforehand. The path of the inference results is defined in `DLModelEvaluator.__init__()`. This may need to be changed.
- Depending on if we want to evaluate the model at a 6-hourly or daily resolution, the `date_range` stills need to be manually modified in ```evaluate.py```. 
- The lead time up to which the model is evaluated is defined in ```src/utils/data_constant```(`EVAL_LEAD_TIMES`). In the future, this could be parsed as an argument.


### Evaluation against gridded data reanalysis


## Acknowledgments
- **European Center for Medium-range Weather Forecasting, ECMWF** - aifs-v1 model and open analysis data
- **Environment and Climate Change Canada, ECCC** - publicly available observations
- **Digital Research Alliance, DRA** - _Fir_ computing cluster
- **UBC Weather Forecast Research Team, WFRT** - regional dataset and expertise