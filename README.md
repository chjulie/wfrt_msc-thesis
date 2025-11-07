# Data-driven weather models for precipitation forecasting

This repository contains the code developed to compare and evaluate the performance of data-driven weather models. 
The aim of this thesis is the assessment of the accuracy and reliability of high-resolution data-driven models. This specific
work focuses on British Columbia, Canada.

For the actual regional data-driven model can be found in this repository: https://github.com/b0ws3r/anemoi-wfrt.

## Models
We compare the following models:
- AIFS-v1 : low-resolution global data-driven model
- wfrt-anemoi : high-resolution regional data-driven model (British Columbia)
- ... : high-resolution Numerical Weather Prediction (NWP) model 

## Evaluation methods
### Evaluation against observation
Model performance are first compared againts observation data. Hourly stations observations are publicly released by _Environment and Climate Change Canada_ (ECCC).

- technical specifications of measurements (quality)
- number and location of stations
- evaluation metrics

### Evaluation against gridded reanalysis
A gridded reanalysis dataset is used as well to assess the models. We use the _Climatex_ dataset, a dynamically downscaled version of ERA5 ! add citation.

- broad technical specs of dataset (details in paper)
- relevance of the evaluation with gridded reanalysis
- evaluation metrics

## How to use this repository

## Acknowledgments