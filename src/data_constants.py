import numpy as np

# directories
# OBS_DATA_DIR = "/Users/Julie/Desktop/wfrt_msc-thesis/data/eccc_data"
# PRED_DATA_DIR = "/Users/Julie/Desktop/wfrt_msc-thesis/data/prediction_data"
ERROR_DATA_DIR = "data/error_data"
OBS_DATA_DIR = "data/eccc_data"
PRED_DATA_DIR = "data/prediction_data"

PRED_PLOT_DIR = "reports/plots/prediction"
GEOM_RES_DIR = "../../reports/plots/geometry"

# DOMAIN BOUNDS (Climatex bounds)
DOMAIN_MINX = -146.74888611
DOMAIN_MINY = 43.16402817
DOMAIN_MAXX = -108.88935089
DOMAIN_MAXY = 66.57196045

# REQUESTED FIELD
UNIVERSAL_FIELDS = "ID%2CSTN_ID%2CUTC_DATE%2C"
VERIF_FIELDS = f"{UNIVERSAL_FIELDS}TEMP%2CPRECIP_AMOUNT%2CWIND_SPEED%2CWIND_DIRECTION"

# TRAIN-TEST SPLIT
TEST_YEAR = "2023"

# SCORECARD
P_LEVELS = np.array([50, 100, 250, 500, 850])
