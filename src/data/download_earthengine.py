import ee 
import sys
sys.path.append('../')

from data_constants import DOMAIN_MINX, DOMAIN_MAXX, DOMAIN_MINY, DOMAIN_MAXY

ee.Authenticate()
ee.Initialize(project='msc-era5')
print(ee.String('Hello from the Earth Engine servers!').getInfo())

regional_domain = ee.Geometry.Rectangle([DOMAIN_MINX, DOMAIN_MINY, DOMAIN_MAXX, DOMAIN_MAXY])
print(f"geometry: {type(regional_domain)}")

scale = 2500    # output resolution in meter
dataset = "projects/climate-engine-pro/assets/ce-hrdpa-daily" #"ECMWF/NRT_FORECAST/IFS/OPER" #'ECMWF/ERA5/HOURLY'
bands = ["total_precipitation"]

i_date = '2023-01-01'
f_date = '2025-01-01'

collection = ee.ImageCollection(dataset).filterDate(i_date, f_date).filterBounds(regional_domain) #.select(bands)
collection_data = collection.getInfo()
print(f"collection: {type(collection_data)}, {collection_data.keys()}")

for k in collection_data.keys():
    if k == "properties":
        print(f" - {k}: dict with keys: {collection_data[k].keys()}")
    elif k == "features":
        print(f" - {k}: list of len: {len(collection_data[k])}: {collection_data[k][700].keys()}")
    else: 
        print(f" - {k}: {collection_data[k]}")

request = {
    'assetId': 'NDVI', #collection_data['id'],
    'fileFormat': 'NUMPY_NDARRAY',
    'region': regional_domain.getInfo(),
}

collection_array = ee.data.getPixels(request)
print(f" * collection_array: {type(collection_array)}, {collection_array.shape}")

# collection_size = collection.size().getInfo()
# collection_ls = collection.toList(collection_size)
# print(f"size: {collection_size}")
# array_data = data.toDictionary()


