#%% Imports

from pyhdf.SD import SD, SDC

from src.utils import PATH

#%% Set up the HDF file path
hdf_file_path = str(PATH.DATA / "external" / "modis" / "MYD09GA.A2012153.h28v13.061.2021209214634.hdf")

#%% Open and read HDF file
hdf = SD(hdf_file_path, SDC.READ)

#%% List contents
print("\n=== FILE ATTRIBUTES ===")
attrs = hdf.attributes()
for attr_name in attrs.keys():
    print(f"{attr_name}: {attrs[attr_name]}")

print("\n=== DATASETS ===")
datasets = hdf.datasets()
for idx, (name, info) in enumerate(datasets.items()):
    dims = info[0]
    type_code = info[1]
    print(f"{idx+1}. {name}: shape={dims}, type={type_code}")

#%% Close the file
hdf.end()
