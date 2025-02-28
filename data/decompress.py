import zipfile
from src.utils import PATH

paths = {
    "microdebris": PATH.RAW_DATA / "Microdebris Data Jan 21 to Dec 24.zip",
    "marine_plastic_pollution": PATH.RAW_DATA / "Marine Plastic Pollution Data.zip",
}

for path in paths.values():
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(PATH.RAW_DATA)