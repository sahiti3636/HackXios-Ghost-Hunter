import rasterio

path = "data/raw/satellite/S1C_IW_GRDH_1SDV_20251230T043952_20251230T044017_005676_00B568_561F.SAFE/measurement/s1c-iw-grd-vv-20251230t043952-20251230t044017-005676-00b568-001.tiff"
try:
    with rasterio.open(path) as src:
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Transform: {src.transform}")
        print(f"GCPs: {len(src.gcps[0])} GCPs found if any")
        if src.gcps[0]:
            print(f"GCP CRS: {src.gcps[1]}")
except Exception as e:
    print(f"Error: {e}")
