import os
import rasterio
from rasterio.vrt import WarpedVRT
import numpy as np
from PIL import Image

def generate_web_image(measurement_path, output_path):
    print(f"Generating image from: {measurement_path}")
    try:
        with rasterio.open(measurement_path) as src:
            with WarpedVRT(src, crs='EPSG:4326') as vrt:
                bounds = [[vrt.bounds.bottom, vrt.bounds.left], [vrt.bounds.top, vrt.bounds.right]]
                print(f"Calculated Bounds (Lat/Lon): {bounds}")
                
                dst_width = 1000
                scale = dst_width / vrt.width
                dst_height = int(vrt.height * scale)
                
                data = vrt.read(1, out_shape=(1, dst_height, dst_width))
                
                p2 = np.percentile(data, 2)
                p98 = np.percentile(data, 98)
                data = np.clip(data, p2, p98)
                if p98 > p2:
                    data = (data - p2) / (p98 - p2) * 255.0
                else:
                    data = data * 0
                data = data.astype(np.uint8)
                
                Image.fromarray(data).save(output_path)
                print(f"✅ Saved to {output_path}")
                return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test on the latest file found
target_file = "data/raw/satellite/S1C_IW_GRDH_1SDV_20251230T043952_20251230T044017_005676_00B568_561F.SAFE/measurement/s1c-iw-grd-vv-20251230t043952-20251230t044017-005676-00b568-001.tiff"
output_test = "test_output_map.png"

generate_web_image(target_file, output_test)
