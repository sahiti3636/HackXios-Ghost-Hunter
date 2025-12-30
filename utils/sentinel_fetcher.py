
import os
import requests
import json
from datetime import datetime
from shapely.geometry import Polygon

class SentinelFetcher:
    def __init__(self):
        self.user = os.getenv('COPERNICUS_USER')
        self.password = os.getenv('COPERNICUS_PASSWORD')
        # New OData Endpoint for Copernicus Data Space Ecosystem (CDSE)
        self.base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.api_ready = False
        
    def connect(self):
        # Simply checking if credentials exist.
        # For OData, we authenticate per request or use tokens, 
        # but basic checks are fine for this architecture.
        if self.user and self.password:
            self.api_ready = True
            print("✅ Configured for Copernicus Data Space Ecosystem (OData)")
            return True
        else:
            print("⚠️ No Copernicus credentials found. Using Mock Fetcher.")
            return False

    def search_scenes(self, polygon_coords, start_date, end_date):
        """
        Search for Sentinel-1 GRD scenes using CDSE OData API.
        """
        if self.api_ready:
            try:
                # Format dates to ISO 8601 required by OData
                full_start = self._format_date_iso(start_date)
                full_end = self._format_date_iso(end_date)
                
                # Format Polygon as WKT (without SRID prefix first, just the coords)
                if len(polygon_coords) == 2 and isinstance(polygon_coords[0], (int, float)):
                    # BBOX Optimization: Input is [min_lon, min_lat, max_lon, max_lat] or similar?
                    # No, let's just assume valid polygon list.
                    # If input is a Bounds list [minx, miny, maxx, maxy], convert to polygon.
                    # Actually standard app usage is List of [Lat, Lon].
                    wkt_poly = self._polygon_to_wkt_coords(polygon_coords)
                elif len(polygon_coords) == 5 and isinstance(polygon_coords[0], (list, tuple)):
                     # Simple box already
                     wkt_poly = self._polygon_to_wkt_coords(polygon_coords)
                else:
                     # If too complex, use Bounds to avoid 414 URI Too Long
                     if len(polygon_coords) > 20: 
                        lons = [p[1] for p in polygon_coords]
                        lats = [p[0] for p in polygon_coords]
                        min_x, max_x = min(lons), max(lons)
                        min_y, max_y = min(lats), max(lats)
                        # Create box: TL, TR, BR, BL, TL
                        # Order for WKT: Lon Lat
                        wkt_poly = f"{min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}, {min_x} {max_y}"
                     else:
                        wkt_poly = self._polygon_to_wkt_coords(polygon_coords)
                
                # Build OData Filter for CDSE
                # Example: OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(...)')
                filter_query = (
                    f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({wkt_poly}))') "
                    f"and Collection/Name eq 'SENTINEL-1' "
                    f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'GRD') "
                    f"and ContentDate/Start ge {full_start} "
                    f"and ContentDate/End le {full_end}"
                )

                # Search URL
                url = f"{self.base_url}/Products?$filter={filter_query}&$top=5&$orderby=ContentDate/Start desc"
                
                # CDSE Catalogue Search is generally open for metadata.
                # If needed, we could add basic auth: auth=(self.user, self.password)
                response = requests.get(url) 
                
                if response.status_code == 200:
                    data = response.json()
                    products = data.get('value', [])
                    print(f"🔍 Found {len(products)} real scenes via CDSE OData.")
                    
                    # Convert to our internal format for the app
                    scenes = []
                    for p in products:
                        scenes.append({
                            "type": "Feature",
                            "properties": {
                                "title": p.get('Name', 'Unknown'),
                                "id": p.get('Id'),
                                "ingestiondate": p.get('ContentDate', {}).get('Start', datetime.now().isoformat()),
                                "size": f"{p.get('ContentLength', 0) / (1024*1024):.1f} MB"
                            }
                        })
                    return scenes
                else:
                    print(f"❌ CDSE API Search failed: {response.status_code}")
                    return []
                    
            except Exception as e:
                print(f"❌ API Search Error: {e}")
                return []
        
        # MOCK FALLBACK for non-connected state
        return []

    def _format_date_iso(self, date_str):
        """Ensure date is in ISO 8601 YYYY-MM-DDTHH:MM:SS.mmmZ format for OData"""
        if not date_str or 'NOW' in date_str.upper():
             if '30DAYS' in date_str.upper():
                 from datetime import timedelta
                 return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
             return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Try to clean up YYYYMMDD to YYYY-MM-DD
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T00:00:00.000Z"
            
        return f"{date_str}T00:00:00.000Z" if 'T' not in date_str else date_str

    def _polygon_to_wkt_coords(self, coords):
        # formatted for OData: lat lon -> lon lat
        swapped = [f"{p[1]} {p[0]}" for p in coords]
        if swapped[0] != swapped[-1]:
            swapped.append(swapped[0])
        return ", ".join(swapped)
    
    def get_access_token(self):
        """
        Get Keycloak access token for downloading products.
        Docs: https://documentation.dataspace.copernicus.eu/APIs/Token.html
        """
        if not self.user or not self.password:
            return None
            
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.user,
            "password": self.password,
            "grant_type": "password",
        }
        
        try:
            r = requests.post(token_url, data=data)
            r.raise_for_status()
            return r.json()['access_token']
        except Exception as e:
            print(f"❌ Keycloak Token Error: {e}")
            return None

    def download_product(self, product_id, product_name, output_dir):
        """
        Download product using OData $value endpoint.
        """
        token = self.get_access_token()
        if not token:
            print("❌ Cannot download: No access token.")
            return None
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{product_name}.zip")
        
        # Don't re-download if exists
        if os.path.exists(file_path):
            print(f"✅ File already exists: {file_path}")
            return file_path
            
        # Create a session to manage headers and redirects
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        
        # Define the download URL
        url = f"{self.base_url}/Products({product_id})/$value"
        
        print(f"⬇️ Starting download for {product_name}...")
        try:
            # We explicitly allow redirects. Requests handles redirects but 
            # might strip headers if crossing domains securely.
            # For CDSE, we know it redirects to 'zipper' or 'download'.
            # A Session usually preserves headers if domains match policy, 
            # but let's be robust and follow manually if needed or trust Session.
            
            # Using session.get with stream=True
            with session.get(url, stream=True, allow_redirects=True) as r:
                if r.status_code == 401:
                    # If redirected and auth failed, try re-adding header to the final URL
                    # sometimes needed if domain swapped completely
                    print("⚠️ Redirect 401: Retrying with explicit auth on final URL...")
                    # Re-create simple headers for the requests.get call
                    auth_headers = {"Authorization": f"Bearer {token}"}
                    with requests.get(r.url, headers=auth_headers, stream=True) as r2:
                         r2.raise_for_status()
                         self._stream_to_file(r2, file_path)
                else:
                    r.raise_for_status()
                    self._stream_to_file(r, file_path)
                             
            print(f"✅ Download complete: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"❌ Download Failed: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path) # Cleanup partial
                except: pass
            return None

    def _stream_to_file(self, response, file_path):
        """Helper to write stream to file with progress"""
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (50 * 1024 * 1024) == 0: 
                        print(f"   ... {downloaded / (1024*1024):.0f} MB downloaded")

    @property
    def api(self):
        # Compatibility property for app.py checking "if fetcher.api:"
        return self.api_ready
