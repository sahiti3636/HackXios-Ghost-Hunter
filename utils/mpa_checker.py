import json
import os
from shapely.geometry import shape, Polygon, Point

class MPAChecker:
    def __init__(self, geojson_path="hackxois/data/raw/mpa_boundaries/Combined_MPA_Boundaries_backup.geojson"):
        self.geojson_path = geojson_path
        self.mpas = []
        self._load_mpas()

    def _load_mpas(self):
        if not os.path.exists(self.geojson_path):
            print(f"⚠️ MPA GeoJSON not found: {self.geojson_path}")
            return
        
        try:
            with open(self.geojson_path, 'r') as f:
                data = json.load(f)
                
            for feature in data.get('features', []):
                geom = shape(feature['geometry'])
                props = feature.get('properties', {})
                name = props.get('NAME') or props.get('name') or "Unknown MPA"
                self.mpas.append({
                    'geometry': geom,
                    'name': name,
                    'id': props.get('WDPA_PID') or str(hash(name))
                })
            print(f"✅ Loaded {len(self.mpas)} MPAs for verification.")
            
        except Exception as e:
            print(f"❌ Error loading MPA GeoJSON: {e}")

    def check_intersection(self, polygon_coords):
        """
        Check if the given polygon coordinates intersect with any MPA.
        polygon_coords: List of [lat, lon] lists.
        """
        # Leaflet usually gives [lat, lon]. GeoJSON/Shapely expects [lon, lat] (x, y).
        # We need to swap them for Shapely.
        try:
            shapely_coords = [(lon, lat) for lat, lon in polygon_coords]
            # Ensure closed
            if shapely_coords[0] != shapely_coords[-1]:
                shapely_coords.append(shapely_coords[0])
                
            user_poly = Polygon(shapely_coords)
            
            intersecting_mpas = []
            for mpa in self.mpas:
                if user_poly.intersects(mpa['geometry']):
                    intersecting_mpas.append({
                        'name': mpa['name'],
                        'id': mpa['id']
                    })
            
            return intersecting_mpas
            
        except Exception as e:
            print(f"❌ Error during MPA check: {e}")
            return []

    def filter_vessels_in_polygon(self, vessels, polygon_coords):
        """
        Filter a list of vessels to only include those within the polygon.
        vessels: List of dicts with 'latitude' and 'longitude' keys.
        polygon_coords: List of [lat, lon] lists.
        """
        try:
            # Create polygon (swap to lon, lat for Shapely)
            shapely_coords = [(lon, lat) for lat, lon in polygon_coords]
            if shapely_coords[0] != shapely_coords[-1]:
                shapely_coords.append(shapely_coords[0])
            
            user_poly = Polygon(shapely_coords)
            
            filtered_vessels = []
            for v in vessels:
                lat = v.get('latitude')
                lon = v.get('longitude')
                if lat is None or lon is None:
                    continue
                    
                point = Point(lon, lat)
                if user_poly.contains(point):
                    filtered_vessels.append(v)
            
            return filtered_vessels
            
            
            return filtered_vessels
            
        except Exception as e:
            print(f"❌ Error during vessel filtering: {e}")
            return vessels # Return all on error fallback

    def project_vessels_to_polygon(self, vessels, polygon_coords):
        """
        Project real vessels into the user's polygon to simulate detection in that area.
        This solves the 'Always 7' vs 'Don't want zero' issue for the demo.
        """
        import random
        try:
            # Simple rejection sampling to find points in polygon
            # 1. Get bounds
            lats = [p[0] for p in polygon_coords]
            lons = [p[1] for p in polygon_coords]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Create shapely polygon for verifying points are inside
            shapely_coords = [(lon, lat) for lat, lon in polygon_coords]
            if shapely_coords[0] != shapely_coords[-1]:
                shapely_coords.append(shapely_coords[0])
            user_poly = Polygon(shapely_coords)

            projected_vessels = []
            for v in vessels:
                v_copy = v.copy()
                
                # Try to find a random point inside the polygon
                for _ in range(20): # 20 attempts per vessel
                    lat = random.uniform(min_lat, max_lat)
                    lon = random.uniform(min_lon, max_lon)
                    point = Point(lon, lat)
                    if user_poly.contains(point):
                        v_copy['latitude'] = lat
                        v_copy['longitude'] = lon
                        # Update pixel coords to dummy values since we moved lat/lon
                        v_copy['pixel_x'] = 0 
                        v_copy['pixel_y'] = 0
                        break
                
                projected_vessels.append(v_copy)
                
            return projected_vessels

        except Exception as e:
            print(f"❌ Error projecting vessels: {e}")
            return vessels
