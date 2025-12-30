import os
import requests
from datetime import datetime
from typing import List, Dict, Optional

class AISFetcher:
    """
    Fetches real-time AIS data from external providers (Spire, MarineTraffic, AISStream).
    """
    
    def __init__(self, provider="spire"):
        self.provider = provider
        self.api_key = os.getenv("AIS_API_KEY")
        
    def fetch_vessels_in_bbox(self, min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
        """
        Fetch vessels within a bounding box.
        """
        if not self.api_key:
            print("⚠️ AIS_API_KEY not set. Using mock data.")
            return []

        if self.provider == "spire":
            return self._fetch_spire(min_lat, min_lon, max_lat, max_lon)
        elif self.provider == "marinetraffic":
            return self._fetch_marinetraffic(min_lat, min_lon, max_lat, max_lon)
        elif self.provider == "datalastic":
            return self._fetch_datalastic(min_lat, min_lon, max_lat, max_lon)
        elif self.provider == "aisstream":
             return self._fetch_aisstream(min_lat, min_lon, max_lat, max_lon)
        else:
            print(f"❌ Unknown AIS provider: {self.provider}")
            return []

    def _fetch_aisstream(self, min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
        """
        Fetch from AISStream.io (Free WebSocket API).
        Connects, subscribes to bbox, listens for 10s, and returns unique vessels.
        """
        import asyncio
        import json
        import websockets
        
        vessels = {}
        
        async def listen():
            uri = "wss://stream.aisstream.io/v0/stream"
            try:
                async with websockets.connect(uri) as websocket:
                    subscribe = {
                        "APIKey": self.api_key,
                        "BoundingBoxes": [[[min_lat, min_lon], [max_lat, max_lon]]],
                        # "FiltersShipMMSI": [], # Optional, remove if unused
                        "FilterMessageTypes": ["PositionReport"]
                    }
                    await websocket.send(json.dumps(subscribe))
                    
                    print(f"📡 Connected to AISStream.io. Listening for 10s in bbox...")
                    
                    # Listen for a short window
                    try:
                        while True:
                            message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            msg_json = json.loads(message)
                            
                            # Debug: Print first message type to confirm flow
                            # print(f"DEBUG: Msg Type: {msg_json.get('MessageType')}")
                            
                            if "Message" in msg_json and "PositionReport" in msg_json["Message"]:
                                report = msg_json["Message"]["PositionReport"]
                                mmsi = report["UserID"]
                                vessels[mmsi] = {
                                    'mmsi': mmsi,
                                    'name': f"Unknown-{mmsi}", 
                                    'lat': report["Latitude"],
                                    'lon': report["Longitude"],
                                    'timestamp': msg_json["MetaData"]["time_utc"],
                                    'status': report.get("NavigationalStatus", "Unknown")
                                }
                    except asyncio.TimeoutError:
                        print("⏱️ Time limit reached.")
            except Exception as e:
                print(f"❌ AISStream Error: {e}")

        # Run the async listener synchronously
        try:
             asyncio.run(listen())
             print(f"✅ Found {len(vessels)} unique vessels via AISStream.")
             return list(vessels.values())
        except Exception as e:
             print(f"❌ Async Execution Failed: {e}")
             return []

    def _fetch_datalastic(self, min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
        """
        Fetch from DataLastic API (Free Tier available).
        Docs: https://datalastic.com/api-docs/
        """
        url = "https://api.datalastic.com/api/v0/vessels"
        params = {
            "api-key": self.api_key,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "days": 1 # Last 24h
        }
        
        try:
            print(f"📡 Querying DataLastic API: {url}")
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                vessels = []
                for v in data.get('data', []):
                    # Normalized Vessel Object
                    vessels.append({
                        'mmsi': v.get('mmsi'),
                        'name': v.get('name'),
                        'lat': float(v.get('lat', 0)),
                        'lon': float(v.get('lon', 0)),
                        'timestamp': v.get('last_position_epoch'), # Epoch time
                        'status': v.get('nav_status', 'Unknown')
                    })
                print(f"✅ Found {len(vessels)} vessels via DataLastic.")
                return vessels
            else:
                print(f"❌ DataLastic API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ DataLastic Connection Failed: {e}")
            return []

    def _fetch_spire(self, min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
        """
        Fetch from Spire Maritime API (Vessels API).
        Docs: https://documentation.spire.com/
        """
        url = "https://api.spire.com/vessels/v2/vessels"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Spire specific bbox format or filtering
        # This is a simplified example
        params = {
            "vessel_type": "fishing", # Focus on fishing for this use case
            "last_pos_lat_min": min_lat,
            "last_pos_lat_max": max_lat,
            "last_pos_lon_min": min_lon,
            "last_pos_lon_max": max_lon
        }
        
        try:
            print(f"📡 Querying Spire AIS API: {url}")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                vessels = []
                for v in data.get('data', []):
                    # Normalized Vessel Object
                    vessels.append({
                        'mmsi': v.get('mmsi'),
                        'name': v.get('name'),
                        'lat': v.get('last_known_position', {}).get('geometry', {}).get('coordinates', [0,0])[1],
                        'lon': v.get('last_known_position', {}).get('geometry', {}).get('coordinates', [0,0])[0],
                        'timestamp': v.get('last_known_position', {}).get('timestamp'),
                        'status': v.get('navigational_status', 'Unknown')
                    })
                print(f"✅ Found {len(vessels)} vessels via Spire.")
                return vessels
            else:
                print(f"❌ Spire API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ Spire Connection Failed: {e}")
            return []

    def _fetch_marinetraffic(self, min_lat, min_lon, max_lat, max_lon) -> List[Dict]:
        """
        Fetch from MarineTraffic API.
        """
        # Placeholder for MT implementation
        print("⚠️ MarineTraffic integration requires specific API license level.")
        return []

    def check_ais_match(self, detected_lat, detected_lon, timestamp, tolerance_km=1.0) -> Optional[Dict]:
        """
        Check if a detection matches any known AIS signal.
        """
        # Logic to be improved with temporal interpolation
        return None
