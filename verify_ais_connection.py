from utils.ais_fetcher import AISFetcher
import os
from dotenv import load_dotenv

# Load env to ensure key is available
load_dotenv()

print("🔍 Testing AISStream Connection...")
print(f"Provider: {os.getenv('AIS_PROVIDER')}")
print(f"Key Present: {bool(os.getenv('AIS_API_KEY'))}")

fetcher = AISFetcher(provider='aisstream')

# unexpected bounds format handling might be needed if I copy paste directly, 
# but fetcher takes min_lat, min_lon, max_lat, max_lon
# Using high traffic area (Dover Strait) to verify data flow
min_lat, min_lon = 51.0, 1.2
max_lat, max_lon = 51.2, 1.6

print(f"📍 Querying BBox: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")

vessels = fetcher.fetch_vessels_in_bbox(min_lat, min_lon, max_lat, max_lon)

if vessels:
    print(f"\n✅ SUCCESS: Found {len(vessels)} vessels.")
    for v in vessels[:3]: # Show first 3
        print(f" - {v}")
else:
    print("\n⚠️  No vessels found or connection failed. Check logs above.")
