import math
from itertools import combinations

# ==========================================
# CONFIGURATION
# ==========================================
PROXIMITY_THRESHOLD_KM = 2.0  # Km distance to consider "close" (Cargo Exchange)
SMALL_VESSEL_PIXEL_LIMIT = 300 # Pixels (Approx size threshold for small boats)
COLLINEARITY_TOLERANCE_KM = 1.0 # Max deviation from line fitting

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371  # Radius of earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def analyze_behavior(vessels_list):
    """
    Analyzes a list of verified vessels for suspicious behaviors.
    
    Args:
        vessels_list (list): List of dicts, each containing:
            - 'image' (id)
            - 'location_data': {'latitude': float, 'longitude': float}
            - 'vessel_size_pixels': int
            - 'ais_verification': {'status': str} (Optional but helpful)
            
    Returns:
        list: The same list with an appended 'behavior_analysis' dict for each vessel.
    """
    
    # Initialize behavior field
    for v in vessels_list:
        v['behavior_analysis'] = {
            'suspicion_level': 'LOW',
            'flags': []
        }

    # 1. INDIVIDUAL CHECKS
    for v in vessels_list:
        size = v.get('vessel_size_pixels', 0)
        
        # Check for Small Ship (if isolated, likely legal fishing, but we mark the property)
        if size > 0 and size < SMALL_VESSEL_PIXEL_LIMIT:
             v['behavior_analysis']['flags'].append("SMALL_VESSEL")

    # 2. PAIRWISE CHECKS (Cargo Exchange / Proximity)
    # Only verify vessels that exist (status YES)
    active_vessels = [v for v in vessels_list if v.get('is_vessel_verified') == 'YES']
    
    # Create a map to easily update the original list objects
    # (Since list contains mutable dicts, updates to active_vessels elements reflect in vessels_list)
    
    pairs = list(combinations(active_vessels, 2))
    for v1, v2 in pairs:
        lat1 = v1['location_data']['latitude']
        lon1 = v1['location_data']['longitude']
        lat2 = v2['location_data']['latitude']
        lon2 = v2['location_data']['longitude']
        
        dist = haversine(lat1, lon1, lat2, lon2)
        
        if dist < PROXIMITY_THRESHOLD_KM:
            msg = f"PROXIMITY_ALERT (Dist: {dist:.2f}km with {v2['image']})"
            v1['behavior_analysis']['flags'].append(msg)
            v1['behavior_analysis']['suspicion_level'] = 'HIGH'
            
            msg2 = f"PROXIMITY_ALERT (Dist: {dist:.2f}km with {v1['image']})"
            v2['behavior_analysis']['flags'].append(msg2)
            v2['behavior_analysis']['suspicion_level'] = 'HIGH'

    # 3. GROUP CHECKS (Fleet / Collinearity) - "More than 3 ships in a line"
    if len(active_vessels) >= 3:
        triplets = list(combinations(active_vessels, 3))
        for triplet in triplets:
            # Check linearity
            # Simple Regression approach: Check correlation of Lat/Lon or check distance from line
            lats = [t['location_data']['latitude'] for t in triplet]
            lons = [t['location_data']['longitude'] for t in triplet]
            
            # Use simple linear regression (predict lat from lon or vice versa)
            # If all 3 points fall near the line, it's collinear.
            # Using triangle area method (Area = 0 -> Collinear)
            # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            # We approximate with lat/lon directly (good enough for local scale formation)
            
            x1, y1 = lons[0], lats[0]
            x2, y2 = lons[1], lats[1]
            x3, y3 = lons[2], lats[2]
            
            area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            
            # Heuristic: If area is very small relative to the spread, they are in a line.
            # Using a simplified threshold here.
            # Ideally convert to meters, but degree approximation: 0.01 deg^2 ~ 1 km^2 area approx at equator
            if area < 0.005: 
                for v in triplet:
                    if "FLEET_FORMATION_DETECTED" not in v['behavior_analysis']['flags']:
                        v['behavior_analysis']['flags'].append("FLEET_FORMATION_DETECTED")
                        v['behavior_analysis']['suspicion_level'] = 'CRITICAL'

    # 4. CONTEXTUAL LOGIC (Small ship in middle of sea = Legal Fishing?)
    for v in active_vessels:
        flags = v['behavior_analysis']['flags']
        
        # If it's small and has NO proximity/fleet alerts -> "Likely Legal Fishing"
        is_small = "SMALL_VESSEL" in flags
        has_partners = any("PROXIMITY" in f or "FLEET" in f for f in flags)
        
        if is_small and not has_partners:
            v['behavior_analysis']['flags'].append("ISOLATED_ACTIVITY (Likely Legal)")
            # If it was marked High for some other reason, we might downgrade, but keep safe for now
            if v['behavior_analysis']['suspicion_level'] == 'LOW':
                v['behavior_analysis']['suspicion_level'] = 'LOW (Verified)'

    return vessels_list
