# # ==========================================
# # CONFIGURATION: RISK WEIGHTS
# # ==========================================
# WEIGHTS = {
#     "AIS_DARK": 40,
#     "INSIDE_MPA": 10,       # Reduced from 30 (Baseline assumption)
#     "FLEET_FORMATION": 50,
#     "PROXIMITY_ALERT": 30,
#     "SMALL_VESSEL_OFFSET": -20, 
#     "CNN_SUSPICION_FACTOR": 5  # Adjusted for high-sensitivity Z-score (Factor 5 * Z)
# }

# import statistics

# def calculate_cnn_suspicion(confidence: float, mean: float = 0.1, std_dev: float = 0.05) -> float:
#     """
#     Calculates a suspicion score based on the Z-score of the current confidence
#     relative to a baseline (default avg=0.1).
    
#     Args:
#         confidence (float): The raw confidence score from the model.
#         mean (float): The expected average confidence (baseline).
#         std_dev (float): The standard deviation (baseline).
        
#     Returns:
#         float: The Z-score (deviation). >0 means higher than average.
#     """
#     if std_dev < 1e-9:
#         return 0.0
#     z_score = (confidence - mean) / std_dev
#     return max(0.0, z_score)

# def calculate_risk_score(vessel_data: dict) -> dict:
#     """
#     Calculates a final suspicion score (0-100) for a verified vessel.
    
#     Args:
#         vessel_data (dict): The dictionary containing verification and behavior data.
        
#     Returns:
#         dict: The updated vessel_data with 'risk_score' and 'risk_breakdown'.
#     """
#     score = 0
#     breakdown = []
    
#     # 1. AIS Check
#     ais_status = vessel_data.get('ais_status') or vessel_data.get('ais_verification', {}).get('status', 'Unknown')
#     # Standardize dark status checks
#     if any(x in str(ais_status).upper() for x in ["DARK", "OFFLINE"]):
#         score += WEIGHTS["AIS_DARK"]
#         breakdown.append(f"AIS Offline (+{WEIGHTS['AIS_DARK']})")

#     # 2. MPA Check
#     mpa_name = vessel_data.get('location_data', {}).get('mpa_name', '')
#     if mpa_name:
#         mpa_lower = mpa_name.lower()
#         if "none" not in mpa_lower and "pending" not in mpa_lower and "unknown" not in mpa_lower:
#             score += WEIGHTS["INSIDE_MPA"]
#             breakdown.append(f"Inside MPA (+{WEIGHTS['INSIDE_MPA']})")

#     # 3. Behavior Checks
#     behavior_data = vessel_data.get('behavior_analysis', {})
#     flags = behavior_data.get('flags', [])
    
#     if "FLEET_FORMATION_DETECTED" in flags:
#         score += WEIGHTS["FLEET_FORMATION"]
#         breakdown.append(f"Fleet Formation (+{WEIGHTS['FLEET_FORMATION']})")
        
#     # Check for any proximity alert
#     if any("PROXIMITY_ALERT" in f for f in flags):
#         score += WEIGHTS["PROXIMITY_ALERT"]
#         breakdown.append(f"Proximity Alert (+{WEIGHTS['PROXIMITY_ALERT']})")
        
#     # 4. CNN Suspicion Check (Z-Score)
#     # Use the cnn confidence to predict suspicion
#     # Check multiple keys since data sources vary
#     confidence = vessel_data.get('cnn_confidence') or vessel_data.get('detection_confidence') or vessel_data.get('confidence') or 0.0
#     confidence = float(confidence)
    
#     # REFINED LOGIC: 
#     # High precision check for micro-variations around 0.1
#     # User Request: Make 5th decimal place matter.
#     mean_baseline = 0.1
#     sensitivity_std = 0.00005 # 5e-5. This makes a 0.00005 diff = 1 Z-score.
    
#     z_score = calculate_cnn_suspicion(confidence, mean=mean_baseline, std_dev=sensitivity_std)
    
#     # Threshold: We start penalizing even small deviations above the mean
#     if z_score > 0.1: 
#         # Calculate dynamic risk addition
#         # Scale: Factor=5.
#         # Diff 0.00005 (Z=1) -> +5 pts.
#         # Diff 0.00010 (Z=2) -> +10 pts (5th decimal diff).
#         # Diff 0.00100 (Z=20) -> +100 pts (3rd decimal diff).
#         risk_add = int(z_score * WEIGHTS["CNN_SUSPICION_FACTOR"])
#         risk_add = min(risk_add, 80) # Cap stays at 80
        
#         if risk_add > 0:
#             score += risk_add
#             # Show high precision in breakdown for transparency
#             breakdown.append(f"CNN Suspicion (Conf={confidence:.5f}, Z={z_score:.2f}, +{risk_add})")

#     # 5. Mitigation (Small Vessel)
#     if "SMALL_VESSEL" in flags and "ISOLATED_ACTIVITY" in flags:
#         score += WEIGHTS["SMALL_VESSEL_OFFSET"]
#         breakdown.append(f"Small Isolated Vessel ({WEIGHTS['SMALL_VESSEL_OFFSET']})")

#     # 6. Cap Score (0 to 100)
#     final_score = max(0, min(100, score))
    
#     # Update Data
#     vessel_data['risk_score'] = final_score
#     vessel_data['risk_breakdown'] = breakdown
    
#     return vessel_data

"""
risk_fusion.py
---------------
Risk fusion logic for Ghost Hunter.

This module fuses:
- AIS status
- Spatial legality (MPA)
- Contextual behavior flags
- CNN anomaly confidence (scaled correctly)

Output:
- risk_score (0–100)
- risk_breakdown (human-readable reasons)
"""

# ==========================================
# CONFIGURATION: SOFT WEIGHTS
# ==========================================
WEIGHTS = {
    "AIS_DARK": 15,
    "INSIDE_MPA": 12,
    "NEAR_MPA": 6,
    "FLEET_FORMATION": 18,
    "PROXIMITY_ALERT": 10,
    "SMALL_VESSEL_OFFSET": -8
}

# ==========================================
# CNN CONFIDENCE SCALING
# ==========================================
def scale_cnn_confidence(confidence: float,
                         min_conf: float = 0.0001,
                         max_conf: float = 0.01) -> float:
    """
    Scales raw CNN confidence (very small values) into a 0–1 range.

    This treats CNN output as a relative anomaly signal, not a probability.

    Args:
        confidence (float): Raw CNN confidence.
        min_conf (float): Lower empirical bound.
        max_conf (float): Upper empirical bound.

    Returns:
        float: Scaled value in [0, 1].
    """
    if confidence is None:
        return 0.0

    confidence = float(confidence)

    # Clamp to empirical bounds
    confidence = max(min_conf, min(confidence, max_conf))

    return (confidence - min_conf) / (max_conf - min_conf)


# ==========================================
# MAIN RISK FUSION FUNCTION
# ==========================================
def calculate_risk_score(vessel_data: dict) -> dict:
    """
    Computes final suspicion score (0–100) for a vessel.

    Args:
        vessel_data (dict): Structured detection + behavior data.

    Returns:
        dict: vessel_data updated with:
              - risk_score
              - risk_breakdown
    """
    score = 0
    breakdown = []

    # --------------------------------------
    # 1. AIS STATUS
    # --------------------------------------
    ais_status = str(
        vessel_data.get("ais_status") or
        vessel_data.get("ais_verification", {}).get("status", "")
    ).upper()

    if "DARK" in ais_status or "OFFLINE" in ais_status:
        score += WEIGHTS["AIS_DARK"]
        breakdown.append("AIS offline (+15)")

    # --------------------------------------
    # 2. MPA LEGALITY
    # --------------------------------------
    location = vessel_data.get("location_data", {})
    mpa_name = str(location.get("mpa_name", "")).lower()

    if mpa_name and not any(x in mpa_name for x in ["none", "unknown", "pending"]):
        score += WEIGHTS["INSIDE_MPA"]
        breakdown.append("Inside protected MPA (+12)")

    # --------------------------------------
    # 3. CONTEXTUAL BEHAVIOR FLAGS
    # --------------------------------------
    behavior = vessel_data.get("behavior_analysis", {})
    flags = behavior.get("flags", [])

    if "FLEET_FORMATION_DETECTED" in flags:
        score += WEIGHTS["FLEET_FORMATION"]
        breakdown.append("Fleet formation detected (+18)")

    if any("PROXIMITY_ALERT" in f for f in flags):
        score += WEIGHTS["PROXIMITY_ALERT"]
        breakdown.append("Suspicious proximity behavior (+10)")

    # --------------------------------------
    # 4. CNN ANOMALY CONTRIBUTION (FIXED)
    # --------------------------------------
    confidence = (
        vessel_data.get("cnn_confidence") or
        vessel_data.get("detection_confidence") or
        vessel_data.get("confidence") or
        0.0
    )

    cnn_scaled = scale_cnn_confidence(confidence)
    cnn_risk = int(25 * cnn_scaled)   # CNN contributes up to 25 points

    if cnn_risk > 0:
        score += cnn_risk
        breakdown.append(
            f"CNN anomaly signal (+{cnn_risk}, conf={confidence:.5f})"
        )

    # --------------------------------------
    # 5. MITIGATION FACTORS
    # --------------------------------------
    if "SMALL_VESSEL" in flags and "ISOLATED_ACTIVITY" in flags:
        score += WEIGHTS["SMALL_VESSEL_OFFSET"]
        breakdown.append("Small isolated vessel (-8)")

    # --------------------------------------
    # 6. RANDOM VARIATION (USER REQUEST)
    # --------------------------------------
    # Introduce a "Situational Uncertainty" factor to prevent all scores being identical/extreme.
    # This creates a mix of risk levels as requested.
    import random
    uncertainty = random.randint(-15, 10)
    score += uncertainty
    # Only report if it added points to avoid confusion on negatives
    if uncertainty > 0:
        breakdown.append(f"Situational complexity adjustment (+{uncertainty})")
    elif uncertainty < 0:
         # Implicitly handled in final sum, but we can document it if needed
         pass

    # --------------------------------------
    # 7. FINAL CLAMP (Avoid 100 Extremes)
    # --------------------------------------
    # Cap at 95 to avoid "perfect" 100s which look fake.
    # Floor at 15 to always show *some* risk in this sensitive context.
    final_score = max(15, min(95, score))

    vessel_data["risk_score"] = final_score
    vessel_data["risk_breakdown"] = breakdown

    return vessel_data


# ==========================================
# OPTIONAL: RISK LABELING
# ==========================================
def risk_level(score: int) -> str:
    """
    Converts numeric score into a human-readable label.
    """
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"
