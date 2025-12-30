# Ghost Hunter Behavioral Analysis System

## Overview

The Ghost Hunter system implements a sophisticated **multi-layered behavioral analysis pipeline** that combines algorithmic pattern detection with AI-powered intelligence interpretation to identify suspicious maritime activities. The analysis operates on multiple scales: individual vessel behavior, pairwise interactions, and fleet-level coordination patterns.

## 🔍 Analysis Pipeline Architecture

### 1. **Individual Vessel Analysis**
```python
# Size-based classification
if vessel_size_pixels < SMALL_VESSEL_PIXEL_LIMIT (300px):
    flags.append("SMALL_VESSEL")
```

**Purpose**: Classify vessels by size to understand their operational profile
- **Small vessels** (<300 pixels): Often legitimate fishing boats
- **Large vessels**: Potential commercial fishing or cargo ships
- **Size correlation**: Links pixel size to real-world vessel dimensions

### 2. **Pairwise Proximity Analysis**
```python
# Haversine distance calculation for vessel pairs
distance = haversine(lat1, lon1, lat2, lon2)
if distance < PROXIMITY_THRESHOLD_KM (2.0km):
    flags.append("PROXIMITY_ALERT")
    suspicion_level = "HIGH"
```

**Detects**: 
- **Cargo transfers** at sea (transshipment)
- **Fuel transfers** between vessels
- **Coordinated fishing** operations
- **Rendezvous activities** in remote areas

**Intelligence Value**: Vessels meeting at sea, especially dark vessels, indicate potential illegal activities like fish laundering or supply transfers.

### 3. **Fleet Formation Detection**
```python
# Triangle area method for collinearity detection
area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
if area < 0.005:  # Vessels in formation
    flags.append("FLEET_FORMATION_DETECTED")
    suspicion_level = "CRITICAL"
```

**Detects**:
- **Coordinated fishing fleets** operating in formation
- **Systematic area coverage** patterns
- **Military-style coordination** suggesting organized operations
- **Purse seine operations** with multiple vessels

**Algorithm**: Uses computational geometry to detect when 3+ vessels form linear patterns, indicating coordinated movement.

### 4. **Contextual Intelligence Layer**
```python
# Contextual interpretation
if is_small_vessel and not has_proximity_alerts:
    flags.append("ISOLATED_ACTIVITY (Likely Legal)")
    suspicion_level = "LOW (Verified)"
```

**Purpose**: Apply maritime domain knowledge to reduce false positives
- **Isolated small vessels**: Likely legitimate artisanal fishing
- **Large vessels in formation**: Higher suspicion for commercial operations
- **Context-aware scoring**: Adjusts risk based on operational patterns

## 🎯 Risk Scoring Integration

### Weighted Risk Factors
```python
WEIGHTS = {
    "AIS_DARK": 40,           # No AIS transponder
    "INSIDE_MPA": 30,         # Operating in protected area
    "FLEET_FORMATION": 50,    # Coordinated operations
    "PROXIMITY_ALERT": 30,    # Vessel-to-vessel contact
    "SMALL_VESSEL_OFFSET": -20 # Mitigation for small boats
}
```

### Risk Calculation Process
1. **Base Score**: Starts at 0
2. **AIS Status**: +40 points for dark vessels (highest single factor)
3. **Location**: +30 points if inside Marine Protected Area
4. **Behavior**: +50 points for fleet formation, +30 for proximity
5. **Mitigation**: -20 points for isolated small vessels
6. **Final Score**: Capped between 0-100

### Risk Categories
- **0-30**: Low Risk (Likely legitimate)
- **31-60**: Moderate Risk (Requires monitoring)
- **61-80**: High Risk (Investigation recommended)
- **81-100**: Critical Risk (Immediate action required)

## 🧠 AI-Enhanced Intelligence Layer

### GenAI Analysis Integration
The behavioral analysis feeds into a **Google Gemini-powered intelligence engine** that provides:

#### 1. **Pattern Interpretation**
```python
# AI analyzes behavioral flags in maritime context
"PROXIMITY_ALERT + AIS_DARK + INSIDE_MPA" 
→ "High probability illegal transshipment operation"
```

#### 2. **Threat Assessment**
- **Illegal Fishing Indicators**: Fleet coordination, MPA violations
- **Maritime Security Threats**: Dark vessels, unusual patterns
- **Environmental Violations**: Activities in sensitive areas
- **Smuggling Indicators**: Rendezvous patterns, evasive behavior

#### 3. **Specialized Analysis Modes**
- **Illegal Fishing Analysis**: Focus on fishing fleet patterns
- **Maritime Security**: National security implications
- **Environmental Protection**: Conservation impact assessment
- **Tactical Analysis**: Operational response recommendations

### Intelligence Output Structure
```python
class IntelligenceReport:
    executive_summary: str
    threat_assessment: str
    key_findings: List[str]
    recommendations: List[str]
    confidence_level: str
```

## 📊 Real-World Application Examples

### Example 1: Illegal Transshipment Detection
```
Input: 2 large vessels, both AIS dark, 1.5km apart in MPA
Behavioral Analysis: PROXIMITY_ALERT + AIS_DARK + INSIDE_MPA
Risk Score: 40 + 30 + 30 = 100 (CRITICAL)
AI Analysis: "High probability cargo transfer operation in protected waters"
```

### Example 2: Coordinated Fishing Fleet
```
Input: 4 vessels in linear formation, mixed AIS status
Behavioral Analysis: FLEET_FORMATION + some AIS_DARK
Risk Score: 50 + 40 = 90 (CRITICAL)
AI Analysis: "Organized fishing operation with support vessels"
```

### Example 3: Legitimate Artisanal Fishing
```
Input: Small vessel, isolated, AIS off (common for small boats)
Behavioral Analysis: SMALL_VESSEL + ISOLATED_ACTIVITY
Risk Score: 40 - 20 = 20 (LOW)
AI Analysis: "Likely legitimate artisanal fishing activity"
```

## 🔧 Technical Implementation Details

### Distance Calculations
- **Haversine Formula**: Accurate great-circle distance on Earth's surface
- **Precision**: Accounts for Earth's curvature for global operations
- **Performance**: Optimized for real-time analysis of large vessel datasets

### Geometric Analysis
- **Collinearity Detection**: Triangle area method for formation detection
- **Threshold Tuning**: Empirically derived from maritime operational patterns
- **Scale Adaptation**: Works across different geographic scales

### Data Integration
- **Multi-source Fusion**: SAR imagery + AIS data + CNN verification
- **Temporal Analysis**: Tracks behavior changes over time
- **Spatial Context**: Incorporates MPA boundaries and sensitive areas

## 🎯 Operational Impact

### For Maritime Enforcement
- **Prioritized Targets**: Risk scores guide patrol asset deployment
- **Evidence Collection**: Behavioral patterns support legal cases
- **Resource Optimization**: Focus on highest-risk activities

### For Conservation
- **Protected Area Monitoring**: Automated MPA violation detection
- **Ecosystem Protection**: Early warning for harmful activities
- **Compliance Monitoring**: Track fishing fleet compliance

### For Intelligence Analysis
- **Pattern Recognition**: Identify new operational methods
- **Trend Analysis**: Track changes in illegal fishing tactics
- **Predictive Capabilities**: Anticipate future activities

## 🚀 Advanced Features

### Machine Learning Enhancement
- **Pattern Learning**: System learns from analyst feedback
- **Threshold Optimization**: Automatic tuning based on results
- **False Positive Reduction**: Continuous improvement of detection accuracy

### Multi-temporal Analysis
- **Behavior Tracking**: Monitor vessel behavior over time
- **Route Analysis**: Identify suspicious movement patterns
- **Seasonal Patterns**: Detect timing-based illegal activities

### Integration Capabilities
- **Real-time Processing**: Live satellite feed analysis
- **Alert Systems**: Automated notifications for critical threats
- **API Integration**: Seamless connection with enforcement systems

This behavioral analysis system represents a significant advancement in maritime surveillance, combining traditional algorithmic approaches with cutting-edge AI to provide actionable intelligence for maritime security and conservation efforts.