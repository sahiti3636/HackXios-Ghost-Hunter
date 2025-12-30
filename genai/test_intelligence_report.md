# MARITIME INTELLIGENCE REPORT
**Generated:** 2025-12-29 21:46:26 UTC
**Source:** Ghost Hunter Maritime Surveillance System

## EXECUTIVE SUMMARY
Seven unverified dark vessels have been detected by the Ghost Hunter SAR system in the region 'sat1' (Andaman Sea vicinity) on 2025-12-29. All detections lack AIS transmissions and have been assigned a default risk score of 40 due to 'AIS Offline' status. Crucially, the Convolutional Neural Network (CNN) vessel verification confidence is extremely low for all targets, suggesting that the system is not confident these are actual vessels or cannot classify them effectively. No suspicious behavioral flags were identified by the automated analysis. While the presence of dark contacts warrants attention, the low verification confidence necessitates immediate follow-up to confirm targets before escalating the threat level or deploying enforcement assets.

## THREAT ASSESSMENT
**Overall Threat Level:** MODERATE-LOW. The primary threat indicator is the 100% 'DARK' status of all seven contacts, which inherently suggests potential intent to evade detection, often associated with Illegal, Unreported, and Unregulated (IUU) fishing or other illicit activities. However, the extremely low CNN verification confidence for all targets significantly lowers the immediate confirmed threat level. It is highly plausible that some or all of these detections could be false positives (e.g., radar artifacts, environmental phenomena) or unclassified targets. Without visual confirmation or higher confidence in vessel identification, the actionable threat is currently unconfirmed, but the potential for illicit activity by genuine dark vessels remains.

## KEY FINDINGS
- **100% Dark Vessels:** All seven detected contacts are operating without AIS, a primary indicator of potential illicit activity or intent to avoid detection.
- **Extremely Low CNN Confidence:** The advanced CNN model assigned confidence scores ranging from 0.0 to 0.0026 for all detections, indicating very low certainty that these are actual vessels or that they can be classified. This is a critical factor influencing overall analysis confidence.
- **Uniform Risk Score:** All detected contacts received an identical risk score of 40, solely attributed to their 'AIS Offline' status. No other factors (e.g., behavior, location) contributed to increasing or decreasing this score according to the current model parameters.
- **Low Behavioral Suspicion:** Automated behavioral analysis flagged all vessels as 'LOW' suspicion with no specific flags (e.g., loitering, unusual courses, proximity to restricted areas), although the veracity of this is contingent on actual vessel confirmation and the sophistication of the behavioral models against unclassified contacts.
- **Proximate Detections:** Vessels 2 and 3 are in close proximity (approx. 6.23°N, 94.98°E and 6.23°N, 94.92°E respectively), suggesting potential coordinated activity or a single event, if confirmed as vessels. Vessels 5 and 6 also show relative proximity.
- **Varied SAR Signatures:** While CNN confidence is low, SAR signatures (mean/max backscatter, SBCI) vary, with Vessel 7 showing the strongest maximum backscatter (6542.0) and SBCI (17.77), and Vessel 6 also presenting a high backscatter (6030.0). This suggests distinct radar returns, even if unverified as vessels.
- **Geographic Context:** The operational area is in the broader Andaman Sea, a known region for trans-regional maritime traffic, potential IUU fishing, and other maritime security challenges. Specific EEZ boundaries or protected areas are not detailed in the data, but proximity should be considered.

## VESSEL ANALYSIS
### Vessel 1 - Risk Score: 40/100
**Location:** 6.2853°N, 93.4469°E
Here's an analysis of the individual vessel detection, incorporating maritime domain expertise and operational implications:

---

**INDIVIDUAL VESSEL DETECTION ANALYSIS (ID: 1)**

**1. Threat Assessment and Classification:**

*   **Primary Threat:** The vessel is confirmed "DARK" (AIS Offline), which is the sole contributor to its elevated risk score (+40). In the identified operating area (Andaman Sea / Malacca Strait vicinity), known for illegal fishing, smuggling, and piracy, a dark vessel is an immediate and significant indicator of potential illicit activity or intent to conceal operations.
*   **Regional Context:** The fact that *all 7* vessel candidates in this scan are without AIS significantly heightens the overall regional threat picture. This isn't an isolated incident, but rather suggests a widespread pattern of AIS suppression, which is highly concerning.
*   **System Contradiction:** The automated `behavior_analysis` flags the `suspicion_level` as "LOW" despite the "DARK" AIS status. This is a critical discrepancy. A vessel operating dark in a high-risk maritime domain should generally trigger at least a "MEDIUM" suspicion level, if not "HIGH", irrespective of other specific behavioral flags. This suggests the automated behavioral model may need recalibration or human override for AIS dark detections.
*   **Initial Classification:** **Moderate-to-High Suspicion.** While the system assigns a "LOW" suspicion, the human analyst overrides this due to the combination of AIS Dark status, the high-risk geographic area, and the fact that all detections in the region are dark.

**2. Technical Signature Interpretation:**

*   **Detection Method:** The detection is based on SBCI (Ship-to-Background Contrast Index) from SAR imagery, which is a reliable method for identifying metallic objects on the sea surface. The `detection_confidence` of 12.4 is moderate, indicating a clear signal above background noise.
*   **Object Size Discrepancy:**
    *   `area_pixels`: 9. This is an extremely small detection. A 9-pixel object in SAR imagery could be a very small boat, marine debris, a radar artifact, or a fragment of a larger object.
    *   `vessel_size_pixels`: 3724. This value is highly contradictory to `area_pixels`. It is most likely referring to the size of the *image chip* or a bounding box in the original SAR image, *not* the actual detected size of the vessel. This discrepancy is a critical point that reduces confidence in the object's nature. If the actual vessel occupies only 9 pixels, it's very small.
*   **Verification Failure:**
    *   `cnn_confidence`: 0.0026. This is an extremely low confidence score from the Convolutional Neural Network (CNN) classifier. It strongly suggests the CNN *failed* to identify the detected object as a vessel.
    *   `is_vessel_verified`: "NO". This explicitly confirms the CNN's failure to verify the object as a vessel.
*   **Conclusion:** While a SAR contrast exists (SBCI detection), the very small detected `area_pixels` (9) coupled with the complete failure of the CNN (extremely low confidence) and the explicit "NO" verification status, cast significant doubt on whether this is indeed a vessel, or merely a small, ambiguous clutter return.

**3. Behavioral Indicators:**

*   **AIS Suppression:** The definitive "DARK" AIS status is the primary and most significant behavioral indicator. Intentional disabling of AIS is a strong signal of intent to hide identity, destination, or activities. This is often associated with illegal fishing (IUU), smuggling, piracy, or other clandestine operations.
*   **Lack of Other Flags:** The `behavior_analysis` states "flags": []. This indicates that beyond AIS suppression, no other specific behavioral patterns (e.g., loitering, unusual speed changes, specific transit routes) were automatically identified as suspicious *by the system*. However, given the AIS Dark status, the absence of these *additional* flags does not mitigate the primary concern.

**4. Recommended Actions:**

*   **Immediate Action (Within Minutes):**
    *   **Human Visual Confirmation:** **Highest Priority.** Immediately task a human analyst to review the provided `image: "output/chips/sat1/vessel_1.png"`. Given the extremely low CNN confidence and tiny pixel area, it is imperative to visually confirm if the 9-pixel detection is indeed a vessel, or if it is clutter/an anomaly.
    *   **Cross-Reference with Other Sources:** Check for any other available intelligence (e.g., optical satellite imagery if within revisit time, maritime patrol aircraft reports, SIGINT/ELINT, open-source intelligence) covering the coordinates (6.285°N, 93.447°E) at or around the timestamp.
*   **Follow-up Actions (Within Hours/Day):**
    *   **If Confirmed as Vessel:**
        *   **Targeted Surveillance:** If the human review confirms a vessel, task higher-resolution SAR or optical satellite imagery for follow-on collection to obtain better details on size, type, heading, speed, and potential activity.
        *   **Pattern of Life Analysis (Regional):** Extend pattern of life analysis to all 7 dark vessels detected in the region. Are they moving in concert? Are they static (potential transshipment)? Are there indications of fishing gear?
        *   **Local Asset Alert:** Alert regional maritime authorities or patrols (coast guard, navy) about a dark vessel in a high-risk area, especially if further analysis elevates the threat.
    *   **If Not Confirmed as Vessel (Clutter/Anomaly):**
        *   **Dismiss Detection:** Mark the detection as non-vessel and remove it from the active threat picture.
        *   **System Feedback:** Provide feedback to the SAR processing and CNN verification teams to improve discrimination for small targets and reduce false positives.
*   **Strategic Action:**
    *   **System Re-evaluation:** Investigate why the `behavior_analysis` module classified this as "LOW" suspicion despite the "DARK" AIS status, and the significant discrepancy between `area_pixels` and `vessel_size_pixels`. These represent potential flaws in the automated intelligence pipeline.

**5. Confidence Assessment:**

*   **Detection Confidence (Raw SAR):** **MODERATE.** The SBCI value of 12.4 indicates a reasonably strong SAR return above background.
*   **Vessel Classification Confidence (Object is a Vessel):** **LOW.** This is the weakest link. The extremely low CNN confidence (0.0026), explicit "NO" verification, and the tiny 9-pixel area of the actual detection (despite the misleading `vessel_size_pixels` value) mean there is very low confidence that this object is actually a vessel. It could very plausibly be clutter or a radar artifact.
*   **Threat Confidence (If it is a Vessel):** **HIGH.** *If* this object is confirmed to be a vessel, then its "DARK" status in this specific high-risk region, coupled with the systemic "DARK" nature of all other regional detections, immediately elevates its threat potential significantly.
*   **Overall Intelligence Confidence:** **LOW-MEDIUM.** While the *reason for suspicion* (AIS Dark) is high, the fundamental confidence in *what the detected object actually is* remains critically low due to the technical verification failures. Until human visual confirmation or higher-resolution imagery clarifies the nature of the 9-pixel object, its intelligence value is hampered by the uncertainty of its classification.

### Vessel 2 - Risk Score: 40/100
**Location:** 6.2305°N, 94.9896°E
Here's an analysis of the individual vessel detection:

## Intelligence Analysis: Vessel ID 2

**MISSION CONTEXT:** Marine surveillance in `sat1` region, with a current scenario of **7 out of 7 detected vessels operating without AIS** (DARK). This establishes a high baseline of concern for the operational area.

---

### 1. Threat Assessment and Classification

*   **Primary Threat Indicator:** This vessel is unequivocally classified as **"DARK" (AIS Offline)**. This is a significant red flag in maritime intelligence, as vessels intentionally disable AIS to avoid detection and scrutiny. This behavior is strongly associated with illicit activities such as illegal, unreported, and unregulated (IUU) fishing, smuggling (drugs, weapons, people), unauthorized transshipment, sanctions evasion, or other clandestine operations.
*   **Risk Score:** A `risk_score` of **40/100** is assigned, explicitly attributed to "AIS Offline (+40)". While not extremely high, it immediately categorizes the vessel as a `Vessel of Interest (VOI)`.
*   **Behavioral Contradiction:** The `behavior_analysis.suspicion_level` is `LOW`, despite being dark. This is a critical observation. It suggests the automated behavioral analysis primarily relies on factors *other than* AIS status for "suspicion level" beyond the initial "AIS Offline" risk trigger (e.g., lack of unusual movement patterns, loitering, deviation from typical routes, etc.). However, for any maritime intelligence analyst, being "DARK" immediately elevates a vessel to at least a `Medium` level of suspicion due to the intent implied by disabling safety and identification systems.
*   **Classification:** This vessel is classified as a **Vessel of Interest (VOI)** with a **High potential for illicit activity** specifically due to its "DARK" status, contributing to a pattern of concerning behavior in the region.

### 2. Technical Signature Interpretation

*   **Sensor & Detection Method:** The vessel was detected via **SAR imagery** using the **SBCI (Ship-to-Background Contrast Index)** method. This indicates a robust detection capability even in adverse weather or low visibility conditions where EO/IR sensors might struggle.
*   **Detection Confidence:** `detection_confidence` is **8.4067** (matching `max_sbci`). This is a high confidence level for the *presence* of a vessel-like object in the SAR image.
*   **CNN Verification:** `cnn_confidence` is **0.0**, and `is_vessel_verified` is **"NO"**. This is a significant technical gap. It means the automated CNN model either failed to identify it as a vessel, did not run successfully, or was not designed to process this specific type of detection. This introduces uncertainty regarding the *exact nature* and *type* of vessel, despite the high SBCI detection confidence.
*   **Physical Characteristics (Ambiguous):**
    *   `area_pixels`: **9**. This is an extremely small detected area, potentially suggesting a very small vessel (e.g., fishing skiff, dinghy) or a partial/low-resolution detection of a larger vessel.
    *   `vessel_size_pixels`: **3600**. This value contradicts `area_pixels=9` dramatically. If this refers to the *actual* estimated size in pixels, it would represent a very large vessel. This is a major discrepancy that requires clarification or correction in the data pipeline. Without this, the estimated size of the vessel is highly ambiguous. For analysis, one must consider both possibilities (very small vs. very large) or assume one is an error. Given the `area_pixels=9` seems to be the direct detection blob, it's more likely a small object or a poor resolution detection.
    *   `mean_backscatter`: 984.8889, `max_backscatter`: 1389.0. These values are consistent with the strong radar reflection expected from a metallic vessel.
*   **Location:** 6.230499°N, 94.989614°E. This location is within the defined mission area. It's in the Eastern Indian Ocean, near significant maritime traffic lanes and sensitive zones (e.g., near the Great Nicobar Island of India and Sumatra of Indonesia). This region is known for diverse maritime activities, including potential for IUU fishing and illicit trafficking.

### 3. Behavioral Indicators

*   **Dark Operation:** The primary behavioral indicator is the deliberate act of operating with AIS off. This is almost universally indicative of an attempt to conceal identity, location, or activities.
*   **No Other Flags:** `behavior_analysis.flags`: `[]`. This indicates no other suspicious movement patterns (e.g., loitering, unusual speed changes, specific rendezvous patterns) were detected by the automated system at the time of analysis. This suggests the vessel might be on a relatively consistent course or has not exhibited other complex suspicious behaviors *yet*.
*   **Regional Pattern:** This vessel is one of `7/7` dark vessels detected in the area. This is a critical context. It indicates a systemic issue or a concerted pattern of dark activity within this region, rather than an isolated incident. This widespread dark activity significantly elevates the overall regional threat picture.

### 4. Recommended Actions

Given the high confidence in detection but uncertainty in specific characteristics and the critical "DARK" status:

1.  **Immediate Cross-Verification:**
    *   **Multi-Source Intelligence Fusion:** Immediately cross-reference with other available ISR (Intelligence, Surveillance, and Reconnaissance) assets, including higher-resolution SAR, electro-optical (EO) imagery, or potentially even acoustic sensors if available for this region.
    *   **Historical Data:** Check for any historical SAR detections or intelligence reports related to this specific location or vessel characteristics (if size ambiguity can be resolved).
2.  **Tracking & Monitoring:**
    *   **Persistent Surveillance:** Prioritize this vessel (and the other 6 dark vessels) for continuous monitoring using available satellite assets (SAR, optical) to track its course, speed, and any changes in behavior.
    *   **Pattern Analysis:** Analyze its trajectory in relation to known shipping lanes, fishing grounds, port facilities, or other areas of interest. Look for rendezvous with other vessels or loitering behavior.
3.  **Investigate Technical Discrepancies:**
    *   **Size Ambiguity:** Urgent investigation into the `area_pixels` (9) vs. `vessel_size_pixels` (3600) discrepancy is required to accurately assess the vessel's physical size and potential capabilities.
    *   **CNN Failure:** Investigate why `cnn_confidence` is 0.0 and `is_vessel_verified` is "NO." This could indicate a data processing error, a limitation of the CNN model for small targets, or an unusual vessel type that the CNN is not trained on. Rectifying this could improve future automated assessments.
4.  **Alert & Interdiction Planning:**
    *   **Inform Authorities:** Report this "DARK" vessel (and the cluster of 7 dark vessels) to relevant maritime law enforcement agencies (e.g., Coast Guard, Navy, Fisheries Agencies) and regional maritime security centers.
    *   **Risk Mitigation:** If tracking reveals suspicious behavior (e.g., heading towards an exclusion zone, known smuggling route, or engaging in illegal transshipment), prepare for potential interdiction scenarios, coordinating with regional partners.

### 5. Confidence Assessment

*   **Detection Confidence:** **High (9/10)**. The SBCI score is robust, indicating a very high probability that a vessel-like object is present.
*   **"DARK" Status Confidence:** **High (9/10)**. AIS status is directly reported as "DARK," which is a binary and generally reliable classification.
*   **Threat Classification Confidence:** **Moderate to High (7/10)**. While the "DARK" status is a clear threat indicator, the `behavior_analysis.suspicion_level` being `LOW` (despite being dark) and the severe `vessel_size_pixels` discrepancy introduce uncertainty regarding the *specific type* of threat or the *immediacy* of the threat. The context of a region filled with dark vessels elevates the overall concern significantly.
*   **Overall Intelligence Confidence:** **Moderate (6.5/10)**. We are highly confident *that* a dark vessel exists at this location. However, we have low confidence in its physical characteristics (size) and the automated behavioral assessment is contradictory to the primary risk factor. Further data (especially visual confirmation or clearer size metrics) is needed to increase confidence in the vessel's precise nature and intent.

### Vessel 3 - Risk Score: 40/100
**Location:** 6.2287°N, 94.9202°E
## Individual Vessel Detection Analysis: Vessel ID 3

**TIMESTAMP:** 2025-12-29T15:43:18.603664Z
**LOCATION:** 6.228693909090909 N, 94.92017320394685 E (Andaman Sea, West of Sumatra)

### 1. Threat Assessment and Classification

*   **Classification:** This vessel is classified as a **Vessel of Interest (VOI)** with a **moderate initial risk profile**.
*   **Rationale:** The primary driver for this classification is its **"DARK" AIS status**, meaning it is not broadcasting its identity or position via AIS. This behavior directly contributes 40 points to its risk score, making it the sole identified risk factor. While the system's `suspicion_level` is currently `LOW` and no specific behavioral `flags` are raised, any "dark" vessel operating in a marine surveillance region warrants further scrutiny due to potential for illicit activities (e.g., illegal fishing, smuggling, unauthorized transshipment, or intelligence gathering). The regional context of 7 out of 7 dark vessels highlights a potential widespread issue or a specific operational characteristic of this area that needs to be understood.

### 2. Technical Signature Interpretation

*   **Detection Method & Quality:** The vessel was detected using SAR imagery via the SBCI method, which is highly effective in all-weather conditions. The `mean_sbci` (7.0) and `max_sbci` (9.5) values, coupled with a `detection_confidence` of 9.48, indicate a strong and reliable SAR signature consistent with a metallic object on the water. The `mean_backscatter` of 1020 and `max_backscatter` of 1378 further corroborate a clear radar return.
*   **Vessel Size:** The `area_pixels` of 15 suggests a relatively small target in the SAR image. This is a critical detail. However, the `vessel_size_pixels` is stated as 3710. This is a significant discrepancy. If `area_pixels` represents the actual detected feature size, then 3710 `vessel_size_pixels` (unless it refers to the bounding box of the image chip or a projected physical size from an external model) is an anomaly. Assuming `area_pixels=15` reflects the actual SAR footprint, this points to a small craft.
*   **Verification Status:** The `cnn_confidence` of 0.0 and `is_vessel_verified: "NO"` are significant. Despite the strong SAR detection, the CNN model was unable to confirm it as a vessel. This could be due to the small size of the target in the image chip, poor resolution for CNN analysis, clutter, or the target not resembling typical vessel shapes known to the CNN model. This failure to verify means we have a strong *detection* but a weak *identification* from automated systems.

### 3. Behavioral Indicators

*   **AIS Status:** **"DARK"** is the primary behavioral flag. While it's the only risk factor identified by the system, it's a critical one for marine surveillance missions.
*   **System Suspicion:** The `suspicion_level: "LOW"` despite being "DARK" AIS is notable. This might suggest that in this specific `sat1` region, operating without AIS is common for certain vessel types (e.g., local fishing vessels, smaller craft). However, for a marine surveillance mission, even "common" dark vessels still represent an intelligence gap. The absence of other `flags` further supports the system's low suspicion, but human intelligence assessment should override automated "low suspicion" when the core behavior (dark AIS) poses an inherent risk in a surveillance context.
*   **Operating Area:** The vessel is located in the Andaman Sea, an area with significant maritime traffic, including commercial shipping lanes, fishing grounds, and proximity to regional choke points. This location amplifies the potential significance of a dark vessel.
*   **Regional Context:** The fact that `7/7` detections are "DARK VESSELS" indicates a prevalent issue or characteristic of this particular surveillance area. This might influence the system's `suspicion_level` towards "LOW" if operating without AIS is highly common among local vessels. However, it also means the entire surveillance picture for this mission segment lacks fundamental transparency.

### 4. Recommended Actions

1.  **Image Chip Review:** Immediately review the `output/chips/sat1/vessel_3.png` image chip manually to assess why the CNN failed (`cnn_confidence: 0.0`). This manual review will help clarify the target's nature (e.g., vessel, platform, false positive) and potentially resolve the `area_pixels` vs. `vessel_size_pixels` discrepancy.
2.  **Cross-Reference with Other Data:**
    *   Check for any historical SAR detections or optical imagery of this specific location or similar signatures that might provide context for this type of "dark" activity.
    *   Consult regional intelligence on common vessel types and typical AIS broadcasting patterns in this part of the Andaman Sea to understand if the "low suspicion" is justified by local norms or if it's a systemic underestimation.
3.  **Prioritize Follow-up Imagery:** If initial manual review confirms a potential vessel, prioritize this location for subsequent SAR or electro-optical (EO) satellite passes (if available) to attempt identification and track movement.
4.  **Intelligence Briefing Update:** If the prevalence of dark vessels (7/7) is new or unexpected, brief relevant intelligence analysts on this trend in the `sat1` region.
5.  **Review System Parameters:** Given the discrepancy between a strong SAR detection, CNN failure, and the `area_pixels`/`vessel_size_pixels` anomaly, review the parameters for the CNN model and the vessel sizing algorithm for this specific sensor/region.

### 5. Confidence Assessment

*   **Detection Confidence:** **HIGH**. The strong SBCI values and high `detection_confidence` indicate a very reliable detection of a radar-reflective object on the water surface by SAR.
*   **Vessel Identification/Verification Confidence:** **LOW**. The `cnn_confidence` of 0.0 and `is_vessel_verified: "NO"` mean there is no automated verification that this is indeed a vessel, nor its type. This requires manual verification.
*   **Size Estimation Confidence:** **LOW**. The large discrepancy between `area_pixels` (15) and `vessel_size_pixels` (3710) significantly reduces confidence in the estimated size of the object without further clarification or manual assessment. Assuming `area_pixels` is the actual target size, this is a small target.
*   **Behavioral Assessment Confidence:** **MODERATE**. While "DARK" AIS is definitive, the system's `suspicion_level: "LOW"` feels contradictory given the surveillance mission. This score may reflect regional norms but doesn't fully capture the inherent intelligence gap created by non-broadcast.

**Overall Confidence in Intelligence Value:** **MODERATE**. We are highly confident *something* is there, and it's not broadcasting AIS. However, without further verification and clarification on its nature and size, its specific intelligence value beyond being a "dark vessel" remains limited. This detection highlights a significant area for immediate follow-up actions.

## RECOMMENDATIONS
- **Priority Verification (Tier 1):** Immediately task available Electro-Optical (EO) satellite assets or Aerial Maritime Patrol Aircraft (MPA)/UAVs to acquire high-resolution imagery and visually confirm the presence and type of vessels for all seven detections, especially for Vessels 2 and 3 (due to proximity) and Vessel 7 (strongest SAR signature).
- **Historical AIS/VMS Cross-reference:** Conduct a historical AIS and Vessel Monitoring System (VMS) data check for the precise locations to identify any past vessel activity or patterns of life that might correspond to these unverified contacts.
- **SAR Re-tasking:** Schedule repeat SAR passes over these locations within the next 12-24 hours to track movement, persistence, and to gather additional data that could aid in confirming vessel status and behavior.
- **Intelligence Correlation:** Cross-reference the coordinates and approximate times with other intelligence sources, such as regional naval reports, law enforcement intelligence, and known IUU vessel databases.
- **Refined Behavioral Analysis:** If confirmed as vessels, re-run behavioral analysis with a broader context, including proximity to known fishing grounds, marine protected areas, and economic exclusion zones (EEZs) for the Andaman Sea region.
- **System Feedback Loop:** Provide feedback to the Ghost Hunter development team regarding the low CNN confidence on dark vessel detections and the uniform risk scoring, to improve future detection algorithms and risk assessment models.
- **Alert Regional Maritime Enforcement:** If any detection is visually confirmed as a vessel and shows persistent dark operation or suspicious behavior, immediately alert relevant regional maritime enforcement agencies (e.g., coast guard, navy) for potential interdiction.

## TECHNICAL NOTES
The Ghost Hunter system utilizes a multi-sensor fusion approach combining SAR imagery, AIS tracking, CNN for vessel verification, and behavioral analysis. Detections are initiated using the Ship-to-Background Contrast Index (SBCI) on SAR imagery. A key technical observation is the consistent `vessel_size_pixels` (approx. 3500-3900) across all detections, despite small `area_pixels` (9-15), which may indicate the `vessel_size_pixels` metric represents an estimated real-world size in pixels rather than the raw detection footprint. The CNN confidence scores (0.0 - 0.0026) are exceptionally low, suggesting the CNN model either struggled to classify these targets as vessels or identified them as non-vessel objects with high certainty of being *not* a vessel. This significantly impacts the confidence in the initial vessel classification. The risk score model is currently heavily weighted on AIS status, assigning +40 for being offline, and no other factors in this dataset contributed to increasing the score beyond this baseline.

## CONFIDENCE ASSESSMENT
**Analysis Confidence:** MODERATE-LOW. We have high confidence in the *data delivery* by the Ghost Hunter system (i.e., that these SAR contacts exist as reported). However, our confidence in these contacts being *actual vessels* is low due to the extremely poor CNN verification scores. Furthermore, our confidence in the automated 'LOW' behavioral suspicion rating is moderate, as it relies on the contacts being confirmed vessels and the behavioral models adequately capturing relevant patterns without more detailed context (e.g., historical tracks, known fishing zones). Therefore, while the potential for illicit activity is flagged by the 'DARK' status, verification of the contacts themselves is paramount before escalating the overall threat assessment.