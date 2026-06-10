# climate-resilience-sandbox
AI-powered scenario planning platform for exploring ENSO-driven climate risks and adaptation strategies.

### Risk Classification Framework

The simulator uses project-defined planning categories (Normal, Elevated, Severe, Critical) to translate model outputs into actionable guidance.

Recommendations are derived from NDMA, IMD, FAO, and Ministry of Agriculture guidance documents. The category boundaries and routing logic are part of the simulator design and are intended for scenario-planning purposes rather than official drought declaration.

### utils.py

- The Regional Resilience Score follows the weighted composite index methodology established by FAO RIMA and the INFORM Severity Index — both of which use expert-defined pillar weightings to aggregate multi-domain indicators into a single resilience score. Our 40/40/20 weighting reflects equal priority of food and water security, consistent with FAO's food-water nexus frameworks, with agricultural risk exposure weighted lower as a derived secondary indicator.  

- The groundwater stress proxy combines reservoir storage and rainfall deficit — two variables whose relationship to groundwater depletion is documented in CGWB-based research and peer-reviewed hydrology literature (Nair et al., 2021, Geophysical Research Letters; PMC 2024). The proxy is a simplification appropriate for MVP scale, documented transparently in our codebase.  

- The Agricultural Risk Exposure metric combines crop yield deviation from historical mean with rainfall deficit severity — an approach consistent with FAO's Agricultural Stress Index System (ASIS) and validated by peer-reviewed literature showing r=0.74 correlation between rainfall deficits and crop yield anomalies (Tandfonline, 2025).
