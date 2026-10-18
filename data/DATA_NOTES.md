# DATA_NOTES.md: Dengue Sentinel Datasets

This document records the provenance, schema, licensing, and processing notes for all datasets used in **Dengue Sentinel** per BUILD.md Section 5, Section 7, Section 8, and Section 9.1.

---

## 1. Historical Dengue Case Counts (OpenDengue / WHO SEARO)

- **Source**: OpenDengue / WHO South-East Asia Regional Office (SEARO)
- **Portal URL**: `https://opendengue.org/data.html` (*verify before use per Hard Rule 5*)
- **Script**: `data/fetch_opendengue.py`
- **Local File in `data/raw/`**: `data/raw/filtered_data_SEARO_1792086539625.csv`
- **Geographic Coverage**: India (`ISO_A0 = IND`), State of Karnataka (`adm_1_name = KARNATAKA`)
- **Spatial Resolution (`S_res`)**: `Admin1` (State level)
- **Temporal Resolution (`T_res`)**: `Year` (Annual reported totals from 1997 to 2024, 26 years available)
- **Key Columns**:
  - `adm_1_name`: Administrative division 1 (`KARNATAKA`)
  - `Year`: Reporting year (1997–2024)
  - `dengue_total`: Total annual reported dengue cases
  - `calendar_start_date` / `calendar_end_date`: Time window (`YYYY-01-01` to `YYYY-12-31`)
  - `case_definition_standardised`: `Total`
- **Backbone Season**:
  - **2024** recorded the highest epidemic in Karnataka history (**32,886 reported cases**), followed by 2023 (**19,300 cases**) and 2017 (**17,844 cases**).
  - 2024 serves as the primary backbone curve for seasonal interpolation and signal generation.
- **Coverage & Pitch Wording (per BUILD.md Section 8)**:
  > *"Real state/district-level trends, simulated ward-level early signals."*
- **Licence**: Open data under Creative Commons Attribution (CC-BY 4.0) via OpenDengue.

---

## 2. Historical Daily Weather (Open-Meteo Archive API)

- **Source**: Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`) (*verify before use per Hard Rule 5*)
- **Script**: `data/fetch_weather.py`
- **Local File in `data/raw/`**: `data/raw/bengaluru_weather_2018_2025.csv`
- **Coordinates**: Bengaluru City Center (`latitude = 12.97`, `longitude = 77.59`)
- **Temporal Range**: Daily from `2018-01-01` to `2025-12-31` (2,922 records)
- **Variables**:
  - `date`: ISO `YYYY-MM-DD`
  - `rainfall`: Daily precipitation sum in millimeters (`precipitation_sum`)
  - `temp`: Daily mean temperature at 2 meters in degrees Celsius (`temperature_2m_mean`)
  - `humidity`: Daily mean relative humidity percentage (`relative_humidity_2m_mean`)
- **Key Characteristics**:
  - Mean temperature: 23.3°C (min 17.7°C, max 31.5°C)
  - Mean humidity: 70.2% (min 26.0%, max 96.0%)
  - Peak daily rainfall: 73.4 mm
- **Classification**: **Real** (copied to all wards per Section 7.1; `is_synthetic = false`).
- **Licence**: Attribution 4.0 International (CC-BY 4.0). Free for research and competition use without API key.

---

## 3. Bengaluru Ward Boundaries (GeoJSON)

- **Asset Path**: `dashboard/assets/bengaluru_wards.geojson`
- **Ward System**: BBMP (Bruhat Bengaluru Mahanagara Palike)
- **Selected Wards**: 20 representative wards spanning North, Central, South, East, and West zones.
- **Focal Outbreak Ward**: `W017` (Banashankari, South Zone) matching Section 7.2.
- **Properties**: `ward_id`, `ward_name`, `zone`, `centroid_lat`, `centroid_lon`.
- **Classification**: Real boundary approximations and coordinates; simulated signal attachment.

---

## 4. Synthetic Streamed Signals (BUILD.md Section 7.1 & 9.1)

- **Signals**:
  - `clinic_fever_visits`: Synthetic daily fever visits per ward (`is_synthetic = true`).
  - `pharmacy_fever_meds`: Synthetic daily antipyretic OTC purchases per ward (`is_synthetic = true`).
- **Generator**: `generator/generate_signals.py`
- **Mechanism**:
  - Negative Binomial noise (`NegBin(mean, dispersion)`) to capture real-world medical overdispersion.
  - Care pathway lags: Pharmacy purchases lead clinical visits (`pharmacy` lag 0-2 days, `clinic` lag 2-4 days).
  - Confounders: Non-dengue viral fever wave and random false spikes injected into non-dengue wards.
