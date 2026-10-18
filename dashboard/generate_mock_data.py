"""Generates mock data for Dengue Sentinel dashboard testing and offline replay.

Produces:
1. dashboard/assets/bengaluru_wards.geojson - Ward polygons with realistic Bengaluru coordinates.
2. data/processed/risk_scores_fallback.csv - Stream of risk scores matching Section 7.2 of SPEC.md.
3. data/processed/signals_fallback.csv - Historical signal events matching Section 7.1 of SPEC.md.

Ensures deterministic output using SEED = 42 per SPEC.md Section 3.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from streaming.config import get_alert_level

SEED = 42
np.random.seed(SEED)

# 20 representative BBMP Bengaluru wards across North, Central, South, East, West
WARDS = [
    {"ward_id": "W001", "name": "Yelahanka", "zone": "Yelahanka", "lat": 13.1007, "lon": 77.5963},
    {"ward_id": "W002", "name": "Hebbal", "zone": "East", "lat": 13.0358, "lon": 77.5970},
    {"ward_id": "W003", "name": "Malleshwaram", "zone": "West", "lat": 13.0031, "lon": 77.5643},
    {"ward_id": "W004", "name": "Rajajinagar", "zone": "West", "lat": 12.9901, "lon": 77.5525},
    {"ward_id": "W005", "name": "Shivajinagar", "zone": "East", "lat": 12.9863, "lon": 77.6047},
    {"ward_id": "W006", "name": "Gandhinagar", "zone": "South", "lat": 12.9779, "lon": 77.5777},
    {"ward_id": "W007", "name": "Shantinagar", "zone": "East", "lat": 12.9569, "lon": 77.5996},
    {"ward_id": "W008", "name": "Indiranagar", "zone": "East", "lat": 12.9784, "lon": 77.6408},
    {"ward_id": "W009", "name": "Domlur", "zone": "East", "lat": 12.9609, "lon": 77.6387},
    {"ward_id": "W010", "name": "Koramangala", "zone": "South", "lat": 12.9352, "lon": 77.6245},
    {"ward_id": "W011", "name": "HSR Layout", "zone": "Bommanahalli", "lat": 12.9121, "lon": 77.6446},
    {"ward_id": "W012", "name": "Bellandur", "zone": "Mahadevapura", "lat": 12.9260, "lon": 77.6762},
    {"ward_id": "W013", "name": "Mahadevapura", "zone": "Mahadevapura", "lat": 12.9920, "lon": 77.6900},
    {"ward_id": "W014", "name": "Whitefield", "zone": "Mahadevapura", "lat": 12.9698, "lon": 77.7499},
    {"ward_id": "W015", "name": "BTM Layout", "zone": "South", "lat": 12.9166, "lon": 77.6101},
    {"ward_id": "W016", "name": "Jayanagar", "zone": "South", "lat": 12.9308, "lon": 77.5838},
    {"ward_id": "W017", "name": "Banashankari", "zone": "South", "lat": 12.9255, "lon": 77.5468},
    {"ward_id": "W018", "name": "Basavanagudi", "zone": "South", "lat": 12.9422, "lon": 77.5753},
    {"ward_id": "W019", "name": "Padmanabhanagar", "zone": "South", "lat": 12.9180, "lon": 77.5577},
    {"ward_id": "W020", "name": "Rajarajeshwari Nagar", "zone": "RR Nagar", "lat": 12.9274, "lon": 77.5154},
]


def create_ward_polygon(center_lon: float, center_lat: float, radius_km: float = 1.2, vertices: int = 7) -> list[list[float]]:
    """Generates an irregular polygon around a centroid to simulate a ward boundary.

    Args:
        center_lon: Longitude of ward centroid.
        center_lat: Latitude of ward centroid.
        radius_km: Average radius in kilometers.
        vertices: Number of polygon vertices.

    Returns:
        List of [lon, lat] coordinates forming a closed polygon.
    """
    coords = []
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(center_lat))

    for i in range(vertices):
        angle = (2 * math.pi / vertices) * i
        # slight jitter per vertex for natural boundary look
        jitter = 0.85 + 0.30 * ((i * 17) % 7) / 7.0
        r = radius_km * jitter
        dlat = (r * math.sin(angle)) / km_per_deg_lat
        dlon = (r * math.cos(angle)) / km_per_deg_lon
        coords.append([round(center_lon + dlon, 6), round(center_lat + dlat, 6)])

    # Close polygon
    coords.append(coords[0])
    return coords


def generate_geojson(output_path: str) -> None:
    """Generates GeoJSON file for Bengaluru wards.

    Args:
        output_path: Destination filepath for GeoJSON.
    """
    features = []
    for ward in WARDS:
        poly_coords = create_ward_polygon(ward["lon"], ward["lat"])
        feature = {
            "type": "Feature",
            "properties": {
                "ward_id": ward["ward_id"],
                "ward_name": ward["name"],
                "zone": ward["zone"],
                "centroid_lat": ward["lat"],
                "centroid_lon": ward["lon"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly_coords],
            },
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"Generated GeoJSON with {len(features)} wards at {output_path}")


def generate_mock_datasets(
    start_date_str: str = "2024-08-16",
    end_date_str: str = "2024-09-15",
    risk_output_csv: str = "data/processed/risk_scores_fallback.csv",
    signals_output_csv: str = "data/processed/signals_fallback.csv",
) -> None:
    """Generates mock risk scores and signal events over a 30-day timeline.

    Ensures that 2024-09-14 for W017 strictly matches Section 7.2 of SPEC.md:
    risk_score=0.82, confidence=0.74, alert_level='high', baseline_flag=True.

    Args:
        start_date_str: ISO start date string.
        end_date_str: ISO end date string.
        risk_output_csv: Path to save fallback risk scores CSV.
        signals_output_csv: Path to save fallback signals CSV.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    num_days = (end_date - start_date).days + 1

    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Outbreak wards for realism: W017 (injected per spec), W010 (Koramangala), W012 (Bellandur)
    outbreak_wards = {"W017": 0.85, "W010": 0.72, "W012": 0.68, "W008": 0.55}

    risk_rows = []
    signal_rows = []

    base_time_ms = int(datetime(2024, 8, 16, 12, 0, 0).timestamp() * 1000)

    for day_idx, current_date in enumerate(dates):
        date_str = current_date.isoformat()
        day_progress = day_idx / max(1, num_days - 1)
        sim_timestamp_ms = base_time_ms + (day_idx * 86400 * 1000)

        # Seasonal city weather
        city_rainfall = round(max(0.0, float(np.random.gamma(shape=1.5, scale=12.0) if day_idx in [5, 6, 12, 13, 14, 20] else np.random.exponential(3.0))), 1)
        city_temp = round(float(25.5 + 2.0 * math.sin(day_idx / 5.0) + np.random.normal(0, 0.4)), 1)
        city_humidity = round(float(74.0 + 8.0 * math.sin(day_idx / 6.0) + np.random.normal(0, 1.5)), 1)

        for ward in WARDS:
            wid = ward["ward_id"]

            # Compute realistic ward signals
            is_outbreak = wid in outbreak_wards
            severity = outbreak_wards.get(wid, 0.2)

            # Signal multipliers
            escalation = (1.0 + (day_progress ** 1.8) * 3.5) if is_outbreak else (1.0 + np.random.normal(0, 0.08))
            base_clinic = 6.0 + 5.0 * severity
            clinic_visits = max(1.0, float(np.random.poisson(lam=max(1.0, base_clinic * escalation))))
            pharmacy_sales = max(2.0, float(np.random.poisson(lam=max(2.0, (base_clinic * 1.5) * escalation))))

            # Save Section 7.1 signal events
            signal_rows.extend([
                {"ward_id": wid, "date": date_str, "signal_type": "rainfall", "value": city_rainfall, "is_synthetic": False},
                {"ward_id": wid, "date": date_str, "signal_type": "temp", "value": city_temp, "is_synthetic": False},
                {"ward_id": wid, "date": date_str, "signal_type": "humidity", "value": city_humidity, "is_synthetic": False},
                {"ward_id": wid, "date": date_str, "signal_type": "clinic_fever_visits", "value": clinic_visits, "is_synthetic": True},
                {"ward_id": wid, "date": date_str, "signal_type": "pharmacy_fever_meds", "value": pharmacy_sales, "is_synthetic": True},
            ])

            # Calculate risk score
            if wid == "W017" and date_str == "2024-09-14":
                # EXACT match to Section 7.2 of SPEC.md
                risk_score = 0.82
                confidence = 0.74
                alert_level = "high"
                baseline_flag = True
                top_drivers = [
                    {"feature": "clinic_visits_7d_change_pct", "value": 41.0, "direction": "up", "text": "Clinic fever visits up 41% over 7 days"},
                    {"feature": "rainfall_lag14", "value": 62.5, "direction": "up", "text": "Heavy rainfall 14 days ago (62 mm)"},
                ]
                scored_at_ms = 1760500000000
                event_time_ms = 1760499999800
            else:
                if is_outbreak:
                    base_risk = 0.25 + (severity * 0.65) * (day_progress ** 1.3)
                    risk_score = round(min(0.96, max(0.12, float(base_risk + np.random.normal(0, 0.03)))), 2)
                else:
                    risk_score = round(min(0.48, max(0.04, float(0.12 + 0.15 * np.random.beta(2, 5)))), 2)

                alert_level = get_alert_level(risk_score)
                confidence = round(float(0.68 + 0.25 * (abs(risk_score - 0.55) * 1.5) + np.random.normal(0, 0.02)), 2)
                confidence = min(0.95, max(0.55, confidence))
                baseline_flag = bool(risk_score > 0.65 or (is_outbreak and day_idx > 18))

                # Synthesize drivers
                clinic_change = round(float((risk_score - 0.2) * 55 + np.random.normal(0, 4)), 1)
                rainfall_lag14 = round(float(city_rainfall * 1.8 + np.random.uniform(10, 45)), 1)

                if alert_level == "high":
                    top_drivers = [
                        {"feature": "clinic_visits_7d_change_pct", "value": clinic_change, "direction": "up", "text": f"Clinic fever visits up {int(clinic_change)}% over 7 days"},
                        {"feature": "rainfall_lag14", "value": rainfall_lag14, "direction": "up", "text": f"Heavy rainfall 14 days ago ({int(rainfall_lag14)} mm)"},
                    ]
                elif alert_level == "medium":
                    top_drivers = [
                        {"feature": "pharmacy_7d_change_pct", "value": round(clinic_change * 0.8, 1), "direction": "up", "text": f"Pharmacy antipyretic sales rising ({int(clinic_change * 0.8)}% 7d change)"},
                        {"feature": "humidity_7d_mean", "value": city_humidity, "direction": "up", "text": f"Elevated trailing humidity ({int(city_humidity)}%) favoring vector breeding"},
                    ]
                else:
                    top_drivers = [
                        {"feature": "clinic_visits_7d_mean", "value": round(clinic_visits, 1), "direction": "neutral", "text": "Fever visits within normal seasonal baseline"},
                    ]

                scored_at_ms = sim_timestamp_ms + 200
                event_time_ms = sim_timestamp_ms

            risk_rows.append({
                "ward_id": wid,
                "date": date_str,
                "risk_score": risk_score,
                "confidence": confidence,
                "alert_level": alert_level,
                "baseline_flag": baseline_flag,
                "top_drivers": json.dumps(top_drivers),
                "scored_at_ms": scored_at_ms,
                "event_time_ms": event_time_ms,
            })

    # Save to CSVs
    os.makedirs(os.path.dirname(risk_output_csv), exist_ok=True)
    df_risk = pd.DataFrame(risk_rows)
    df_risk.to_csv(risk_output_csv, index=False)
    print(f"Generated {len(df_risk)} risk score records in {risk_output_csv}")

    os.makedirs(os.path.dirname(signals_output_csv), exist_ok=True)
    df_signals = pd.DataFrame(signal_rows)
    df_signals.to_csv(signals_output_csv, index=False)
    print(f"Generated {len(df_signals)} signal events in {signals_output_csv}")


if __name__ == "__main__":
    geojson_path = os.path.join(os.path.dirname(__file__), "assets", "bengaluru_wards.geojson")
    generate_geojson(geojson_path)
    generate_mock_datasets()
