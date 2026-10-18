"""Downloads historical daily weather for Bengaluru from Open-Meteo archive API.

Adheres strictly to SPEC.md Section 4, 8, and Hard Rule 4:
- Real historical weather data from Open-Meteo (no API key required).
- Fetches daily rainfall (precipitation_sum), temperature (temperature_2m_mean),
  and humidity (relative_humidity_2m_mean) for Bengaluru (lat 12.97, lon 77.59).
- Saves output directly to data/raw/bengaluru_weather_2018_2025.csv.
- Fails loudly with a clear error if the request fails or data is invalid.
- Never generates or substitutes sample data.
"""

import argparse
import os
import sys
import pandas as pd
import requests


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
BENGALURU_LAT = 12.97
BENGALURU_LON = 77.59
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "raw", "bengaluru_weather_2018_2025.csv"
)


def fetch_bengaluru_weather(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    latitude: float = BENGALURU_LAT,
    longitude: float = BENGALURU_LON,
    timeout_sec: int = 30,
) -> pd.DataFrame:
    """Fetches daily rainfall, temperature, and humidity from Open-Meteo archive API.

    Args:
        start_date: ISO date string for start of historical window (e.g., '2018-01-01').
        end_date: ISO date string for end of historical window (e.g., '2025-12-31').
        latitude: Latitude coordinate for city center (default: 12.97 for Bengaluru).
        longitude: Longitude coordinate for city center (default: 77.59 for Bengaluru).
        timeout_sec: Request timeout in seconds.

    Returns:
        DataFrame with columns ['date', 'rainfall', 'temp', 'humidity'].

    Raises:
        RuntimeError: If Open-Meteo request fails, returns non-200, or provides empty data.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    print(f"Requesting weather from Open-Meteo for coordinates ({latitude}, {longitude}) from {start_date} to {end_date}...")

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout_sec)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to connect to Open-Meteo API ({OPEN_METEO_ARCHIVE_URL}): {e}. "
            "Please check internet connectivity. Never using synthetic fallback."
        ) from e

    if response.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo API returned error status {response.status_code}: {response.text}. "
            "Cannot proceed without real weather data."
        )

    data = response.json()
    if "daily" not in data:
        raise RuntimeError(
            f"Malformed Open-Meteo API response: missing 'daily' payload key. Raw response: {data}"
        )

    daily = data["daily"]
    required_keys = ["time", "precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean"]
    missing = [k for k in required_keys if k not in daily]
    if missing:
        raise RuntimeError(
            f"Open-Meteo API response missing required daily series: {missing}"
        )

    df = pd.DataFrame({
        "date": daily["time"],
        "rainfall": daily["precipitation_sum"],
        "temp": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
    })

    if df.empty:
        raise RuntimeError("Open-Meteo returned 0 rows for the specified date range.")

    return df


def main() -> None:
    """CLI entrypoint to download Bengaluru weather and save to data/raw/."""
    parser = argparse.ArgumentParser(
        description="Download historical daily Bengaluru weather from Open-Meteo (2018-2025)."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path in data/raw/")
    parser.add_argument("--lat", type=float, default=BENGALURU_LAT, help="City latitude")
    parser.add_argument("--lon", type=float, default=BENGALURU_LON, help="City longitude")
    args = parser.parse_args()

    df = fetch_bengaluru_weather(
        start_date=args.start_date,
        end_date=args.end_date,
        latitude=args.lat,
        longitude=args.lon,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Successfully saved {len(df)} daily weather records to: {args.output}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("Summary statistics:")
    print(df[["rainfall", "temp", "humidity"]].describe().round(2))


if __name__ == "__main__":
    main()
