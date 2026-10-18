"""data/convert_wards_kml.py
Converts BBMP KML ward map to GeoJSON format using geopandas.
Adheres strictly to SPEC.md and BUILD.md:
  - Preserves spatial geometry and coordinate system (EPSG:4326).
  - Handles packed HTML description fields or schema-extracted attributes.
  - Generates standardized ward_no, ward_name, and ward_id properties.
  - Validates bounding box against expected Bengaluru metropolitan limits.
"""

import os
import re
import sys
from html import unescape
import geopandas as gpd


# File paths
KML_PRIMARY = os.path.join("data", "raw", "BBMP-WARD-MAP.kml")
KML_FALLBACK = os.path.join("data", "raw", "BBPM-WARD-MAP.kml")
OUT_GEOJSON = os.path.join("data", "raw", "wards.geojson")

# Expected Bengaluru geographic limits
EXPECTED_LAT_MIN = 12.7
EXPECTED_LAT_MAX = 13.3
EXPECTED_LON_MIN = 77.3
EXPECTED_LON_MAX = 77.9


def parse_html_description(desc: str) -> tuple[str | None, str | None]:
    """Parses ward number and ward name if packed inside an HTML description.

    Args:
        desc: Raw description string which may contain HTML markup.

    Returns:
        Tuple of (ward_no, ward_name) or (None, None) if not parsed.
    """
    if not isinstance(desc, str) or not desc.strip():
        return None, None

    desc_clean = unescape(desc)

    # 1. Look for explicit ward number key-value patterns in HTML/text
    ward_no = None
    ward_name = None

    no_match = re.search(
        r"(?:ward[-_\s]*(?:no|num|number)|id)[\s:<>\w/=\"\']*?(\d+)",
        desc_clean,
        re.IGNORECASE,
    )
    if not no_match:
        no_match = re.search(
            r"<td>(?:ward[-_\s]*(?:no|num|number)|id)</td>\s*<td>([^<]+)</td>",
            desc_clean,
            re.IGNORECASE,
        )
    if no_match:
        ward_no = no_match.group(1).strip()

    name_match = re.search(
        r"(?:ward[-_\s]*name|name)[\s:<>\w/=\"\']*?([A-Za-z0-9\s\-_]+?)(?:<|$)",
        desc_clean,
        re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"<td>(?:ward[-_\s]*name|name)</td>\s*<td>([^<]+)</td>",
            desc_clean,
            re.IGNORECASE,
        )
    if name_match:
        ward_name = name_match.group(1).strip()

    # 2. Look for hyphenated composite like "3 - Atturu" or "17: Chikkasandra"
    if not ward_no or not ward_name:
        plain_text = re.sub(r"<[^>]+>", " ", desc_clean).strip()
        combo_match = re.match(r"^\s*(\d+)\s*[-:]\s*([A-Za-z0-9\s\-_]+)", plain_text)
        if combo_match:
            if not ward_no:
                ward_no = combo_match.group(1).strip()
            if not ward_name:
                ward_name = combo_match.group(2).strip()

    return ward_no, ward_name


def convert_kml_to_geojson():
    """Main execution function to convert KML to wards.geojson."""
    # Locate KML file
    if os.path.exists(KML_PRIMARY):
        kml_path = KML_PRIMARY
    elif os.path.exists(KML_FALLBACK):
        kml_path = KML_FALLBACK
    else:
        raise FileNotFoundError(
            f"CRITICAL ERROR: Neither '{KML_PRIMARY}' nor '{KML_FALLBACK}' exists in data/raw/. "
            "Please ensure the KML boundary map is present."
        )

    print("=" * 80)
    print(f"LOADING KML: {kml_path}")
    print("=" * 80)

    gdf = gpd.read_file(kml_path)

    # Check whether ward number and name are packed in an HTML description field
    has_html_desc = (
        "description" in gdf.columns
        and gdf["description"].notna().any()
        and gdf["description"].str.contains("<", na=False).any()
    )

    if has_html_desc:
        print("[*] HTML description field detected. Parsing ward_no and ward_name from HTML...")
        parsed = gdf["description"].apply(parse_html_description)
        gdf["ward_no"] = [p[0] for p in parsed]
        gdf["ward_name"] = [p[1] for p in parsed]
    else:
        print("[*] Description field is empty or non-HTML. Deriving ward_no and ward_name from schema fields...")
        # id2 holds the ward number from ExtendedData SchemaData
        if "id2" in gdf.columns and gdf["id2"].notna().any():
            gdf["ward_no"] = gdf["id2"].astype(int)
        elif "id" in gdf.columns and gdf["id"].notna().any():
            gdf["ward_no"] = gdf["id"].astype(int)
        elif "proposed_ward_name_en" in gdf.columns:
            gdf["ward_no"] = gdf["proposed_ward_name_en"].str.extract(r"^(\d+)", expand=False).astype(int)

        # name_en holds the ward name
        if "name_en" in gdf.columns and gdf["name_en"].notna().any():
            gdf["ward_name"] = gdf["name_en"].astype(str)
        elif "proposed_ward_name_en" in gdf.columns:
            gdf["ward_name"] = gdf["proposed_ward_name_en"].str.replace(r"^\d+[-_\s]*", "", regex=True)

    # Format standardized ward_id
    if "ward_no" in gdf.columns:
        gdf["ward_id"] = gdf["ward_no"].apply(lambda x: f"W{int(x):03d}")

    # Reorder columns to place key identifiers first
    priority_cols = ["ward_id", "ward_no", "ward_name"]
    other_cols = [c for c in gdf.columns if c not in priority_cols and c != "geometry"]
    gdf = gdf[priority_cols + other_cols + ["geometry"]]

    # Save to GeoJSON
    os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)
    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"\n[+] Successfully saved GeoJSON to: {OUT_GEOJSON}")

    # Print requested statistics
    print("\n" + "=" * 80)
    print("GEOJSON DATASET SUMMARY")
    print("=" * 80)
    print(f"Feature count: {len(gdf)}")
    print(f"\nProperty names ({len(priority_cols + other_cols)} properties):")
    for i, prop in enumerate(priority_cols + other_cols, 1):
        print(f"  {i:2d}. {prop}")

    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"\nBounding Box (minx, miny, maxx, maxy):")
    print(f"  minx (lon_min): {minx:.6f}")
    print(f"  miny (lat_min): {miny:.6f}")
    print(f"  maxx (lon_max): {maxx:.6f}")
    print(f"  maxy (lat_max): {maxy:.6f}")

    print("\nFirst 5 ward IDs and names:")
    for idx, r in gdf.head(5).iterrows():
        print(f"  {idx + 1}. ward_id: {r['ward_id']} | ward_no: {r['ward_no']} | ward_name: {r['ward_name']}")

    # Bounding box validation
    lat_ok = EXPECTED_LAT_MIN <= miny and maxy <= EXPECTED_LAT_MAX
    lon_ok = EXPECTED_LON_MIN <= minx and maxx <= EXPECTED_LON_MAX
    bounds_ok = lat_ok and lon_ok

    print("\n" + "-" * 60)
    print("BOUNDING BOX VALIDATION (lat: [12.7, 13.3], lon: [77.3, 77.9])")
    print("-" * 60)
    print(f"  Latitude check: [{miny:.4f}, {maxy:.4f}] inside [12.7, 13.3] -> {'PASS' if lat_ok else 'FAIL'}")
    print(f"  Longitude check: [{minx:.4f}, {maxx:.4f}] inside [77.3, 77.9] -> {'PASS' if lon_ok else 'FAIL'}")
    if bounds_ok:
        print("  Overall Check: PASS (Bounding box strictly inside target bounds)")
    else:
        print("  CRITICAL ERROR: Bounding box falls OUTSIDE the expected range!")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    convert_kml_to_geojson()
