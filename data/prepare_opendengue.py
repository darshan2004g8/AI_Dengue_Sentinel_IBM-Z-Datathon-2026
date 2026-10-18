"""data/prepare_opendengue.py
Processes OpenDengue SEARO extracts to generate official regional datasets:
  1. Identifies the Temporal and Spatial extracts from data/raw/ by inspecting T_res and S_res.
  2. Extracts national monthly cases for 2024 -> data/processed/india_monthly_2024.csv.
  3. Extracts Karnataka annual cases (1997-2024) -> data/processed/karnataka_yearly.csv.
  4. Checks for duplicate state-year records across data sources (UUIDs).
  5. Fails loudly with clear errors if any file or column is missing.
  6. Never generates or substitutes sample data (SPEC.md Hard Rule 4).
"""

import os
import sys
import pandas as pd


# Expected input filepaths
FILE_A = os.path.join("data", "raw", "filtered_data_SEARO_1792086539625.csv")
FILE_B = os.path.join("data", "raw", "filtered_data_SEARO_1792086576496.csv")

OUT_INDIA_MONTHLY = os.path.join("data", "processed", "india_monthly_2024.csv")
OUT_KARNATAKA_YEARLY = os.path.join("data", "processed", "karnataka_yearly.csv")

REQUIRED_COLUMNS = [
    "adm_0_name",
    "adm_1_name",
    "calendar_start_date",
    "calendar_end_date",
    "Year",
    "dengue_total",
    "S_res",
    "T_res",
    "UUID",
]


def validate_file(filepath: str) -> pd.DataFrame:
    """Validates existence and schema of an OpenDengue raw extract file.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the raw file is not found.
        ValueError: If any expected column is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Required OpenDengue raw file missing at: '{filepath}'. "
            "Never generating synthetic fallback per SPEC.md Hard Rule 4. "
            "Please download the dataset from OpenDengue."
        )

    df = pd.read_csv(filepath)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CRITICAL ERROR: File '{filepath}' is missing expected columns: {missing_cols}."
        )

    return df


def inspect_and_classify_extracts(df_a: pd.DataFrame, df_b: pd.DataFrame, path_a: str, path_b: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Inspects T_res and S_res distributions to classify temporal and spatial extracts.

    In OpenDengue:
    - The Temporal extract preserves continuous national time-series (Admin0) even when
      state-level (Admin1) data exists for those same years.
    - The Spatial extract prioritizes spatial granularity (Admin1) and deduplicates
      by pruning Admin0 rows where Admin1 is available.

    Args:
        df_a: DataFrame for File A.
        df_b: DataFrame for File B.
        path_a: Filepath for File A.
        path_b: Filepath for File B.

    Returns:
        Tuple of (df_temporal, df_spatial, path_temporal, path_spatial).
    """
    print("=" * 80)
    print("OPENDENGUE EXTRACT INSPECTION & CLASSIFICATION (S_res and T_res)")
    print("=" * 80)

    for label, df, p in [("File A", df_a, path_a), ("File B", df_b, path_b)]:
        print(f"\n{label}: {p}")
        print(f"  Total records: {len(df)}")
        print("  Spatial Resolution (S_res) counts:")
        for s_res, cnt in df["S_res"].value_counts().items():
            print(f"    - {s_res}: {cnt}")
        print("  Temporal Resolution (T_res) counts:")
        for t_res, cnt in df["T_res"].value_counts().items():
            print(f"    - {t_res}: {cnt}")
        print("  S_res x T_res Breakdown:")
        ct = pd.crosstab(df["S_res"], df["T_res"])
        print(ct.to_string())

    admin0_a = (df_a["S_res"] == "Admin0").sum()
    admin0_b = (df_b["S_res"] == "Admin0").sum()

    print("\nClassification Logic:")
    print(f"  - File A ({path_a}) has {admin0_a} Admin0 rows ({df_a[df_a['S_res'] == 'Admin0']['T_res'].value_counts().to_dict()}).")
    print(f"    It retains continuous national annual time series (1991-2014) alongside state data.")
    print(f"  - File B ({path_b}) has {admin0_b} Admin0 rows ({df_b[df_b['S_res'] == 'Admin0']['T_res'].value_counts().to_dict()}).")
    print(f"    It prunes Admin0 rows for 1997-2009 where Admin1 state breakdown is available to avoid spatial duplication.")

    if admin0_a >= admin0_b:
        temporal_df, spatial_df = df_a, df_b
        temporal_path, spatial_path = path_a, path_b
    else:
        temporal_df, spatial_df = df_b, df_a
        temporal_path, spatial_path = path_b, path_a

    print(f"\n>>> Identified TEMPORAL Extract: {temporal_path} (Full national temporal coverage)")
    print(f">>> Identified SPATIAL Extract:  {spatial_path} (Spatially deduplicated state coverage)")
    print("=" * 80)

    return temporal_df, spatial_df, temporal_path, spatial_path


def process_india_monthly(df_temporal: pd.DataFrame) -> pd.DataFrame:
    """Extracts national monthly dengue cases for Year 2024 from the temporal extract.

    Criteria:
      - S_res == "Admin0"
      - T_res == "Month"
      - Year == 2024

    Args:
        df_temporal: Temporal extract DataFrame.

    Returns:
        Filtered DataFrame with columns [calendar_start_date, calendar_end_date, dengue_total].
    """
    mask = (
        (df_temporal["S_res"] == "Admin0")
        & (df_temporal["T_res"] == "Month")
        & (df_temporal["Year"] == 2024)
    )
    df_filtered = df_temporal[mask].copy()

    if df_filtered.empty:
        raise ValueError(
            "CRITICAL ERROR: No records matched criteria (S_res=='Admin0', T_res=='Month', Year==2024) in temporal extract."
        )

    df_filtered = df_filtered.sort_values("calendar_start_date").reset_index(drop=True)
    out_df = df_filtered[["calendar_start_date", "calendar_end_date", "dengue_total"]].copy()

    os.makedirs(os.path.dirname(OUT_INDIA_MONTHLY), exist_ok=True)
    out_df.to_csv(OUT_INDIA_MONTHLY, index=False)
    print(f"\n[+] Saved {len(out_df)} monthly rows to: {OUT_INDIA_MONTHLY}")

    return out_df


def process_karnataka_yearly(df_spatial: pd.DataFrame) -> pd.DataFrame:
    """Extracts annual dengue cases for Karnataka from the spatial extract.

    Criteria:
      - adm_1_name == "KARNATAKA"
      - T_res == "Year"
    Also checks for duplicate state-year rows from different UUID sources.

    Args:
        df_spatial: Spatial extract DataFrame.

    Returns:
        Filtered DataFrame with Karnataka annual cases.
    """
    mask = (
        (df_spatial["adm_1_name"].astype(str).str.upper() == "KARNATAKA")
        & (df_spatial["T_res"] == "Year")
    )
    df_filtered = df_spatial[mask].copy()

    if df_filtered.empty:
        raise ValueError(
            "CRITICAL ERROR: No records found for adm_1_name=='KARNATAKA' and T_res=='Year' in spatial extract."
        )

    df_filtered["Year"] = df_filtered["Year"].astype(int)
    df_filtered = df_filtered.sort_values("Year").reset_index(drop=True)

    # Check for duplicate state-year rows from different UUID sources
    print("\n" + "-" * 60)
    print("CHECKING FOR DUPLICATE STATE-YEAR ROWS (KARNATAKA)")
    print("-" * 60)

    dupes = df_filtered[df_filtered.duplicated(subset=["adm_1_name", "Year"], keep=False)]
    if not dupes.empty:
        print(f"[!] WARNING: Found {len(dupes)} duplicate state-year row(s) from different UUID sources:")
        print(dupes[["adm_1_name", "Year", "dengue_total", "UUID", "calendar_start_date", "calendar_end_date"]].to_string(index=False))
    else:
        print(f"[+] Clean data verified: 0 duplicate state-year rows found.")
        print(f"    Total unique years: {len(df_filtered)} ({df_filtered['Year'].min()} - {df_filtered['Year'].max()})")
        print("    Data sources (UUIDs) contributing across time:")
        for uuid, grp in df_filtered.groupby("UUID"):
            print(f"      - {uuid}: Years {grp['Year'].min()} - {grp['Year'].max()} ({len(grp)} years)")

    os.makedirs(os.path.dirname(OUT_KARNATAKA_YEARLY), exist_ok=True)
    df_filtered.to_csv(OUT_KARNATAKA_YEARLY, index=False)
    print(f"\n[+] Saved {len(df_filtered)} Karnataka annual records to: {OUT_KARNATAKA_YEARLY}")

    return df_filtered


def main():
    """Main execution entrypoint."""
    # Validate inputs
    df_a = validate_file(FILE_A)
    df_b = validate_file(FILE_B)

    # Classify temporal vs spatial
    df_temporal, df_spatial, path_temporal, path_spatial = inspect_and_classify_extracts(
        df_a, df_b, FILE_A, FILE_B
    )

    # Process and save India monthly 2024
    india_monthly_df = process_india_monthly(df_temporal)

    # Process and save Karnataka yearly
    karnataka_yearly_df = process_karnataka_yearly(df_spatial)

    # Print both output files in full
    print("\n" + "=" * 80)
    print(f"PRINTING OUTPUT FILE 1 IN FULL (CSV): {OUT_INDIA_MONTHLY}")
    print("=" * 80)
    with open(OUT_INDIA_MONTHLY, "r") as f:
        print(f.read().strip())

    print("\n" + "=" * 80)
    print(f"PRINTING OUTPUT FILE 2 IN FULL (CSV): {OUT_KARNATAKA_YEARLY}")
    print("=" * 80)
    with open(OUT_KARNATAKA_YEARLY, "r") as f:
        print(f.read().strip())
    print("=" * 80)


if __name__ == "__main__":
    main()
