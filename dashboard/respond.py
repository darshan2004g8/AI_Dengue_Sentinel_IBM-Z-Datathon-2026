"""
dashboard/respond.py
Operational Decision & Response Engine for Dengue Sentinel.
Adheres strictly to BUILD.md Section 9.4:
  - If ward population exists: exposure = risk_score * population * (1 + vulnerability).
    Benefit of n visits = exposure * (1 - (1 - r)^n), with r = 0.30.
    Assigns each team to the ward with highest marginal benefit (greedy).
  - Otherwise ranks wards by risk_score descending.
  - Only confirmed medium or high alert wards are eligible for team dispatch.
  - Wards above team capacity are placed on the Watchlist.
  - Plain-language action text per Section 9.4:
      medium: "Schedule inspection and larval source removal in {ward} within 72 hours."
      high: "Prioritise inspection and larval source removal in {ward} within 48 hours; consider targeted fogging at the cluster; notify nearby hospitals to prepare for more fever cases."
  - Explicit UI disclaimer: "Priority scores rank wards; they do not predict cases averted."
"""

import json
from typing import Any
import pandas as pd


UI_DISCLAIMER = "Priority scores rank wards; they do not predict cases averted."


def generate_ward_action(
    ward_id: str,
    ward_name: str,
    risk_score: float,
    alert_level: str,
    top_drivers: list[dict[str, Any]] | None = None,
    team_rank: int | None = None,
) -> dict[str, str]:
    """
    Generate plain-language operational directives and hospital advisories per Section 9.4.
    
    Inputs:
        ward_id (str): Ward identifier (e.g. 'W017').
        ward_name (str): Ward name.
        risk_score (float): Outbreak probability.
        alert_level (str): 'high', 'medium', or 'low'.
        top_drivers (list of dict, optional): List of top driver dictionaries.
        team_rank (int, optional): Priority order.
    Outputs:
        dict: Operational directives, hospital advisory, and urgency text.
    """
    drivers = top_drivers or []
    driver_texts = [d.get("text", "") for d in drivers if isinstance(d, dict)]

    if alert_level == "high":
        operational = (
            f"Prioritise inspection and larval source removal in {ward_id} ({ward_name}) within 48 hours; "
            f"consider targeted fogging at the cluster; notify nearby hospitals to prepare for more fever cases."
        )
        hospital = (
            f"Notify hospitals and PHCs serving {ward_name} to prepare for fever surge, "
            f"stock NS1 antigen test kits, and activate outpatient triage."
        )
        urgency = "IMMEDIATE (24-48h)"
    elif alert_level == "medium":
        operational = (
            f"Schedule inspection and larval source removal in {ward_id} ({ward_name}) within 72 hours."
        )
        hospital = (
            f"Alert community health workers in {ward_name} to step up active surveillance "
            f"and issue neighborhood vector prevention advisories."
        )
        urgency = "PRIORITY (72-96h)"
    else:
        operational = f"Maintain routine seasonal surveillance in {ward_id} ({ward_name}). No team dispatch required."
        hospital = "Standard epidemiological surveillance."
        urgency = "ROUTINE"

    driver_context = "; ".join(driver_texts) if driver_texts else "Signals within normal bounds."

    return {
        "operational_action": operational,
        "hospital_advisory": hospital,
        "driver_context": driver_context,
        "urgency": urgency,
    }


def compute_marginal_benefit(current_teams, exposure, r=0.30):
    """
    Calculate marginal benefit gained by deploying one additional team to a ward.
    Benefit(n) = exposure * (1 - (1 - r)^n).
    
    Inputs:
        current_teams (int): Current teams assigned to ward.
        exposure (float): Exposure metric (risk_score * population * (1 + vulnerability)).
        r (float): Diminishing effectiveness rate (default 0.30).
    Outputs:
        float: Marginal gain of next team.
    """
    curr_b = exposure * (1.0 - (1.0 - r) ** current_teams)
    next_b = exposure * (1.0 - (1.0 - r) ** (current_teams + 1))
    return next_b - curr_b


def allocate_response_teams(
    current_scores: list[dict[str, Any]] | pd.DataFrame,
    num_teams: int = 5,
    ward_names_map: dict[str, str] | None = None,
    ward_population_map: dict[str, float] | None = None,
    ward_vulnerability_map: dict[str, float] | None = None,
    r: float = 0.30,
) -> dict[str, Any]:
    """
    Allocate discrete response teams to high-risk municipal wards.
    If population exists: uses exposure-weighted greedy marginal benefit.
    Otherwise: ranks by risk_score descending.
    
    Inputs:
        current_scores: List of dicts or DataFrame of latest risk outputs.
        num_teams (int): Total available field teams (1 to 15, default 5).
        ward_names_map (dict, optional): Map from ward_id to human-readable name.
        ward_population_map (dict, optional): Map from ward_id to population.
        ward_vulnerability_map (dict, optional): Map from ward_id to vulnerability index.
        r (float): Diminishing returns factor per team (default 0.30).
    Outputs:
        dict: Allocation summary, dispatched teams, watchlist, and low-risk list.
    """
    if isinstance(current_scores, pd.DataFrame):
        records = current_scores.to_dict(orient="records")
    else:
        records = list(current_scores)

    names = ward_names_map or {}
    pops = ward_population_map or {}
    vulns = ward_vulnerability_map or {}

    # Eligible wards: only confirmed medium and high alerts (alert_level != 'low')
    actionable = [r for r in records if str(r.get("alert_level", "")).lower() in ("medium", "high")]
    low_risk = [r for r in records if str(r.get("alert_level", "")).lower() not in ("medium", "high")]

    # Calculate exposure for actionable wards
    has_population = bool(pops and any(wid in pops for wid in [r.get("ward_id") for r in actionable]))
    for w in actionable:
        wid = str(w.get("ward_id", ""))
        rscore = float(w.get("risk_score", 0.0))
        pop = float(pops.get(wid, 10000.0 if has_population else 1.0))
        vuln = float(vulns.get(wid, 0.0))
        w["_exposure"] = rscore * pop * (1.0 + vuln)
        w["_teams_assigned"] = 0

    if has_population and actionable and num_teams > 0:
        # Greedy marginal allocation
        for _ in range(num_teams):
            marginal_gains = [
                compute_marginal_benefit(w["_teams_assigned"], w["_exposure"], r)
                for w in actionable
            ]
            best_idx = int(pd.Series(marginal_gains).argmax())
            actionable[best_idx]["_teams_assigned"] += 1
        
        # Sort primarily by teams assigned, then exposure
        actionable.sort(key=lambda x: (x["_teams_assigned"], x["_exposure"]), reverse=True)
    else:
        # Rank strictly by risk_score descending
        actionable.sort(key=lambda x: float(x.get("risk_score", 0.0)), reverse=True)
        for idx in range(min(num_teams, len(actionable))):
            actionable[idx]["_teams_assigned"] = 1

    assigned_teams = []
    watchlist = []

    for idx, ward in enumerate(actionable):
        wid = str(ward.get("ward_id", ""))
        wname = names.get(wid, wid)
        rscore = float(ward.get("risk_score", 0.0))
        alevel = str(ward.get("alert_level", "medium")).lower()
        conf = float(ward.get("confidence", 0.0))
        b_flag = bool(ward.get("baseline_flag", False))
        teams_for_ward = ward.get("_teams_assigned", 0)

        raw_drivers = ward.get("top_drivers", [])
        if isinstance(raw_drivers, str):
            try:
                drivers = json.loads(raw_drivers)
            except Exception:
                drivers = []
        else:
            drivers = raw_drivers or []

        actions = generate_ward_action(
            ward_id=wid,
            ward_name=wname,
            risk_score=rscore,
            alert_level=alevel,
            top_drivers=drivers,
            team_rank=idx + 1,
        )

        item = {
            "rank": idx + 1,
            "ward_id": wid,
            "ward_name": wname,
            "risk_score": rscore,
            "confidence": conf,
            "alert_level": alevel,
            "baseline_flag": b_flag,
            "teams_allocated": teams_for_ward,
            "operational_action": actions["operational_action"],
            "hospital_advisory": actions["hospital_advisory"],
            "driver_context": actions["driver_context"],
            "urgency": actions["urgency"],
        }

        if teams_for_ward > 0:
            item["team_id"] = f"Team-{len(assigned_teams) + 1:02d}"
            item["status"] = "DISPATCHED"
            assigned_teams.append(item)
        else:
            item["status"] = "WATCHLIST"
            item["contingency"] = "Monitor clinic and pharmacy signals daily; prioritize if additional teams become available."
            watchlist.append(item)

    teams_deployed = sum(w.get("teams_allocated", 1) for w in assigned_teams)
    summary = {
        "num_teams_available": num_teams,
        "teams_deployed": min(teams_deployed, num_teams),
        "teams_idle": max(0, num_teams - teams_deployed),
        "actionable_wards_count": len(actionable),
        "high_risk_count": sum(1 for r in actionable if str(r.get("alert_level")).lower() == "high"),
        "medium_risk_count": sum(1 for r in actionable if str(r.get("alert_level")).lower() == "medium"),
        "watchlist_count": len(watchlist),
        "low_risk_count": len(low_risk),
        "allocation_mode": "exposure_greedy" if has_population else "risk_rank",
        "ui_disclaimer": UI_DISCLAIMER
    }

    return {
        "assigned_teams": assigned_teams,
        "watchlist": watchlist,
        "low_risk_wards": low_risk,
        "summary": summary,
    }
