from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "ssa_trained_model.json"
SSA_TO_APP_REGION = {
    "BOS": "northeast",
    "NYC": "northeast",
    "PHL": "northeast",
    "ATL": "south",
    "DAL": "south",
    "CHI": "midwest",
    "KCM": "midwest",
    "DEN": "west",
    "SEA": "west",
    "SFO": "west",
}
SSA_TO_ALJ_REGION = {
    "BOS": "01",
    "NYC": "02",
    "PHL": "03",
    "ATL": "04",
    "CHI": "05",
    "DAL": "06",
    "KCM": "07",
    "DEN": "08",
    "SFO": "09",
    "SEA": "10",
}
STATE_TO_SSA_REGION = {
    "AL": "ATL",
    "AK": "SEA",
    "AZ": "SFO",
    "AR": "DAL",
    "CA": "SFO",
    "CO": "DEN",
    "CT": "BOS",
    "DE": "PHL",
    "DC": "PHL",
    "FL": "ATL",
    "GA": "ATL",
    "HI": "SFO",
    "ID": "SEA",
    "IL": "CHI",
    "IN": "CHI",
    "IA": "KCM",
    "KS": "KCM",
    "KY": "ATL",
    "LA": "DAL",
    "ME": "BOS",
    "MD": "PHL",
    "MA": "BOS",
    "MI": "CHI",
    "MN": "CHI",
    "MS": "ATL",
    "MO": "KCM",
    "MT": "DEN",
    "NE": "KCM",
    "NV": "SFO",
    "NH": "BOS",
    "NJ": "NYC",
    "NM": "DAL",
    "NY": "NYC",
    "NC": "ATL",
    "ND": "DEN",
    "OH": "CHI",
    "OK": "DAL",
    "OR": "SEA",
    "PA": "PHL",
    "PR": "NYC",
    "RI": "BOS",
    "SC": "ATL",
    "SD": "DEN",
    "TN": "ATL",
    "TX": "DAL",
    "UT": "DEN",
    "VT": "BOS",
    "VA": "PHL",
    "WA": "SEA",
    "WV": "PHL",
    "WI": "CHI",
    "WY": "DEN",
}


@dataclass(frozen=True)
class BaselineEstimate:
    probability: float
    source: str
    detail: str
    artifact_loaded: bool


@dataclass(frozen=True)
class StageBaseline:
    initial: float
    reconsideration: float
    alj_fully_favorable: float
    alj_partially_favorable: float
    alj_denial: float
    source: str


@dataclass(frozen=True)
class StageTiming:
    initial_months: float
    reconsideration_months: float
    alj_hearing_months: float
    source: str


def estimate_baseline(claimant: dict[str, Any]) -> BaselineEstimate:
    artifact = load_artifact()
    if not artifact:
        return BaselineEstimate(
            probability=0.38,
            source="Fallback baseline",
            detail="SSA trained artifact not found; run scripts/train_ssa_model.py.",
            artifact_loaded=False,
        )

    baseline = float(artifact["baseline_probability"])
    state = claimant.get("state")
    if state:
        state_rates = artifact.get("recent_state_probabilities", {})
        if state in state_rates:
            return BaselineEstimate(
                probability=float(state_rates[state]),
                source="SSA state baseline",
                detail=f"{state} recent adult favorable determination rate",
                artifact_loaded=True,
            )

    ssa_region = _ssa_region_for_claimant(claimant)
    region_rates = artifact.get("region_probabilities", {})
    if ssa_region in region_rates:
        return BaselineEstimate(
            probability=float(region_rates[ssa_region]),
            source="SSA region baseline",
            detail=f"{ssa_region} adult favorable determination rate",
            artifact_loaded=True,
        )

    return BaselineEstimate(
        probability=baseline,
        source="SSA national baseline",
        detail="National adult favorable determination rate",
        artifact_loaded=True,
    )


def estimate_stage_baselines(claimant: dict[str, Any]) -> StageBaseline:
    artifact = load_artifact()
    if not artifact:
        return StageBaseline(
            initial=0.32,
            reconsideration=0.12,
            alj_fully_favorable=0.43,
            alj_partially_favorable=0.08,
            alj_denial=0.49,
            source="Fallback stage baselines",
        )

    level_baselines = artifact.get("level_baselines", {})
    state = claimant.get("state")
    ssa_region = _ssa_region_for_claimant(claimant)
    alj_region = SSA_TO_ALJ_REGION.get(ssa_region or "")

    initial = _lookup_stage_rate(artifact, "initial", state, ssa_region)
    reconsideration = _lookup_stage_rate(artifact, "reconsideration", state, ssa_region)
    alj_rates = artifact.get("level_region_probabilities", {}).get("alj", {})
    alj = alj_rates.get(alj_region, {}) if alj_region else {}

    return StageBaseline(
        initial=initial if initial is not None else float(level_baselines.get("initial", 0.32)),
        reconsideration=(
            reconsideration
            if reconsideration is not None
            else float(level_baselines.get("reconsideration", 0.12))
        ),
        alj_fully_favorable=float(
            alj.get("fully_favorable", level_baselines.get("alj_fully_favorable", 0.43))
        ),
        alj_partially_favorable=float(
            alj.get("partially_favorable", level_baselines.get("alj_partially_favorable", 0.08))
        ),
        alj_denial=float(alj.get("denial", level_baselines.get("alj_denial", 0.49))),
        source="SSA stage baselines",
    )


def estimate_stage_timing(claimant: dict[str, Any]) -> StageTiming:
    artifact = load_artifact()
    if not artifact:
        return StageTiming(
            initial_months=6.0,
            reconsideration_months=4.0,
            alj_hearing_months=18.0,
            source="Fallback timing estimates",
        )

    timing = artifact.get("decision_time_months", {})
    baselines = timing.get("baselines", {})
    state = claimant.get("state")
    ssa_region = _ssa_region_for_claimant(claimant)
    alj_region = SSA_TO_ALJ_REGION.get(ssa_region or "")

    return StageTiming(
        initial_months=_lookup_timing(timing, "initial", state, ssa_region)
        or float(baselines.get("initial", 6.0)),
        reconsideration_months=_lookup_timing(timing, "reconsideration", state, ssa_region)
        or float(baselines.get("reconsideration", 4.0)),
        alj_hearing_months=_lookup_timing(timing, "alj", None, alj_region)
        or float(baselines.get("alj_hearing", 18.0)),
        source="SSA workload timing estimates",
    )


def load_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def app_region_for_state(state: str) -> str | None:
    ssa_region = STATE_TO_SSA_REGION.get(state)
    if not ssa_region:
        return None
    return SSA_TO_APP_REGION.get(ssa_region)


def _ssa_region_for_claimant(claimant: dict[str, Any]) -> str | None:
    state = claimant.get("state")
    if state:
        return STATE_TO_SSA_REGION.get(state)

    app_region = claimant.get("region")
    for ssa_region, mapped_app_region in SSA_TO_APP_REGION.items():
        if app_region == mapped_app_region:
            return ssa_region
    return None


def _lookup_stage_rate(
    artifact: dict[str, Any],
    stage: str,
    state: str | None,
    ssa_region: str | None,
) -> float | None:
    stage_rates = artifact.get("level_state_probabilities", {}).get(stage, {})
    if state and state in stage_rates:
        return float(stage_rates[state])

    region_rates = artifact.get("level_region_probabilities", {}).get(stage, {})
    if ssa_region and ssa_region in region_rates:
        return float(region_rates[ssa_region])

    return None


def _lookup_timing(
    timing: dict[str, Any],
    stage: str,
    state: str | None,
    ssa_region: str | None,
) -> float | None:
    state_timing = timing.get("state", {}).get(stage, {})
    if state and state in state_timing:
        return float(state_timing[state])

    region_timing = timing.get("region", {}).get(stage, {})
    if ssa_region and ssa_region in region_timing:
        return float(region_timing[ssa_region])

    return None
