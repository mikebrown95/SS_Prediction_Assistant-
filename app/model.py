from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log
from typing import Any

try:
    from .ssa_rules import assess_ssdi_rules
    from .trained_baseline import (
        STATE_TO_SSA_REGION,
        app_region_for_state,
        estimate_baseline,
        estimate_stage_baselines,
        estimate_stage_timing,
    )
except ImportError:
    from ssa_rules import assess_ssdi_rules
    from trained_baseline import (
        STATE_TO_SSA_REGION,
        app_region_for_state,
        estimate_baseline,
        estimate_stage_baselines,
        estimate_stage_timing,
    )


CONDITION_WEIGHTS = {
    "musculoskeletal": 0.35,
    "mental_health": 0.28,
    "cardiovascular": 0.5,
    "neurological": 0.58,
    "cancer": 0.72,
    "respiratory": 0.42,
    "immune": 0.38,
    "endocrine": 0.18,
    "other": 0.0,
}

EDUCATION_WEIGHTS = {
    "less_than_high_school": 0.28,
    "high_school": 0.16,
    "some_college": 0.04,
    "bachelors": -0.12,
    "graduate": -0.2,
}

REGION_WEIGHTS = {
    "northeast": -0.02,
    "midwest": 0.03,
    "south": 0.08,
    "west": -0.04,
    "territory": 0.02,
}

WORK_LEVEL_WEIGHTS = {
    "none": 0.42,
    "sedentary": 0.18,
    "light": 0.02,
    "medium": -0.12,
    "heavy": -0.22,
}

SEX_WEIGHTS = {
    "female": 0.02,
    "male": 0.0,
    "another": 0.0,
    "prefer_not_to_say": 0.0,
}


@dataclass(frozen=True)
class Prediction:
    probability: float
    band: str
    score: float
    factors: list[dict[str, str | float]]
    award_levels: list[dict[str, str | float]]
    predicted_award_level: str
    predicted_outcome: str
    rule_findings: list[dict[str, str]]


def predict_award_probability(payload: dict[str, Any]) -> Prediction:
    claimant = _parse_payload(payload)
    score = _score_claimant(claimant)
    rules = assess_ssdi_rules(claimant)
    probability = _apply_rule_cap(_logistic(score), rules.probability_cap)
    award_levels = _award_level_probabilities(claimant, probability)

    return Prediction(
        probability=round(probability, 4),
        band=_probability_band(probability),
        score=round(score, 3),
        factors=_factor_explanations(claimant),
        award_levels=award_levels,
        predicted_award_level=_predicted_award_level(award_levels),
        predicted_outcome=_predicted_outcome(award_levels),
        rule_findings=[
            {"rule": finding.rule, "status": finding.status, "detail": finding.detail}
            for finding in rules.findings
        ],
    )


def _parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    age = _number(payload.get("age"), "age", minimum=18, maximum=67)
    years_worked = _number(payload.get("yearsWorked"), "yearsWorked", minimum=0, maximum=50)
    impairment_months = _number(
        payload.get("impairmentMonths"),
        "impairmentMonths",
        minimum=0,
        maximum=600,
    )

    state = payload.get("state") or ""
    if state:
        state = _choice(str(state).upper(), "state", {key: 0 for key in STATE_TO_SSA_REGION})

    claimant = {
        "age": age,
        "education": _choice(payload.get("education"), "education", EDUCATION_WEIGHTS),
        "region": _choice(
            app_region_for_state(state) if state else payload.get("region"),
            "region",
            REGION_WEIGHTS,
        ),
        "state": state,
        "condition": _choice(payload.get("condition"), "condition", CONDITION_WEIGHTS),
        "conditionSeverity": _optional_number(
            payload.get("conditionSeverity"),
            "conditionSeverity",
            minimum=1,
            maximum=10,
            default=5.5,
        ),
        "workLevel": _choice(payload.get("workLevel"), "workLevel", WORK_LEVEL_WEIGHTS),
        "sex": _choice(payload.get("sex"), "sex", SEX_WEIGHTS),
        "yearsWorked": years_worked,
        "impairmentMonths": impairment_months,
        "monthlyEarnings": _optional_number(
            payload.get("monthlyEarnings"),
            "monthlyEarnings",
            minimum=0,
            maximum=100000,
            default=0,
        ),
        "isBlind": bool(payload.get("isBlind")),
        "expectedToResultInDeath": bool(payload.get("expectedToResultInDeath")),
        "hasSpecialistEvidence": bool(payload.get("hasSpecialistEvidence")),
        "hasRecentWorkAttempt": bool(payload.get("hasRecentWorkAttempt")),
        "dateOfDisability": _optional_date(payload.get("dateOfDisability"), "dateOfDisability"),
    }

    return claimant


def _score_claimant(claimant: dict[str, Any]) -> float:
    baseline = estimate_baseline(claimant)
    age = claimant["age"]
    age_component = max(0.0, (age - 45) / 22) * 0.85
    severity_component = (claimant["conditionSeverity"] - 5.5) * 0.28
    duration_component = min(claimant["impairmentMonths"], 36) / 36 * 0.42
    work_history_component = min(claimant["yearsWorked"], 35) / 35 * 0.2
    specialist_component = 0.28 if claimant["hasSpecialistEvidence"] else -0.18
    recent_work_component = -0.3 if claimant["hasRecentWorkAttempt"] else 0.0

    return (
        _logit(baseline.probability)
        + age_component
        + severity_component
        + duration_component
        + work_history_component
        + specialist_component
        + recent_work_component
        + CONDITION_WEIGHTS[claimant["condition"]]
        + EDUCATION_WEIGHTS[claimant["education"]]
        + REGION_WEIGHTS[claimant["region"]]
        + WORK_LEVEL_WEIGHTS[claimant["workLevel"]]
        + SEX_WEIGHTS[claimant["sex"]]
    )


def _factor_explanations(claimant: dict[str, Any]) -> list[dict[str, str | float]]:
    baseline = estimate_baseline(claimant)
    return [
        {
            "label": baseline.source,
            "impact": "neutral",
            "detail": baseline.detail,
        },
        {
            "label": "Age",
            "impact": "higher" if claimant["age"] >= 50 else "lower",
            "detail": f"{claimant['age']:.0f} years old",
        },
        {
            "label": "Condition",
            "impact": _impact(CONDITION_WEIGHTS[claimant["condition"]]),
            "detail": claimant["condition"].replace("_", " ").title(),
        },
        {
            "label": "Expected duration",
            "impact": "higher" if claimant["impairmentMonths"] >= 12 else "lower",
            "detail": f"{claimant['impairmentMonths']:.0f} months",
        },
        {
            "label": "Education",
            "impact": _impact(EDUCATION_WEIGHTS[claimant["education"]]),
            "detail": claimant["education"].replace("_", " ").title(),
        },
        {
            "label": "Prior work level",
            "impact": _impact(WORK_LEVEL_WEIGHTS[claimant["workLevel"]]),
            "detail": claimant["workLevel"].title(),
        },
    ]


def _award_level_probabilities(
    claimant: dict[str, Any],
    overall_probability: float,
) -> list[dict[str, str | float]]:
    stage = estimate_stage_baselines(claimant)
    timing = estimate_stage_timing(claimant)
    initial_months = max(timing.initial_months, 0.0)
    reconsideration_months = initial_months + max(timing.reconsideration_months, 0.0)
    alj_months = reconsideration_months + max(timing.alj_hearing_months, 0.0)
    initial = _clamp(stage.initial)
    reconsideration = (1 - initial) * _clamp(stage.reconsideration)
    remaining_after_recon = (1 - initial) * (1 - _clamp(stage.reconsideration))
    alj_fully = remaining_after_recon * _clamp(stage.alj_fully_favorable)
    alj_partial = remaining_after_recon * _clamp(stage.alj_partially_favorable)

    trained_award_total = initial + reconsideration + alj_fully + alj_partial
    if trained_award_total <= 0:
        scale = 0.0
    else:
        scale = overall_probability / trained_award_total

    levels = [
        {
            "level": "initial",
            "label": "Initial application",
            "probability": initial * scale,
            "detail": "Allowed at initial SSDI determination",
            "estimatedMonthsFromDisability": initial_months,
        },
        {
            "level": "reconsideration",
            "label": "Reconsideration",
            "probability": reconsideration * scale,
            "detail": "Allowed after initial denial at reconsideration",
            "estimatedMonthsFromDisability": reconsideration_months,
        },
        {
            "level": "alj_fully_favorable",
            "label": "ALJ fully favorable",
            "probability": alj_fully * scale,
            "detail": "Awarded fully at hearing level",
            "estimatedMonthsFromDisability": alj_months,
        },
        {
            "level": "alj_partially_favorable",
            "label": "ALJ partially favorable",
            "probability": alj_partial * scale,
            "detail": "Awarded partially at hearing level",
            "estimatedMonthsFromDisability": alj_months,
        },
    ]
    award_total = sum(float(level["probability"]) for level in levels)
    levels.append(
        {
            "level": "not_awarded",
            "label": "Not awarded through hearing",
            "probability": max(0.0, 1 - award_total),
            "detail": "No award through ALJ decision path",
            "estimatedMonthsFromDisability": alj_months,
        }
    )

    return [
        {
            **level,
            "probability": round(min(max(float(level["probability"]), 0.0), 1.0), 4),
            "estimatedMonthsFromDisability": round(
                float(level["estimatedMonthsFromDisability"]), 1
            ),
            "estimatedDecisionDate": _estimated_decision_date(
                claimant["dateOfDisability"],
                float(level["estimatedMonthsFromDisability"]),
            ),
        }
        for level in levels
    ]


def _number(value: Any, field_name: str, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc

    if number < minimum or number > maximum:
        raise ValueError(f"{field_name} must be between {minimum:g} and {maximum:g}")

    return number


def _optional_number(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    if value is None or value == "":
        return default
    return _number(value, field_name, minimum=minimum, maximum=maximum)


def _optional_date(value: Any, field_name: str) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc


def _choice(value: Any, field_name: str, choices: dict[str, float]) -> str:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return str(value)


def _logistic(score: float) -> float:
    return 1 / (1 + exp(-score))


def _logit(probability: float) -> float:
    bounded = min(max(probability, 0.001), 0.999)
    return log(bounded / (1 - bounded))


def _probability_band(probability: float) -> str:
    if probability >= 0.67:
        return "Higher"
    if probability >= 0.4:
        return "Moderate"
    return "Lower"


def _impact(weight: float) -> str:
    if weight > 0.05:
        return "higher"
    if weight < -0.05:
        return "lower"
    return "neutral"


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _predicted_award_level(levels: list[dict[str, str | float]]) -> str:
    award_levels = [level for level in levels if level["level"] != "not_awarded"]
    return str(max(award_levels, key=lambda level: float(level["probability"]))["label"])


def _predicted_outcome(levels: list[dict[str, str | float]]) -> str:
    return str(max(levels, key=lambda level: float(level["probability"]))["label"])


def _apply_rule_cap(probability: float, cap: float | None) -> float:
    if cap is None:
        return probability
    return min(probability, cap)


def _estimated_decision_date(disability_date: date | None, months: float) -> str:
    if disability_date is None:
        return ""
    days = round(months * 30.4375)
    return date.fromordinal(disability_date.toordinal() + days).isoformat()
