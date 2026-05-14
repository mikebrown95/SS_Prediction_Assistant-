from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SGA_NON_BLIND_2026 = 1690
SGA_BLIND_2026 = 2830


@dataclass(frozen=True)
class RuleFinding:
    rule: str
    status: str
    detail: str


@dataclass(frozen=True)
class RuleAssessment:
    findings: list[RuleFinding]
    probability_cap: float | None


def assess_ssdi_rules(claimant: dict[str, Any]) -> RuleAssessment:
    findings = [
        _duration_finding(claimant),
        _sga_finding(claimant),
        _duration_of_work_finding(claimant),
        _recent_work_finding(claimant),
    ]
    caps = [_cap_for_finding(finding) for finding in findings]
    active_caps = [cap for cap in caps if cap is not None]
    return RuleAssessment(findings=findings, probability_cap=min(active_caps) if active_caps else None)


def _duration_finding(claimant: dict[str, Any]) -> RuleFinding:
    months = claimant["impairmentMonths"]
    if months >= 12 or claimant["expectedToResultInDeath"]:
        return RuleFinding(
            rule="Duration requirement",
            status="pass",
            detail="Condition is expected to last at least 12 months or result in death.",
        )
    return RuleFinding(
        rule="Duration requirement",
        status="risk",
        detail="SSA rules generally require a condition expected to last at least 12 months or result in death.",
    )


def _sga_finding(claimant: dict[str, Any]) -> RuleFinding:
    earnings = claimant["monthlyEarnings"]
    limit = SGA_BLIND_2026 if claimant["isBlind"] else SGA_NON_BLIND_2026
    if earnings <= limit:
        return RuleFinding(
            rule="Substantial gainful activity",
            status="pass",
            detail=f"Monthly earnings ${earnings:,.0f} are at or below the 2026 SGA screen of ${limit:,.0f}.",
        )
    return RuleFinding(
        rule="Substantial gainful activity",
        status="risk",
        detail=f"Monthly earnings ${earnings:,.0f} exceed the 2026 SGA screen of ${limit:,.0f}.",
    )


def _duration_of_work_finding(claimant: dict[str, Any]) -> RuleFinding:
    required = _required_duration_years(claimant["age"])
    worked = claimant["yearsWorked"]
    if worked >= required:
        return RuleFinding(
            rule="Duration of work test",
            status="pass",
            detail=f"{worked:.1f} years worked meets the estimated {required:.1f}-year duration test.",
        )
    return RuleFinding(
        rule="Duration of work test",
        status="risk",
        detail=f"{worked:.1f} years worked is below the estimated {required:.1f}-year duration test.",
    )


def _recent_work_finding(claimant: dict[str, Any]) -> RuleFinding:
    required = _required_recent_work_years(claimant["age"])
    worked = claimant["yearsWorked"]
    if worked >= required:
        return RuleFinding(
            rule="Recent work test",
            status="pass",
            detail=f"Work history meets the estimated recent work screen for age {claimant['age']:.0f}.",
        )
    return RuleFinding(
        rule="Recent work test",
        status="risk",
        detail=f"Work history may not meet the estimated recent work screen for age {claimant['age']:.0f}.",
    )


def _required_recent_work_years(age: float) -> float:
    if age <= 24:
        return 1.5
    if age < 31:
        return max(1.5, (age - 21) / 2)
    return 5.0


def _required_duration_years(age: float) -> float:
    if age < 28:
        return 1.5
    if age < 30:
        return 2.0
    if age < 34:
        return 2.0
    if age < 38:
        return 3.0
    if age < 42:
        return 4.0
    if age < 44:
        return 5.0
    if age < 46:
        return 5.5
    if age < 48:
        return 6.0
    if age < 50:
        return 6.5
    if age < 52:
        return 7.0
    if age < 54:
        return 7.5
    if age < 56:
        return 8.0
    if age < 58:
        return 8.5
    if age < 60:
        return 9.0
    return 9.5


def _cap_for_finding(finding: RuleFinding) -> float | None:
    if finding.status == "pass":
        return None
    if finding.rule == "Substantial gainful activity":
        return 0.15
    if finding.rule == "Duration requirement":
        return 0.25
    if finding.rule in {"Duration of work test", "Recent work test"}:
        return 0.30
    return None
