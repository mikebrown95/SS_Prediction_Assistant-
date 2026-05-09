from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DATA_URL = "https://www.ssa.gov/disability/data/SSA-SA-FYWL.csv"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "app" / "artifacts"
CSV_PATH = DATA_DIR / "SSA-SA-FYWL.csv"
MONTHLY_PATH = DATA_DIR / "SSA-SA-MOWL.csv"
ALJ_PATH = DATA_DIR / "ALJ Dispostion Data.csv"
HEARING_PROCESSING_PATH = DATA_DIR / "AVG Processing Time by Office.csv"
MODEL_PATH = ARTIFACT_DIR / "ssa_trained_model.json"
SSA_REGION_TO_ALJ_REGION = {
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

FIELD_NAMES = [
    "file_name",
    "file_version",
    "update_date",
    "region_code",
    "state_code",
    "date_type",
    "fiscal_year",
    "adult_population",
    "adult_beneficiaries",
    "adult_beneficiary_pct",
    "eligible_adult_population",
    "adult_receipts",
    "eligible_adult_filing_rate",
    "favorable_adult_determinations",
    "eligible_adult_allowance_rate",
    "all_adult_determinations",
    "adult_favorable_determination_rate",
    "child_population",
    "child_beneficiaries",
    "child_beneficiary_pct",
    "eligible_child_population",
    "child_receipts",
    "eligible_child_filing_rate",
    "favorable_child_determinations",
    "eligible_child_allowance_rate",
    "all_child_determinations",
    "child_favorable_determination_rate",
    "all_determinations",
    "all_favorable_determinations",
    "all_favorable_determination_rate",
]


@dataclass(frozen=True)
class WorkloadRow:
    region_code: str
    state_code: str
    fiscal_year: int
    adult_receipts: int
    favorable_adult_determinations: int
    all_adult_determinations: int
    adult_favorable_determination_rate: float
    eligible_adult_filing_rate: float
    adult_beneficiary_pct: float

    @property
    def adult_favorable_probability(self) -> float:
        return self.adult_favorable_determination_rate / 100


@dataclass(frozen=True)
class MonthlyRow:
    region_code: str
    state_code: str
    formatted_date: str
    initial_ssdi_pending: int
    initial_ssdi_determinations: int
    initial_ssdi_allowances: int
    recon_ssdi_pending: int
    recon_ssdi_determinations: int
    recon_ssdi_allowances: int


@dataclass(frozen=True)
class AljRow:
    region: str
    fully_favorable: int
    partially_favorable: int
    total_denials: int
    total_decisions: int


@dataclass(frozen=True)
class HearingProcessingRow:
    region: str
    office: str
    average_processing_days: int
    dispositions: int


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        print(f"Downloading {DATA_URL}")
        download_csv(DATA_URL, CSV_PATH)

    rows = load_rows(CSV_PATH)
    monthly_rows = load_monthly_rows(MONTHLY_PATH) if MONTHLY_PATH.exists() else []
    alj_rows = load_alj_rows(ALJ_PATH) if ALJ_PATH.exists() else []
    hearing_rows = (
        load_hearing_processing_rows(HEARING_PROCESSING_PATH)
        if HEARING_PROCESSING_PATH.exists()
        else []
    )
    artifact = train_model(rows, monthly_rows, alj_rows, hearing_rows)
    MODEL_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Rows: {artifact['training']['row_count']}")
    print(f"Monthly rows: {artifact['training']['monthly_row_count']}")
    print(f"ALJ rows: {artifact['training']['alj_row_count']}")
    print(f"Hearing processing rows: {artifact['training']['hearing_processing_row_count']}")
    print(f"Fiscal years: {artifact['training']['min_year']}-{artifact['training']['max_year']}")
    print(f"Weighted adult favorable rate: {artifact['baseline_probability']:.4f}")
    print(f"Initial SSDI baseline: {artifact['level_baselines']['initial']:.4f}")
    print(f"Recon SSDI baseline: {artifact['level_baselines']['reconsideration']:.4f}")
    print(f"ALJ award baseline: {artifact['level_baselines']['alj_award']:.4f}")
    print(f"Holdout MAE: {artifact['metrics']['holdout_mae']:.4f}")
    print(f"Wrote {MODEL_PATH}")


def load_rows(path: Path) -> list[WorkloadRow]:
    rows: list[WorkloadRow] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for line_number, values in enumerate(reader, start=1):
            if not values:
                continue
            if line_number == 1 and values[0].strip().lower() == "file name":
                continue
            if len(values) != len(FIELD_NAMES):
                raise ValueError(
                    f"{path} line {line_number} has {len(values)} fields; expected {len(FIELD_NAMES)}"
                )
            record = {key: value.strip() for key, value in zip(FIELD_NAMES, values, strict=True)}
            determinations = _int(record["all_adult_determinations"])
            favorable_rate = _float(record["adult_favorable_determination_rate"])
            if determinations <= 0:
                continue
            rows.append(
                WorkloadRow(
                    region_code=record["region_code"],
                    state_code=record["state_code"],
                    fiscal_year=_int(record["fiscal_year"]),
                    adult_receipts=_int(record["adult_receipts"]),
                    favorable_adult_determinations=_int(record["favorable_adult_determinations"]),
                    all_adult_determinations=determinations,
                    adult_favorable_determination_rate=favorable_rate,
                    eligible_adult_filing_rate=_float(record["eligible_adult_filing_rate"]),
                    adult_beneficiary_pct=_float(record["adult_beneficiary_pct"]),
                )
            )
    if not rows:
        raise ValueError("No usable adult determination rows were found.")
    return rows


def download_csv(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "SSDI-Predictor/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"SSA download failed with HTTP {exc.code}. Download the CSV manually from "
            f"{url} into {path}, then rerun this script."
        ) from exc
    if body.lstrip().lower().startswith(b"<html"):
        raise RuntimeError(
            "SSA returned HTML instead of CSV. Download the CSV manually from "
            f"{url} into {path}, then rerun this script."
        )
    path.write_bytes(body)


def load_monthly_rows(path: Path) -> list[MonthlyRow]:
    rows: list[MonthlyRow] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for record in csv.DictReader(file):
            state = (record.get("State Code") or "").strip()
            region = (record.get("Region Code") or "").strip()
            initial_determinations = _int(record.get("Determinations (Initial SSDI Only)", "0"))
            recon_determinations = _int(record.get("Determinations (Recon SSDI Only)", "0"))
            if not state or initial_determinations <= 0:
                continue
            rows.append(
                MonthlyRow(
                    region_code=region,
                    state_code=state,
                    formatted_date=(record.get("Formatted Date") or "").strip(),
                    initial_ssdi_pending=_int(record.get("Closing Pending (Initial SSDI Only)", "0")),
                    initial_ssdi_determinations=initial_determinations,
                    initial_ssdi_allowances=_int(record.get("Allowances (Initial SSDI Only)", "0")),
                    recon_ssdi_pending=_int(record.get("Closing Pending (Recon SSDI Only)", "0")),
                    recon_ssdi_determinations=recon_determinations,
                    recon_ssdi_allowances=_int(record.get("Allowances (Recon SSDI Only)", "0")),
                )
            )
    return rows


def load_alj_rows(path: Path) -> list[AljRow]:
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("<data")
    if start > 0:
        text = text[start:]
    root = ET.fromstring(text)
    rows: list[AljRow] = []
    for row in root.findall("row"):
        decisions = _int(_xml_text(row, "TOTAL_DECISIONS"))
        if decisions <= 0:
            continue
        rows.append(
            AljRow(
                region=_xml_text(row, "REGION").zfill(2),
                fully_favorable=_int(_xml_text(row, "FULLY_FAVORABLE")),
                partially_favorable=_int(_xml_text(row, "PARTIALLY_FAVORABLE")),
                total_denials=_int(_xml_text(row, "TOTAL_DENIALS")),
                total_decisions=decisions,
            )
        )
    return rows


def load_hearing_processing_rows(path: Path) -> list[HearingProcessingRow]:
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("<data")
    if start > 0:
        text = text[start:]
    root = ET.fromstring(text)
    rows: list[HearingProcessingRow] = []
    for row in root.findall("row"):
        processing_days = _int(_xml_text(row, "AVERAGE_PROCESSING_TIME"))
        dispositions = _int(_xml_text(row, "DISPOSITIONS"))
        if processing_days <= 0 or dispositions <= 0:
            continue
        rows.append(
            HearingProcessingRow(
                region=_xml_text(row, "REGION").zfill(2),
                office=_xml_text(row, "OFFICE"),
                average_processing_days=processing_days,
                dispositions=dispositions,
            )
        )
    return rows


def train_model(
    rows: list[WorkloadRow],
    monthly_rows: list[MonthlyRow] | None = None,
    alj_rows: list[AljRow] | None = None,
    hearing_rows: list[HearingProcessingRow] | None = None,
) -> dict[str, object]:
    monthly_rows = monthly_rows or []
    alj_rows = alj_rows or []
    hearing_rows = hearing_rows or []
    max_year = max(row.fiscal_year for row in rows)
    train_rows = [row for row in rows if row.fiscal_year < max_year]
    holdout_rows = [row for row in rows if row.fiscal_year == max_year]

    baseline = _weighted_rate(train_rows)
    state_rates = _group_rates(train_rows, "state_code", baseline)
    region_rates = _group_rates(train_rows, "region_code", baseline)
    recent_state_rates = _group_rates(
        [row for row in train_rows if row.fiscal_year >= max_year - 4],
        "state_code",
        baseline,
    )
    national_by_year = {
        str(year): _weighted_rate([row for row in train_rows if row.fiscal_year == year])
        for year in sorted({row.fiscal_year for row in train_rows})
    }
    level_model = _train_level_model(monthly_rows, alj_rows, hearing_rows)

    predictions = [_predict_aggregate(row, baseline, state_rates, region_rates) for row in holdout_rows]
    actuals = [row.adult_favorable_probability for row in holdout_rows]
    mae = mean(abs(predicted - actual) for predicted, actual in zip(predictions, actuals, strict=True))

    return {
        "model_type": "aggregate_state_region_allowance_baseline",
        "source_url": DATA_URL,
        "trained_at": datetime.now(UTC).isoformat(),
        "target": "adult_favorable_determination_rate",
        "baseline_probability": round(baseline, 6),
        "state_probabilities": {key: round(value, 6) for key, value in sorted(state_rates.items())},
        "recent_state_probabilities": {
            key: round(value, 6) for key, value in sorted(recent_state_rates.items())
        },
        "region_probabilities": {key: round(value, 6) for key, value in sorted(region_rates.items())},
        "national_probability_by_year": {
            key: round(value, 6) for key, value in national_by_year.items()
        },
        **level_model,
        "metrics": {
            "holdout_year": max_year,
            "holdout_rows": len(holdout_rows),
            "holdout_mae": round(mae, 6),
        },
        "training": {
            "row_count": len(rows),
            "monthly_row_count": len(monthly_rows),
            "alj_row_count": len(alj_rows),
            "hearing_processing_row_count": len(hearing_rows),
            "train_row_count": len(train_rows),
            "min_year": min(row.fiscal_year for row in rows),
            "max_year": max_year,
            "state_count": len({row.state_code for row in rows}),
            "region_count": len({row.region_code for row in rows}),
            "note": (
                "The public SSA files are aggregate workload and disposition data. "
                "They support state/stage baselines, not person-level demographic, education, or condition training."
            ),
        },
    }


def _train_level_model(
    monthly_rows: list[MonthlyRow],
    alj_rows: list[AljRow],
    hearing_rows: list[HearingProcessingRow],
) -> dict[str, object]:
    recent_monthly = _recent_monthly_rows(monthly_rows, months=24)
    initial_baseline = _weighted_monthly_rate(recent_monthly, "initial")
    recon_baseline = _weighted_monthly_rate(recent_monthly, "reconsideration")
    alj_baseline = _weighted_alj_award_rate(alj_rows)
    alj_full_baseline = _weighted_alj_rate(alj_rows, "fully")
    alj_partial_baseline = _weighted_alj_rate(alj_rows, "partially")
    alj_denial_baseline = _weighted_alj_rate(alj_rows, "denial")

    initial_by_state = _monthly_group_rates(recent_monthly, "state_code", "initial")
    recon_by_state = _monthly_group_rates(recent_monthly, "state_code", "reconsideration")
    initial_by_region = _monthly_group_rates(recent_monthly, "region_code", "initial")
    recon_by_region = _monthly_group_rates(recent_monthly, "region_code", "reconsideration")
    initial_timing_by_state = _monthly_group_timing(recent_monthly, "state_code", "initial")
    recon_timing_by_state = _monthly_group_timing(recent_monthly, "state_code", "reconsideration")
    initial_timing_by_region = _monthly_group_timing(recent_monthly, "region_code", "initial")
    recon_timing_by_region = _monthly_group_timing(recent_monthly, "region_code", "reconsideration")
    alj_timing_baseline = _weighted_hearing_months(hearing_rows)
    alj_timing_by_region = _hearing_group_timing(hearing_rows)
    alj_by_region = _alj_group_rates(alj_rows)

    return {
        "level_baselines": {
            "initial": round(initial_baseline, 6),
            "reconsideration": round(recon_baseline, 6),
            "alj_award": round(alj_baseline, 6),
            "alj_fully_favorable": round(alj_full_baseline, 6),
            "alj_partially_favorable": round(alj_partial_baseline, 6),
            "alj_denial": round(alj_denial_baseline, 6),
        },
        "level_state_probabilities": {
            "initial": {key: round(value, 6) for key, value in sorted(initial_by_state.items())},
            "reconsideration": {key: round(value, 6) for key, value in sorted(recon_by_state.items())},
        },
        "level_region_probabilities": {
            "initial": {key: round(value, 6) for key, value in sorted(initial_by_region.items())},
            "reconsideration": {key: round(value, 6) for key, value in sorted(recon_by_region.items())},
            "alj": {
                key: {inner_key: round(inner_value, 6) for inner_key, inner_value in value.items()}
                for key, value in sorted(alj_by_region.items())
            },
        },
        "decision_time_months": {
            "baselines": {
                "initial": round(_weighted_monthly_timing(recent_monthly, "initial"), 1),
                "reconsideration": round(
                    _weighted_monthly_timing(recent_monthly, "reconsideration"), 1
                ),
                "alj_hearing": round(alj_timing_baseline, 1),
            },
            "state": {
                "initial": {key: round(value, 1) for key, value in sorted(initial_timing_by_state.items())},
                "reconsideration": {
                    key: round(value, 1) for key, value in sorted(recon_timing_by_state.items())
                },
            },
            "region": {
                "initial": {
                    key: round(value, 1) for key, value in sorted(initial_timing_by_region.items())
                },
                "reconsideration": {
                    key: round(value, 1) for key, value in sorted(recon_timing_by_region.items())
                },
                "alj": {key: round(value, 1) for key, value in sorted(alj_timing_by_region.items())},
            },
            "note": (
                "Initial and reconsideration timing use recent monthly closing pending divided by "
                "monthly determinations. ALJ timing uses hearing-office average processing time "
                "weighted by dispositions."
            ),
        },
    }


def _predict_aggregate(
    row: WorkloadRow,
    baseline: float,
    state_rates: dict[str, float],
    region_rates: dict[str, float],
) -> float:
    state = state_rates.get(row.state_code, baseline)
    region = region_rates.get(row.region_code, baseline)
    return (state * 0.75) + (region * 0.15) + (baseline * 0.10)


def _group_rates(rows: list[WorkloadRow], attr: str, fallback: float) -> dict[str, float]:
    grouped: dict[str, list[WorkloadRow]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, attr), []).append(row)
    return {key: _weighted_rate(value) if value else fallback for key, value in grouped.items()}


def _recent_monthly_rows(rows: list[MonthlyRow], months: int) -> list[MonthlyRow]:
    if not rows:
        return []
    recent_dates = sorted({row.formatted_date for row in rows if row.formatted_date})
    selected_dates = set(recent_dates[-months:])
    return [row for row in rows if row.formatted_date in selected_dates]


def _weighted_monthly_rate(rows: list[MonthlyRow], stage: str) -> float:
    if stage == "initial":
        determinations = sum(row.initial_ssdi_determinations for row in rows)
        allowances = sum(row.initial_ssdi_allowances for row in rows)
    else:
        determinations = sum(row.recon_ssdi_determinations for row in rows)
        allowances = sum(row.recon_ssdi_allowances for row in rows)
    return allowances / determinations if determinations else 0.0


def _monthly_group_rates(rows: list[MonthlyRow], attr: str, stage: str) -> dict[str, float]:
    grouped: dict[str, list[MonthlyRow]] = {}
    for row in rows:
        key = getattr(row, attr)
        if key:
            grouped.setdefault(key, []).append(row)
    return {key: _weighted_monthly_rate(value, stage) for key, value in grouped.items()}


def _weighted_monthly_timing(rows: list[MonthlyRow], stage: str) -> float:
    if stage == "initial":
        determinations = sum(row.initial_ssdi_determinations for row in rows)
        pending = sum(row.initial_ssdi_pending for row in rows)
    else:
        determinations = sum(row.recon_ssdi_determinations for row in rows)
        pending = sum(row.recon_ssdi_pending for row in rows)
    return pending / determinations if determinations else 0.0


def _monthly_group_timing(rows: list[MonthlyRow], attr: str, stage: str) -> dict[str, float]:
    grouped: dict[str, list[MonthlyRow]] = {}
    for row in rows:
        key = getattr(row, attr)
        if key:
            grouped.setdefault(key, []).append(row)
    return {key: _weighted_monthly_timing(value, stage) for key, value in grouped.items()}


def _weighted_hearing_months(rows: list[HearingProcessingRow]) -> float:
    dispositions = sum(row.dispositions for row in rows)
    weighted_days = sum(row.average_processing_days * row.dispositions for row in rows)
    return (weighted_days / dispositions / 30.4375) if dispositions else 12.0


def _hearing_group_timing(rows: list[HearingProcessingRow]) -> dict[str, float]:
    grouped: dict[str, list[HearingProcessingRow]] = {}
    for row in rows:
        grouped.setdefault(row.region, []).append(row)
    return {key: _weighted_hearing_months(value) for key, value in grouped.items()}


def _weighted_alj_award_rate(rows: list[AljRow]) -> float:
    decisions = sum(row.total_decisions for row in rows)
    awards = sum(row.fully_favorable + row.partially_favorable for row in rows)
    return awards / decisions if decisions else 0.0


def _weighted_alj_rate(rows: list[AljRow], outcome: str) -> float:
    decisions = sum(row.total_decisions for row in rows)
    if not decisions:
        return 0.0
    if outcome == "fully":
        count = sum(row.fully_favorable for row in rows)
    elif outcome == "partially":
        count = sum(row.partially_favorable for row in rows)
    else:
        count = sum(row.total_denials for row in rows)
    return count / decisions


def _alj_group_rates(rows: list[AljRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[AljRow]] = {}
    for row in rows:
        grouped.setdefault(row.region, []).append(row)
    return {
        key: {
            "fully_favorable": _weighted_alj_rate(value, "fully"),
            "partially_favorable": _weighted_alj_rate(value, "partially"),
            "denial": _weighted_alj_rate(value, "denial"),
            "award": _weighted_alj_award_rate(value),
        }
        for key, value in grouped.items()
    }


def _weighted_rate(rows: list[WorkloadRow]) -> float:
    determinations = sum(row.all_adult_determinations for row in rows)
    favorable = sum(row.favorable_adult_determinations for row in rows)
    if determinations == 0:
        raise ValueError("Cannot calculate rate with zero determinations.")
    return favorable / determinations


def _int(value: str) -> int:
    return int((value or "0").replace(",", "").strip() or "0")


def _float(value: str) -> float:
    return float((value or "0").replace(",", "").strip() or "0")


def _xml_text(row: ET.Element, name: str) -> str:
    child = row.find(name)
    return child.text.strip() if child is not None and child.text else ""


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1) from exc
