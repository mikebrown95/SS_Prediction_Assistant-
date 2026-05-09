import unittest

from scripts.train_ssa_model import FIELD_NAMES, load_rows, train_model


def make_row(state, region, year, favorable, determinations, rate):
    values = {
        "file_name": "SSA-SA-FYWL.csv",
        "file_version": "1",
        "update_date": "01/01/2026",
        "region_code": region,
        "state_code": state,
        "date_type": "FY",
        "fiscal_year": str(year),
        "adult_population": "1000000",
        "adult_beneficiaries": "70000",
        "adult_beneficiary_pct": "7",
        "eligible_adult_population": "930000",
        "adult_receipts": "10000",
        "eligible_adult_filing_rate": "1.08",
        "favorable_adult_determinations": str(favorable),
        "eligible_adult_allowance_rate": "0.35",
        "all_adult_determinations": str(determinations),
        "adult_favorable_determination_rate": str(rate),
        "child_population": "200000",
        "child_beneficiaries": "10000",
        "child_beneficiary_pct": "5",
        "eligible_child_population": "190000",
        "child_receipts": "1000",
        "eligible_child_filing_rate": "0.52",
        "favorable_child_determinations": "300",
        "eligible_child_allowance_rate": "0.16",
        "all_child_determinations": "1000",
        "child_favorable_determination_rate": "30",
        "all_determinations": str(determinations + 1000),
        "all_favorable_determinations": str(favorable + 300),
        "all_favorable_determination_rate": "33",
    }
    return ",".join(values[name] for name in FIELD_NAMES)


class TrainingTests(unittest.TestCase):
    def test_loads_no_header_ssa_csv_and_trains_artifact(self):
        path = self._write_fixture(
            "\n".join(
                [
                    make_row("AL", "ATL", 2021, 3000, 10000, 30),
                    make_row("TX", "DAL", 2021, 4000, 10000, 40),
                    make_row("AL", "ATL", 2022, 3500, 10000, 35),
                    make_row("TX", "DAL", 2022, 4500, 10000, 45),
                ]
            )
        )

        rows = load_rows(path)
        artifact = train_model(rows)

        self.assertEqual(artifact["training"]["row_count"], 4)
        self.assertEqual(artifact["metrics"]["holdout_year"], 2022)
        self.assertIn("AL", artifact["state_probabilities"])
        self.assertIn("ATL", artifact["region_probabilities"])

    def _write_fixture(self, content):
        import tempfile
        from pathlib import Path

        handle = tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8")
        with handle:
            handle.write(content)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
