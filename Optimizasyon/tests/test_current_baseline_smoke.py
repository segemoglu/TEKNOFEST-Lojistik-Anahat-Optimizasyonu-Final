"""Current-data smoke test for the generated development forecast baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import ConservativeDirectBaselineBuilder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 without changing its metadata or contents."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(forecast_path: Path) -> dict[Path, str]:
    """Snapshot every raw file plus the development forecast workbook."""

    raw_files = sorted(
        path for path in (REPOSITORY_ROOT / "raw" / "current").iterdir()
        if path.is_file()
    )
    return {path: _sha256(path) for path in (*raw_files, forecast_path)}


class CurrentBaselineSmokeTests(unittest.TestCase):
    """Verify the full current forecast is delivered without source mutation."""

    def test_current_forecast_builds_valid_non_truck_baseline(self) -> None:
        """All 4,046 records pass validator and preserve protected hashes."""

        loader = ExcelProblemLoader(project_root=REPOSITORY_ROOT)
        static = loader.load_static_problem_data()
        forecast_path = configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=REPOSITORY_ROOT,
        )
        before = _protected_hashes(forecast_path)
        demands = loader.load_demands_from_workbook(
            forecast_path,
            static_data=static,
        )
        problem = ProblemData(static, demands)
        result = ConservativeDirectBaselineBuilder(
            problem,
            EvaluationPolicy.from_manifest(loader.manifest),
        ).build()
        after = _protected_hashes(forecast_path)

        self.assertEqual(len(demands), 4_046)
        self.assertTrue(result.validation_report.is_valid)
        self.assertFalse(result.validation_report.errors)
        self.assertEqual(result.mandatory_vehicle_count, 112)
        self.assertEqual(
            result.validation_report.metrics["mandatory_missing_count"], 0
        )
        self.assertEqual(
            result.validation_report.metrics["mandatory_excess_count"], 0
        )
        self.assertLessEqual(
            result.validation_report.metrics["peak_truck_quota_ratio"], 1.0
        )
        self.assertEqual(
            result.validation_report.metrics["delivered_desi"],
            sum(demand.desi for demand in demands),
        )
        spot_trips = [
            trip for trip in result.plan.trips if trip.ownership_type == "spot"
        ]
        self.assertTrue(all(trip.vehicle_type != "Tır" for trip in spot_trips))
        self.assertTrue(
            all(
                operation.start_at.date() == operation.end_at.date()
                for trip in result.plan.trips
                for operation in trip.handling_operations
            )
        )
        error_codes = {issue.code for issue in result.validation_report.errors}
        self.assertFalse(
            {"demand_underdelivered", "demand_overdelivered"} & error_codes
        )
        self.assertGreaterEqual(result.build_seconds, 0.0)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
