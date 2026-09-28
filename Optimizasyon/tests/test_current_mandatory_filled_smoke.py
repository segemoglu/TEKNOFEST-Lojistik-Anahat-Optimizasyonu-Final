"""Current-data smoke comparison for same-O-D and mandatory-filled strategies."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import unittest

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import (
    GreedyMandatoryRentalFilledBuilder,
    GreedySameODConsolidatedBuilder,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Return one protected workbook/PDF hash without changing the file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(forecast_path: Path) -> dict[Path, str]:
    """Snapshot every raw file and the development forecast workbook."""

    raw_files = sorted(
        path for path in (REPOSITORY_ROOT / "raw" / "current").iterdir()
        if path.is_file()
    )
    return {path: _sha256(path) for path in (*raw_files, forecast_path)}


class CurrentMandatoryFilledSmokeTests(unittest.TestCase):
    """Require 6B to stay valid and strictly improve on 6A current data."""

    def test_mandatory_filling_is_valid_cheaper_and_source_preserving(self) -> None:
        """6B carries mandatory load, uses fewer spots, costs less and keeps hashes."""

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
        policy = EvaluationPolicy.from_manifest(loader.manifest)
        same_od = GreedySameODConsolidatedBuilder(problem, policy).build()
        filled = GreedyMandatoryRentalFilledBuilder(problem, policy).build()
        after = _protected_hashes(forecast_path)

        loaded_mandatory = tuple(
            trip for trip in filled.plan.trips
            if trip.ownership_type == "mandatory_rented"
            and trip.legs[0].assignments
        )
        mandatory_desi = math.fsum(
            assignment.desi
            for trip in loaded_mandatory
            for assignment in trip.legs[0].assignments
        )
        final_counts = Counter(
            assignment.demand_part_id
            for trip in filled.plan.trips
            for leg in trip.legs
            for assignment in leg.assignments
            if assignment.is_final_delivery
        )
        error_codes = {issue.code for issue in filled.validation_report.errors}

        self.assertEqual(filled.plan.demand_parts, same_od.plan.demand_parts)
        self.assertTrue(filled.validation_report.is_valid)
        self.assertTrue(loaded_mandatory)
        self.assertGreater(mandatory_desi, 0.0)
        self.assertLess(filled.spot_vehicle_count, same_od.spot_vehicle_count)
        self.assertLess(filled.total_cost_tl, same_od.total_cost_tl)
        self.assertEqual(set(final_counts), {
            part.part_id for part in filled.plan.demand_parts
        })
        self.assertTrue(all(count == 1 for count in final_counts.values()))
        self.assertFalse(
            {
                "demand_underdelivered",
                "demand_overdelivered",
                "demand_part_underdelivered",
                "demand_part_overdelivered",
            }
            & error_codes
        )
        self.assertLessEqual(
            filled.validation_report.metrics["peak_truck_quota_ratio"],
            1.0,
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
