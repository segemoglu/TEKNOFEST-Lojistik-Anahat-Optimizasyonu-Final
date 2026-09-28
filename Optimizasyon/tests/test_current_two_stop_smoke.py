"""Current-data smoke comparison for 6B and same-origin two-stop 6D."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import (
    ExcelProblemLoader,
    configured_project_path,
)
from src.io.transport_plan_workbook import (
    TransportPlanWorkbookExporter,
    read_transport_plan_workbook,
)
from src.planning import (
    GreedyMandatoryRentalFilledBuilder,
    GreedySameOriginTwoStopBuilder,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Return one protected source/workbook hash without changing the file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(forecast_path: Path) -> dict[Path, str]:
    """Snapshot every immutable raw file and development forecast workbook."""

    raw_files = sorted(
        path for path in (REPOSITORY_ROOT / "raw" / "current").iterdir()
        if path.is_file()
    )
    return {path: _sha256(path) for path in (*raw_files, forecast_path)}


class CurrentTwoStopSmokeTests(unittest.TestCase):
    """Require 6D to preserve cargo/rentals and strictly improve current 6B."""

    def test_two_stop_plan_is_valid_cheaper_and_source_preserving(self) -> None:
        """Real forecast creates valid two-stop savings with unchanged inputs."""

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
        base = GreedyMandatoryRentalFilledBuilder(problem, policy).build()
        result = GreedySameOriginTwoStopBuilder(problem, policy).build()
        after = _protected_hashes(forecast_path)

        two_stop = tuple(
            trip for trip in result.plan.trips
            if trip.ownership_type == "spot" and len(trip.legs) == 2
        )
        final_counts = Counter(
            assignment.demand_part_id
            for trip in result.plan.trips
            for leg in trip.legs
            for assignment in leg.assignments
            if assignment.is_final_delivery
        )
        base_mandatory = tuple(
            trip for trip in base.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        result_mandatory = tuple(
            trip for trip in result.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            export = TransportPlanWorkbookExporter(
                problem,
                policy,
                loader.manifest,
            ).write(result.plan, Path(temp_dir) / "TAŞIMA PLANI.xlsx")
            reloaded_rows = read_transport_plan_workbook(
                export.path,
                loader.manifest,
            )

        self.assertTrue(base.validation_report.is_valid)
        self.assertTrue(result.validation_report.is_valid)
        self.assertEqual(result.plan.demand_parts, base.plan.demand_parts)
        self.assertTrue(two_stop)
        self.assertLess(result.spot_vehicle_count, base.spot_vehicle_count)
        self.assertLess(result.total_cost_tl, base.total_cost_tl)
        self.assertEqual(set(final_counts), {
            part.part_id for part in result.plan.demand_parts
        })
        self.assertTrue(all(count == 1 for count in final_counts.values()))
        self.assertEqual(result_mandatory, base_mandatory)
        # One row per demand part per leg; an empty mandatory leg still writes one.
        self.assertEqual(
            export.audit.row_count,
            sum(
                max(1, len(leg.assignments))
                for trip in result.plan.trips
                for leg in trip.legs
            ),
        )
        self.assertEqual(reloaded_rows, export.rows)
        self.assertAlmostEqual(export.audit.total_cost_tl, result.total_cost_tl, places=2)
        self.assertLessEqual(
            result.validation_report.metrics["peak_truck_quota_ratio"],
            1.0,
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
