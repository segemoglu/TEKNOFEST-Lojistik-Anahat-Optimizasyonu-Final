"""Current-data smoke comparison for 6D and bounded multi-stop 6E."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.io.transport_plan_workbook import TransportPlanWorkbookExporter
from src.planning import (
    GreedySameOriginMultiStopBuilder,
    GreedySameOriginTwoStopBuilder,
)
from src.planning.multi_stop import (
    MAX_DELIVERY_STOPS,
    MIN_MULTI_STOP_DELIVERIES,
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
        path
        for path in (REPOSITORY_ROOT / "raw" / "current").iterdir()
        if path.is_file()
    )
    return {path: _sha256(path) for path in (*raw_files, forecast_path)}


class CurrentMultiStopSmokeTests(unittest.TestCase):
    """Require real 6E savings, cargo fidelity, valid export and source hashes."""

    def test_multi_stop_plan_strictly_beats_6d_and_round_trips(self) -> None:
        """Current forecast produces valid 3/4-stop savings and exact workbook."""

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
        base = GreedySameOriginTwoStopBuilder(problem, policy).build()
        result = GreedySameOriginMultiStopBuilder(
            problem,
            policy,
        ).build_from_two_stop(base)
        after = _protected_hashes(forecast_path)

        multi_stop = tuple(
            trip
            for trip in result.plan.trips
            if trip.ownership_type == "spot" and len(trip.legs) >= 3
        )
        final_counts = Counter(
            assignment.demand_part_id
            for trip in result.plan.trips
            for leg in trip.legs
            for assignment in leg.assignments
            if assignment.is_final_delivery
        )
        base_mandatory = tuple(
            trip
            for trip in base.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        result_mandatory = tuple(
            trip
            for trip in result.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            export = TransportPlanWorkbookExporter(
                problem,
                policy,
                loader.manifest,
            ).write(result.plan, Path(temp_dir) / "TAŞIMA PLANI.xlsx")

        self.assertTrue(base.validation_report.is_valid)
        self.assertTrue(result.validation_report.is_valid)
        self.assertTrue(multi_stop)
        # The builder must actually use its room to grow and must never exceed
        # it.  Both bounds follow MAX_DELIVERY_STOPS instead of a literal, so
        # raising the cap stays a one-line change; a hard-coded 4 here is what
        # failed when it moved to 20.
        self.assertTrue(
            any(len(trip.legs) > MIN_MULTI_STOP_DELIVERIES for trip in multi_stop)
        )
        self.assertTrue(
            all(len(trip.legs) <= MAX_DELIVERY_STOPS for trip in multi_stop)
        )
        self.assertLess(result.total_cost_tl, base.total_cost_tl)
        self.assertLess(result.spot_vehicle_count, base.spot_vehicle_count)
        self.assertEqual(result.plan.demand_parts, base.plan.demand_parts)
        self.assertEqual(
            final_counts,
            Counter({part.part_id: 1 for part in result.plan.demand_parts}),
        )
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
        self.assertAlmostEqual(
            export.audit.total_cost_tl,
            result.total_cost_tl,
            places=2,
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
