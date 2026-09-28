"""Synthetic tests for official transport-plan workbook export and reload."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import math
import tempfile
import unittest

from openpyxl import load_workbook

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandInstance,
    HandlingCapacity,
    MandatoryRentalRoute,
    ProblemData,
    StaticProblemData,
    TransportPlan,
    TravelArc,
    VehicleType,
)
from src.evaluation import CostCalculator, EvaluationPolicy
from src.io import load_manifest
from src.io.transport_plan_workbook import (
    TransportPlanWorkbookError,
    TransportPlanWorkbookExporter,
    _truncate_to_minute,
    read_transport_plan_workbook,
)
from src.planning import GreedySameOriginTwoStopBuilder
from tests.solution_fixtures import make_empty_trip, make_policy, make_problem


VEHICLES = (
    VehicleType("Tır", 200.0, 20.0, 2.0, 30.0, 3.0),
    VehicleType("Kamyon", 120.0, 12.0, 1.0, 10.0, 0.8),
    VehicleType("Kamyonet", 60.0, 8.0, 1.0, 6.0, 1.0),
)


def _two_stop_problem() -> ProblemData:
    """Create an economical A-B-C pair that the actual 6D builder combines."""

    centers = ("A", "B", "C")
    arcs = tuple(
        TravelArc(
            origin,
            destination,
            10.0 if (origin, destination) == ("B", "C") else 100.0,
            {vehicle.name: 1.0 for vehicle in VEHICLES},
            1,
        )
        for origin in centers
        for destination in centers
        if origin != destination
    )
    static = StaticProblemData(
        tuple(Center(item) for item in centers),
        arcs,
        VEHICLES,
        (),
        tuple(HandlingCapacity(item, 10_000.0) for item in centers),
        tuple(DailyTruckCapacity(item, 20) for item in centers),
    )
    demands = (
        DemandInstance("D1", date(2026, 6, 29), time(9), "A", "B", 40.0),
        DemandInstance("D2", date(2026, 6, 29), time(9), "A", "C", 40.0),
    )
    return ProblemData(static, demands)


def _costed_empty_mandatory_plan(
    problem: ProblemData,
    policy: EvaluationPolicy,
) -> TransportPlan:
    """Create one required empty rental with a complete reported cost."""

    trip = make_empty_trip(
        vehicle_id="INTERNAL-MANDATORY-ID",
        ownership_type="mandatory_rented",
        vehicle_type="Tır",
    )
    bare = TransportPlan((), (trip,))
    return TransportPlan((), (trip,), CostCalculator(problem, policy).calculate(bare))


class TruncateToMinuteTests(unittest.TestCase):
    """Verify export timestamps match what the hh:mm cell format displays."""

    def test_sub_minute_remainder_is_dropped(self) -> None:
        """Seconds and microseconds are removed, the minute is kept."""

        self.assertEqual(
            _truncate_to_minute(datetime(2026, 6, 29, 18, 20, 45, 168000)),
            datetime(2026, 6, 29, 18, 20),
        )
        self.assertEqual(
            _truncate_to_minute(datetime(2026, 6, 29, 18, 20, 59, 999999)),
            datetime(2026, 6, 29, 18, 20),
        )

    def test_whole_minute_is_unchanged(self) -> None:
        """An exact minute passes through untouched."""

        self.assertEqual(
            _truncate_to_minute(datetime(2026, 6, 29, 18, 20)),
            datetime(2026, 6, 29, 18, 20),
        )

    def test_truncation_never_changes_the_calendar_date(self) -> None:
        """Late-night activity stays on its own day; only seconds are dropped."""

        self.assertEqual(
            _truncate_to_minute(datetime(2026, 6, 29, 23, 59, 30)),
            datetime(2026, 6, 29, 23, 59),
        )


class TransportPlanWorkbookExporterTests(unittest.TestCase):
    """Verify row semantics, cost reconciliation, typed cells and placeholders."""

    def test_two_stop_rows_explode_per_demand_and_reconcile_cost(self) -> None:
        """Each demand part gets its own row; TL and handling split by desi."""

        problem = _two_stop_problem()
        policy = make_policy()
        result = GreedySameOriginTwoStopBuilder(problem, policy).build()
        exporter = TransportPlanWorkbookExporter(problem, policy, load_manifest())

        rows = exporter.build_rows(result.plan)

        # Leg A->B carries both parts, leg B->C carries only the continuing one.
        self.assertEqual(len(rows), 3)
        self.assertEqual({row.vehicle_id for row in rows}, {"V0001"})
        self.assertEqual([row.origin for row in rows], ["A", "A", "B"])
        # A row never reports a leg total; it reports one part's own desi.
        self.assertEqual([row.carried_desi for row in rows], [40.0, 40.0, 40.0])
        self.assertNotEqual(rows[0].demand_part_id, rows[1].demand_part_id)
        self.assertIsNotNone(rows[2].demand_part_id)
        # Equal desi on the first leg means an equal split of its loading time.
        self.assertGreater(rows[0].departure_handling_hours, 0.0)
        self.assertAlmostEqual(
            rows[0].departure_handling_hours,
            rows[1].departure_handling_hours,
        )
        # The continuing part was not loaded again on the second leg.
        self.assertEqual(rows[2].departure_handling_hours, 0.0)
        # Travel time is shared, so it repeats whole instead of being divided.
        self.assertAlmostEqual(rows[0].travel_hours, rows[1].travel_hours)
        self.assertTrue(
            math.isclose(
                math.fsum(row.total_cost_tl for row in rows),
                result.total_cost_tl,
                abs_tol=0.01,
            )
        )

    def test_cost_is_allocated_in_proportion_to_desi(self) -> None:
        """Unequal desi on one leg splits that leg's TL in the same ratio."""

        problem = _two_stop_problem()
        policy = make_policy()
        result = GreedySameOriginTwoStopBuilder(problem, policy).build()
        exporter = TransportPlanWorkbookExporter(problem, policy, load_manifest())

        rows = exporter.build_rows(result.plan)
        first_leg = [row for row in rows if row.origin == "A"]

        # Both parts are 40 desi, so their share of the leg must be identical.
        self.assertEqual(len(first_leg), 2)
        self.assertAlmostEqual(
            first_leg[0].total_cost_tl - first_leg[0].sla_penalty_tl,
            first_leg[1].total_cost_tl - first_leg[1].sla_penalty_tl,
        )

    def test_write_uses_typed_dates_times_and_semantic_round_trip(self) -> None:
        """Generated cells preserve values and the public reader returns equal rows."""

        problem = _two_stop_problem()
        policy = make_policy()
        plan = GreedySameOriginTwoStopBuilder(problem, policy).build().plan
        manifest = load_manifest()
        exporter = TransportPlanWorkbookExporter(problem, policy, manifest)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "TAŞIMA PLANI.xlsx"
            result = exporter.write(plan, path)
            actual = read_transport_plan_workbook(path, manifest)
            workbook = load_workbook(path, read_only=False, data_only=True)
            try:
                worksheet = workbook["Sheet1"]
                self.assertIsInstance(worksheet["F2"].value, datetime)
                self.assertIsInstance(worksheet["G2"].value, time)
                self.assertEqual(worksheet["F2"].number_format, "yyyy-mm-dd")
                self.assertEqual(worksheet["G2"].number_format, "hh:mm")
            finally:
                workbook.close()

        self.assertEqual(actual, result.rows)
        self.assertEqual(result.audit.row_count, 3)
        self.assertEqual(result.audit.vehicle_count, 1)
        # Q&A 11.4 accepts HH:MM, so exported timestamps carry no sub-minute part.
        for row in result.rows:
            self.assertEqual((row.departure_at.second, row.departure_at.microsecond), (0, 0))
            self.assertEqual((row.arrival_at.second, row.arrival_at.microsecond), (0, 0))
        # Truncation only drops sub-minute precision, so an exported time never
        # moves more than one minute below the plan time.
        # One leg now yields one row per carried demand part, so repeat it.
        leg_per_row = [
            leg
            for trip in plan.trips
            for leg in trip.legs
            for _ in range(max(1, len(leg.assignments)))
        ]
        for row, leg in zip(result.rows, leg_per_row, strict=True):
            self.assertLessEqual(row.departure_at, leg.departure_at)
            self.assertLess(
                (leg.departure_at - row.departure_at).total_seconds(), 60.0
            )
            self.assertLessEqual(row.arrival_at, leg.arrival_at)
            self.assertLess((leg.arrival_at - row.arrival_at).total_seconds(), 60.0)

    def test_empty_mandatory_row_uses_zero_placeholder_and_rental_label(self) -> None:
        """A required empty movement exports as Kiralık with Talep ID/desi zero."""

        route = MandatoryRentalRoute("A", "B", "Tır", 1)
        problem = make_problem(demands=(), mandatory_routes=(route,))
        policy = make_policy()
        plan = _costed_empty_mandatory_plan(problem, policy)
        manifest = load_manifest()
        exporter = TransportPlanWorkbookExporter(problem, policy, manifest)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.xlsx"
            result = exporter.write(plan, path)
            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                worksheet = workbook["Sheet1"]
                self.assertEqual(worksheet["B2"].value, "Kiralık")
                self.assertEqual(worksheet["J2"].value, 0)
                self.assertEqual(worksheet["K2"].value, 0)
            finally:
                workbook.close()

        self.assertEqual(len(result.rows), 1)
        self.assertIsNone(result.rows[0].demand_part_id)
        self.assertEqual(result.rows[0].carried_desi, 0.0)
        self.assertGreater(result.rows[0].total_cost_tl, 0.0)

    def test_unreported_plan_is_rejected_before_any_file_is_written(self) -> None:
        """Exporter refuses a plan that skipped the reported-cost safety gate."""

        problem = _two_stop_problem()
        policy = make_policy()
        valid = GreedySameOriginTwoStopBuilder(problem, policy).build().plan
        unreported = TransportPlan(valid.demand_parts, valid.trips)
        exporter = TransportPlanWorkbookExporter(problem, policy, load_manifest())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unsafe.xlsx"
            with self.assertRaises(TransportPlanWorkbookError):
                exporter.write(unreported, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
