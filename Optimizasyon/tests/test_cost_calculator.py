"""Synthetic tests for vehicle grouping, ceil and SLA cost formulas."""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from src.domain import DemandAssignment, DemandPart, TransportPlan
from src.evaluation import (
    CostCalculationError,
    CostCalculator,
    ceil_billable_hours,
)

from tests.solution_fixtures import (
    make_delivery_trip,
    make_demand,
    make_empty_trip,
    make_policy,
    make_problem,
)


class CostCalculatorTests(unittest.TestCase):
    """Verify cost calculations without invoking any optimizer or loader."""

    def test_sixteen_exact_hours_are_billed_as_sixteen(self) -> None:
        """An exact 16-hour activity envelope must not be rounded to 17."""

        problem = make_problem(demands=())
        plan = TransportPlan(
            demand_parts=(),
            trips=(make_empty_trip(duration=timedelta(hours=16)),),
        )

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertEqual(breakdown.vehicle_cost_lines[0].charged_hours, 16)
        self.assertEqual(ceil_billable_hours(16.0), 16)
        self.assertEqual(breakdown.total_cost_tl, 620.0)

    def test_sixteen_hours_one_minute_are_billed_as_seventeen(self) -> None:
        """Any positive fraction after 16 hours must ceil to 17 hours."""

        problem = make_problem(demands=())
        plan = TransportPlan(
            demand_parts=(),
            trips=(
                make_empty_trip(duration=timedelta(hours=16, minutes=1)),
            ),
        )

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertEqual(breakdown.vehicle_cost_lines[0].charged_hours, 17)
        self.assertEqual(ceil_billable_hours(16 + 1 / 60), 17)
        self.assertEqual(breakdown.total_cost_tl, 640.0)

    def test_two_demands_on_one_vehicle_create_one_vehicle_cost_line(self) -> None:
        """Demand multiplicity must not multiply physical vehicle cost."""

        demands = (make_demand("D1"), make_demand("D2"))
        problem = make_problem(demands=demands)
        parts = (DemandPart("P1", "D1", 20), DemandPart("P2", "D2", 20))
        assignments = (
            DemandAssignment("P1", 20, is_final_delivery=False),
            DemandAssignment("P2", 20, is_final_delivery=False),
        )
        trip = make_delivery_trip(assignments=assignments)
        plan = TransportPlan(demand_parts=parts, trips=(trip,))

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertEqual(len(breakdown.vehicle_cost_lines), 1)
        self.assertEqual(breakdown.vehicle_cost_lines[0].vehicle_id, "V1")
        self.assertEqual(breakdown.total_cost_tl, 340.0)

    def test_empty_mandatory_vehicle_still_has_normal_cost(self) -> None:
        """A mandatory vehicle with zero assignments still pays time and km."""

        problem = make_problem(demands=())
        trip = make_empty_trip(ownership_type="mandatory_rented")
        plan = TransportPlan(demand_parts=(), trips=(trip,))

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertEqual(breakdown.mandatory_rented_hourly_cost_tl, 10.0)
        self.assertEqual(breakdown.mandatory_rented_distance_cost_tl, 200.0)
        self.assertEqual(breakdown.total_cost_tl, 210.0)

    def test_vehicle_activity_after_july_fifth_is_not_discarded(self) -> None:
        """A 6 July trip must contribute its complete vehicle cost."""

        problem = make_problem(demands=())
        trip = make_empty_trip(start_at=datetime(2026, 7, 6, 9))
        plan = TransportPlan(demand_parts=(), trips=(trip,))

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertEqual(breakdown.total_cost_tl, 320.0)
        self.assertEqual(breakdown.vehicle_cost_lines[0].trip_ids, ("T1",))

    def test_sla_delay_uses_an_independent_ceil_per_final_part(self) -> None:
        """A delayed final delivery must use part desi × ceil(delay) × 0.4."""

        problem = make_problem(demands=(make_demand(),))
        part = DemandPart("P1", "D1", 40)
        assignment = DemandAssignment("P1", 40, is_final_delivery=True)
        trip = make_delivery_trip(
            assignments=(assignment,),
            start_at=datetime(2026, 7, 1, 9),
        )
        plan = TransportPlan(demand_parts=(part,), trips=(trip,))

        breakdown = CostCalculator(problem, make_policy()).calculate(plan)

        sla_line = breakdown.sla_penalty_lines[0]
        self.assertEqual(sla_line.charged_delay_hours, 26)
        self.assertEqual(sla_line.penalty_tl, 416.0)

    def test_hourly_rounding_granularity_is_configurable(self) -> None:
        """The same vehicle may be billed once per vehicle or once per trip."""

        problem = make_problem(demands=())
        trips = (
            make_empty_trip(vehicle_id="V1", trip_id="T1"),
            make_empty_trip(
                vehicle_id="V1",
                trip_id="T2",
                start_at=datetime(2026, 6, 29, 12),
            ),
        )
        plan = TransportPlan(demand_parts=(), trips=trips)

        vehicle_cost = CostCalculator(
            problem,
            make_policy(granularity="vehicle"),
        ).calculate(plan)
        trip_cost = CostCalculator(
            problem,
            make_policy(granularity="trip"),
        ).calculate(plan)

        self.assertEqual(len(vehicle_cost.vehicle_cost_lines), 1)
        self.assertEqual(len(trip_cost.vehicle_cost_lines), 2)
        self.assertEqual(vehicle_cost.total_cost_tl, 680.0)
        self.assertEqual(trip_cost.total_cost_tl, 640.0)

    def test_mandatory_vehicle_id_cannot_be_reused_on_another_day(self) -> None:
        """Daily mandatory vehicles must remain separate physical identities."""

        problem = make_problem(demands=())
        trips = (
            make_empty_trip(
                vehicle_id="MAND-1",
                trip_id="T-29",
                ownership_type="mandatory_rented",
            ),
            make_empty_trip(
                vehicle_id="MAND-1",
                trip_id="T-30",
                start_at=datetime(2026, 6, 30, 9),
                ownership_type="mandatory_rented",
            ),
        )

        with self.assertRaises(CostCalculationError):
            CostCalculator(problem, make_policy()).calculate(
                TransportPlan(demand_parts=(), trips=trips)
            )

    def test_same_physical_vehicle_cannot_run_overlapping_trips(self) -> None:
        """Overlapping trip envelopes must not understate one vehicle's cost."""

        problem = make_problem(demands=())
        trips = (
            make_empty_trip(vehicle_id="V1", trip_id="T1"),
            make_empty_trip(
                vehicle_id="V1",
                trip_id="T2",
                start_at=datetime(2026, 6, 29, 9, 30),
            ),
        )

        with self.assertRaises(CostCalculationError):
            CostCalculator(problem, make_policy()).calculate(
                TransportPlan(demand_parts=(), trips=trips)
            )


if __name__ == "__main__":
    unittest.main()
