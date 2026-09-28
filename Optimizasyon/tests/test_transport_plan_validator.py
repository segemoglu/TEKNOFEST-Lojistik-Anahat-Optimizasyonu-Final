"""Synthetic tests for the independent structured plan validator."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
import unittest

from src.domain import (
    CostBreakdown,
    DemandAssignment,
    DemandPart,
    MandatoryRentalRoute,
    SlaPenaltyLine,
    TransportPlan,
    VehicleCostLine,
)
from src.evaluation import TransportPlanValidator

from tests.solution_fixtures import (
    make_complete_plan,
    make_delivery_trip,
    make_demand,
    make_empty_trip,
    make_policy,
    make_problem,
)


def issue_codes(report) -> set[str]:
    """Return only error codes so each test remains focused and readable."""

    return {issue.code for issue in report.errors}


def manual_complete_plan_cost() -> CostBreakdown:
    """Return the hand-calculated 40-desi fixture cost, independent of code.

    The fixture uses 0.4 + 60 + 0.4 = 60.8 minutes, billed as 2
    hours.  Spot rates are 20 TL/hour and 3 TL/km over 100 km, so the
    immutable expected total is 40 TL + 300 TL = 340 TL.
    """

    line = VehicleCostLine(
        cost_unit_id="vehicle:V1",
        granularity="vehicle",
        vehicle_id="V1",
        ownership_type="spot",
        trip_ids=("T1",),
        leg_ids=("L-T1",),
        usage_hours=60.8 / 60.0,
        charged_hours=2,
        distance_km=100.0,
        hourly_cost_tl=40.0,
        distance_cost_tl=300.0,
        total_cost_tl=340.0,
    )
    sla_line = SlaPenaltyLine(
        demand_id="D1",
        demand_part_id="P1",
        delivered_desi=40.0,
        due_at=datetime(2026, 6, 30, 9),
        delivered_at=datetime(2026, 6, 29, 10, 0, 48),
        delay_hours=0.0,
        charged_delay_hours=0,
        penalty_tl=0.0,
    )
    return CostBreakdown(
        vehicle_cost_lines=(line,),
        sla_penalty_lines=(sla_line,),
        spot_hourly_cost_tl=40.0,
        spot_distance_cost_tl=300.0,
        total_cost_tl=340.0,
    )


class TransportPlanValidatorTests(unittest.TestCase):
    """Verify plan checks return reports and never reject valid concurrency."""

    def test_capacity_excess_is_reported(self) -> None:
        """Carried desi above the vehicle type capacity must be an error."""

        problem = make_problem(vehicle_capacity=30)
        plan = make_complete_plan()

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("vehicle_capacity_exceeded", issue_codes(report))
        self.assertFalse(report.is_valid)
        issue = next(
            item for item in report.errors if item.code == "vehicle_capacity_exceeded"
        )
        self.assertEqual(issue.demand_id, "D1")
        self.assertEqual(issue.vehicle_id, "V1")
        self.assertIsNotNone(issue.event_date)
        self.assertIn("excess_desi", issue.related_values)

    def test_correctly_split_demand_is_fully_delivered(self) -> None:
        """Two parts whose final deliveries sum to the demand must be valid."""

        problem = make_problem()
        parts = (DemandPart("P1", "D1", 15), DemandPart("P2", "D1", 25))
        assignments = (
            DemandAssignment("P1", 15, is_final_delivery=True),
            DemandAssignment("P2", 25, is_final_delivery=True),
        )
        trip = make_delivery_trip(assignments=assignments)
        plan = TransportPlan(demand_parts=parts, trips=(trip,))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertTrue(report.is_valid, report.errors)
        self.assertTrue(report.metrics["all_demands_fully_delivered"])

    def test_under_delivery_is_reported(self) -> None:
        """Delivering less than the part and original demand must be visible."""

        problem = make_problem()
        plan = make_complete_plan(delivered_desi=30, part_desi=40)

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("demand_part_underdelivered", issue_codes(report))
        self.assertIn("demand_underdelivered", issue_codes(report))

    def test_over_delivery_is_reported(self) -> None:
        """Delivering more than requested must not be treated as completion."""

        problem = make_problem()
        plan = make_complete_plan(delivered_desi=50, part_desi=40)

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("demand_part_overdelivered", issue_codes(report))
        self.assertIn("demand_overdelivered", issue_codes(report))

    def test_daily_truck_quota_excess_is_reported(self) -> None:
        """A Tır departure at a zero-quota centre must exceed the daily quota."""

        problem = make_problem(truck_quota_a=0)
        plan = make_complete_plan()

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("daily_truck_quota_exceeded", issue_codes(report))

    def test_overlapping_handling_of_two_vehicles_is_valid(self) -> None:
        """Parallel handling is legal when their combined daily desi fits."""

        demands = (make_demand("D1", desi=20), make_demand("D2", desi=20))
        problem = make_problem(demands=demands)
        parts = (DemandPart("P1", "D1", 20), DemandPart("P2", "D2", 20))
        trip_one = make_delivery_trip(
            assignments=(DemandAssignment("P1", 20, is_final_delivery=True),),
            vehicle_id="V1",
            trip_id="T1",
        )
        trip_two = make_delivery_trip(
            assignments=(DemandAssignment("P2", 20, is_final_delivery=True),),
            vehicle_id="V2",
            trip_id="T2",
        )
        plan = TransportPlan(demand_parts=parts, trips=(trip_one, trip_two))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertTrue(report.is_valid, report.errors)
        self.assertNotIn("handling_overlap", issue_codes(report))

    def test_missing_mandatory_daily_vehicle_is_reported(self) -> None:
        """Every configured mandatory route must operate at its daily count."""

        route = MandatoryRentalRoute("A", "B", "Tır", 1)
        problem = make_problem(mandatory_routes=(route,))
        plan = make_complete_plan()

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("mandatory_vehicle_missing", issue_codes(report))
        self.assertEqual(report.metrics["mandatory_missing_count"], 1)

    def test_post_july_fifth_mandatory_schedule_continues_until_delivery(self) -> None:
        """A 6 July delivery requires the same mandatory route on 6 July."""

        route = MandatoryRentalRoute("A", "B", "Tır", 1)
        demand = make_demand(
            demand_date=date(2026, 7, 5),
            desi=40,
        )
        problem = make_problem(demands=(demand,), mandatory_routes=(route,))
        delivery_plan = make_complete_plan(
            start_at=datetime(2026, 7, 6, 9),
            vehicle_id="V-SPOT",
            trip_id="T-SPOT",
        )
        mandatory_july_fifth = make_empty_trip(
            vehicle_id="V-MAND-05",
            trip_id="T-MAND-05",
            start_at=datetime(2026, 7, 5, 8),
            ownership_type="mandatory_rented",
        )
        plan = TransportPlan(
            demand_parts=delivery_plan.demand_parts,
            trips=(mandatory_july_fifth, *delivery_plan.trips),
        )
        policy = make_policy(
            start_date=date(2026, 7, 5),
            forecast_end_date=date(2026, 7, 5),
        )

        report = TransportPlanValidator(problem, policy).validate(plan)

        self.assertIn("mandatory_vehicle_missing", issue_codes(report))
        self.assertEqual(report.metrics["mandatory_post_window_missing_count"], 1)

    def test_duplicate_physical_vehicle_cost_is_reported(self) -> None:
        """Two reported vehicle lines for one vehicle must be rejected."""

        problem = make_problem()
        base_plan = make_complete_plan()
        policy = make_policy()
        expected = manual_complete_plan_cost()
        duplicate_line = expected.vehicle_cost_lines[0]
        reported = replace(
            expected,
            vehicle_cost_lines=(duplicate_line, duplicate_line),
        )
        plan = replace(base_plan, reported_cost=reported)

        report = TransportPlanValidator(problem, policy).validate(plan)

        self.assertIn("duplicate_physical_vehicle_cost", issue_codes(report))
        self.assertGreater(report.metrics["duplicate_vehicle_cost_count"], 0)

    def test_recomputed_cost_breakdown_passes_ceil_and_sla_checks(self) -> None:
        """A hand-computed report must pass independent cost validation."""

        problem = make_problem()
        base_plan = make_complete_plan()
        policy = make_policy()
        reported = manual_complete_plan_cost()
        plan = replace(base_plan, reported_cost=reported)

        report = TransportPlanValidator(problem, policy).validate(plan)

        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(report.metrics["computed_total_cost_tl"], reported.total_cost_tl)

    def test_travel_time_mismatch_is_reported(self) -> None:
        """A leg duration different from the matrix must be rejected."""

        problem = make_problem()
        base_plan = make_complete_plan()
        trip = base_plan.trips[0]
        leg = trip.legs[0]
        changed_leg = replace(
            leg,
            arrival_at=leg.arrival_at + timedelta(minutes=5),
        )
        changed_trip = replace(trip, legs=(changed_leg,))
        plan = replace(base_plan, trips=(changed_trip,))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("travel_time_mismatch", issue_codes(report))

    def test_load_must_finish_before_departure(self) -> None:
        """A vehicle departure before linked loading ends must be an error."""

        problem = make_problem()
        base_plan = make_complete_plan()
        trip = base_plan.trips[0]
        leg = trip.legs[0]
        load = trip.handling_operations[0]
        changed_leg = replace(
            leg,
            departure_at=load.end_at - timedelta(seconds=1),
            arrival_at=load.end_at - timedelta(seconds=1) + timedelta(hours=1),
        )
        changed_trip = replace(trip, legs=(changed_leg,))
        plan = replace(base_plan, trips=(changed_trip,))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("invalid_load_departure_order", issue_codes(report))

    def test_daily_handling_capacity_excess_is_reported(self) -> None:
        """Parallel or sequential operations still share the daily desi cap."""

        problem = make_problem(handling_capacity=30)
        plan = make_complete_plan()

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("daily_handling_capacity_exceeded", issue_codes(report))

    def test_empty_first_departure_has_no_minimum_fill_error(self) -> None:
        """An empty vehicle movement is valid when no demand is required."""

        problem = make_problem(demands=())
        plan = TransportPlan(demand_parts=(), trips=(make_empty_trip(),))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertTrue(report.is_valid, report.errors)
        self.assertNotIn("minimum_fill", issue_codes(report))


if __name__ == "__main__":
    unittest.main()
