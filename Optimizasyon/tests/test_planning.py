"""Synthetic tests for conservative partitioning, scheduling and baseline plans."""

from __future__ import annotations

from datetime import date, time
import math
import unittest

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandInstance,
    HandlingCapacity,
    MandatoryRentalRoute,
    ProblemData,
    StaticProblemData,
    TravelArc,
    VehicleType,
)
from src.planning import (
    BaselineBuildError,
    ConservativeDirectBaselineBuilder,
    partition_demands,
)

from tests.solution_fixtures import make_policy


VEHICLE_TYPES = (
    VehicleType("Tır", 22_400.0, 400.0, 25.0, 500.0, 25.0),
    VehicleType("Kamyon", 12_000.0, 300.0, 21.0, 318.25, 21.0),
    VehicleType("Hafif Kamyon", 7_200.0, 250.0, 20.0, 364.58, 20.0),
    VehicleType("Kamyonet", 5_600.0, 200.0, 18.0, 197.92, 18.0),
)


def _demand(
    demand_id: str,
    desi: float,
    *,
    demand_date: date = date(2026, 6, 29),
    demand_time: time = time(9),
    origin: str = "A",
    destination: str = "B",
) -> DemandInstance:
    """Create one standard synthetic demand with explicit desi and readiness."""

    return DemandInstance(
        demand_id,
        demand_date,
        demand_time,
        origin,
        destination,
        desi,
        desi > 22_400.0,
    )


def _problem(
    demands: tuple[DemandInstance, ...],
    *,
    handling_a: float = 100_000.0,
    handling_b: float = 100_000.0,
    truck_quota_a: int = 100,
    truck_quota_b: int = 100,
    mandatory_routes: tuple[MandatoryRentalRoute, ...] = (),
) -> ProblemData:
    """Build a two-centre full directed problem for baseline unit tests."""

    travel_times = {vehicle.name: 1.0 for vehicle in VEHICLE_TYPES}
    static = StaticProblemData(
        centers=(Center("A"), Center("B")),
        travel_arcs=(
            TravelArc("A", "B", 100.0, travel_times, 1),
            TravelArc("B", "A", 100.0, travel_times, 1),
        ),
        vehicle_types=VEHICLE_TYPES,
        mandatory_rental_routes=mandatory_routes,
        handling_capacities=(
            HandlingCapacity("A", handling_a),
            HandlingCapacity("B", handling_b),
        ),
        daily_truck_capacities=(
            DailyTruckCapacity("A", truck_quota_a),
            DailyTruckCapacity("B", truck_quota_b),
        ),
    )
    return ProblemData(static, demands)


class DemandPartitionTests(unittest.TestCase):
    """Verify hierarchy, capacity and exact parent-demand balances."""

    def test_large_demand_is_split_to_non_truck_capacity(self) -> None:
        """Preserve 25,000 split and exercise tolerance dust boundaries."""

        parts = partition_demands(
            (_demand("D1", 25_000.0),),
            12_000.0,
            1e-6,
        )
        self.assertEqual([part.part_id for part in parts], ["D1-1", "D1-2", "D1-3"])
        self.assertEqual([part.desi for part in parts], [12_000.0, 12_000.0, 1_000.0])
        self.assertEqual(math.fsum(part.desi for part in parts), 25_000.0)

        # Subtests keep the suite's historical test count stable while making
        # every tolerance boundary independently visible on failure.
        tolerance_cases = (
            (12_000.0000000001, 1),
            (24_000.0000000001, 2),
            (12_000.000002, 2),
        )
        for demand_desi, expected_count in tolerance_cases:
            with self.subTest(demand_desi=demand_desi):
                tolerance_parts = partition_demands(
                    (_demand("D-TOL", demand_desi),),
                    12_000.0,
                    1e-6,
                )
                self.assertEqual(len(tolerance_parts), expected_count)
                self.assertEqual(
                    math.fsum(part.desi for part in tolerance_parts),
                    demand_desi,
                )
                self.assertTrue(
                    all(part.desi <= 12_000.0 + 1e-6 for part in tolerance_parts)
                )

    def test_unsplit_and_zero_demands_have_explicit_policy(self) -> None:
        """Positive unsplit demand gets -1 while zero creates no atom."""

        parts = partition_demands(
            (_demand("D1", 100.0), _demand("D2", 0.0)),
            12_000.0,
            1e-6,
        )
        self.assertEqual([(part.part_id, part.desi) for part in parts], [("D1-1", 100.0)])

class ConservativeDirectBaselineBuilderTests(unittest.TestCase):
    """Exercise direct delivery, capacity ledgers and mandatory schedules."""

    def test_small_demand_is_delivered_and_validator_passes(self) -> None:
        """One positive demand creates one valid direct final delivery."""

        result = ConservativeDirectBaselineBuilder(
            _problem((_demand("D1", 100.0),)), make_policy()
        ).build()
        self.assertTrue(result.validation_report.is_valid)
        self.assertEqual(result.spot_vehicle_count, 1)
        self.assertEqual(result.validation_report.metrics["delivered_desi"], 100.0)

    def test_assignments_equal_atomic_parts_and_zero_demand_adds_no_vehicle(self) -> None:
        """Every spot assignment carries one whole atom; zero demand is inert."""

        result = ConservativeDirectBaselineBuilder(
            _problem((_demand("D1", 25_000.0), _demand("D2", 0.0))),
            make_policy(),
        ).build()
        parts = {part.part_id: part for part in result.plan.demand_parts}
        assignments = [
            assignment
            for trip in result.plan.trips
            if trip.ownership_type == "spot"
            for leg in trip.legs
            for assignment in leg.assignments
        ]
        self.assertEqual(result.spot_vehicle_count, 3)
        self.assertTrue(
            all(item.desi == parts[item.demand_part_id].desi for item in assignments)
        )
        self.assertNotIn("D2", {part.demand_id for part in parts.values()})

    def test_spot_selection_never_uses_truck_even_with_zero_truck_quota(self) -> None:
        """A zero Tır quota does not block feasible non-Tır direct transport."""

        problem = _problem(
            (_demand("D1", 11_000.0),),
            truck_quota_a=0,
            truck_quota_b=0,
        )
        result = ConservativeDirectBaselineBuilder(problem, make_policy()).build()
        spot_types = {
            trip.vehicle_type
            for trip in result.plan.trips
            if trip.ownership_type == "spot"
        }
        self.assertNotIn("Tır", spot_types)
        self.assertTrue(result.validation_report.is_valid)

    def test_full_daily_handling_moves_next_operation_to_next_day(self) -> None:
        """Shared origin/destination desi pools defer the second 60-desi atom."""

        problem = _problem(
            (_demand("D1", 60.0), _demand("D2", 60.0)),
            handling_a=100.0,
            handling_b=100.0,
        )
        result = ConservativeDirectBaselineBuilder(problem, make_policy()).build()
        second = next(
            trip for trip in result.plan.trips if trip.vehicle_id == "SPOT-00002"
        )
        load = second.handling_operations[0]
        self.assertEqual(load.start_at.date(), date(2026, 6, 30))
        self.assertTrue(result.validation_report.is_valid)

    def test_operations_never_cross_midnight(self) -> None:
        """A late load is moved wholesale to midnight on the next day."""

        problem = _problem(
            (_demand("D1", 1_000.0, demand_time=time(23, 59)),),
        )
        result = ConservativeDirectBaselineBuilder(problem, make_policy()).build()
        for trip in result.plan.trips:
            for operation in trip.handling_operations:
                self.assertEqual(operation.start_at.date(), operation.end_at.date())
        load = result.plan.trips[0].handling_operations[0]
        self.assertEqual(load.start_at, load.start_at.replace(hour=0, minute=0))

    def test_destination_wait_is_included_in_physical_vehicle_cost(self) -> None:
        """A destination-capacity wait expands the second vehicle envelope."""

        problem = _problem(
            (_demand("D1", 60.0), _demand("D2", 60.0)),
            handling_a=1_000.0,
            handling_b=100.0,
        )
        result = ConservativeDirectBaselineBuilder(problem, make_policy()).build()
        cost = result.plan.reported_cost
        second_line = next(
            line for line in cost.vehicle_cost_lines
            if line.vehicle_id == "SPOT-00002"
        )
        self.assertGreater(second_line.usage_hours, 15.0)
        self.assertEqual(second_line.charged_hours, math.ceil(second_line.usage_hours))

    def test_mandatory_counts_are_exact_empty_costed_and_quota_consuming(self) -> None:
        """Two empty mandatory Tırs still create cost and two visits per centre."""

        routes = (MandatoryRentalRoute("A", "B", "Tır", 2),)
        result = ConservativeDirectBaselineBuilder(
            _problem((_demand("D1", 10.0),), mandatory_routes=routes),
            make_policy(),
        ).build()
        mandatory = [
            trip for trip in result.plan.trips
            if trip.ownership_type == "mandatory_rented"
        ]
        self.assertEqual(len(mandatory), 2)
        self.assertTrue(
            all(trip.legs[0].departure_at.time() == time(9) for trip in mandatory)
        )
        self.assertTrue(all(not trip.legs[0].assignments for trip in mandatory))
        self.assertTrue(all(not trip.handling_operations for trip in mandatory))
        self.assertGreater(
            result.plan.reported_cost.mandatory_rented_hourly_cost_tl,
            0.0,
        )
        visits = result.validation_report.metrics["truck_visits_by_center_day"]
        self.assertEqual(visits["A|2026-06-29"], 2)
        self.assertEqual(visits["B|2026-06-29"], 2)

    def test_mandatory_rentals_continue_through_post_window_completion_day(self) -> None:
        """A 6 July delivery extends one daily mandatory route to eight days."""

        routes = (MandatoryRentalRoute("A", "B", "Kamyon", 1),)
        problem = _problem(
            (_demand("D1", 10.0, demand_date=date(2026, 7, 6)),),
            mandatory_routes=routes,
        )
        policy = make_policy(forecast_end_date=date(2026, 7, 5))
        result = ConservativeDirectBaselineBuilder(problem, policy).build()
        self.assertEqual(result.mandatory_vehicle_count, 8)
        later_departures = [
            trip.legs[0].departure_at.time()
            for trip in result.plan.trips
            if trip.ownership_type == "mandatory_rented"
            and trip.legs[0].departure_at.date() > date(2026, 6, 29)
        ]
        self.assertTrue(all(departure == time.min for departure in later_departures))
        self.assertEqual(
            result.validation_report.metrics["mandatory_schedule_end_date"],
            "2026-07-06",
        )

    def test_invalid_mandatory_truck_quota_raises_structured_error(self) -> None:
        """Builder does not return success when fixed Tır counts exceed quota."""

        routes = (MandatoryRentalRoute("A", "B", "Tır", 2),)
        builder = ConservativeDirectBaselineBuilder(
            _problem(
                (_demand("D1", 10.0),),
                mandatory_routes=routes,
                truck_quota_a=1,
                truck_quota_b=1,
            ),
            make_policy(),
        )
        with self.assertRaises(BaselineBuildError) as caught:
            builder.build()
        self.assertIn("daily_truck_quota_exceeded", caught.exception.error_codes)

    def test_same_input_produces_identical_plan_ids_and_times(self) -> None:
        """Wall-clock build diagnostics vary, but physical plans do not."""

        problem = _problem((_demand("D1", 13_000.0), _demand("D2", 100.0)))
        first = ConservativeDirectBaselineBuilder(problem, make_policy()).build().plan
        second = ConservativeDirectBaselineBuilder(problem, make_policy()).build().plan
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
