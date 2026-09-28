"""Synthetic tests for the same-origin two-destination greedy baseline."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time
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
    GreedyMandatoryRentalFilledBuilder,
    GreedySameOriginTwoStopBuilder,
    HandlingCapacityLedger,
)
from tests.solution_fixtures import make_policy


CENTERS = ("A", "B", "C", "D", "E")
VEHICLES = (
    VehicleType("Tır", 200.0, 20.0, 2.0, 30.0, 3.0),
    VehicleType("Kamyon", 120.0, 12.0, 1.0, 10.0, 0.8),
    VehicleType("Kamyonet", 60.0, 8.0, 1.0, 6.0, 1.0),
)


def _demand(
    demand_id: str,
    destination: str,
    desi: float = 40.0,
    *,
    origin: str = "A",
    demand_date: date = date(2026, 6, 29),
    demand_time: time = time(9),
) -> DemandInstance:
    """Create one explicit synthetic demand for a route-pairing scenario."""

    return DemandInstance(
        demand_id,
        demand_date,
        demand_time,
        origin,
        destination,
        desi,
        desi > 200.0,
    )


def _problem(
    demands: tuple[DemandInstance, ...],
    *,
    distances: dict[tuple[str, str], float] | None = None,
    travel_hours: dict[tuple[str, str], float] | None = None,
    missing_arcs: frozenset[tuple[str, str]] = frozenset(),
    target_days: int = 1,
    handling: dict[str, float] | None = None,
    mandatory_routes: tuple[MandatoryRentalRoute, ...] = (),
) -> ProblemData:
    """Build a directed five-centre network with configurable route economics."""

    distance_map = distances or {}
    travel_map = travel_hours or {}
    arcs = tuple(
        TravelArc(
            origin,
            destination,
            distance_map.get((origin, destination), 100.0),
            {
                vehicle.name: travel_map.get((origin, destination), 1.0)
                for vehicle in VEHICLES
            },
            target_days,
        )
        for origin in CENTERS
        for destination in CENTERS
        if origin != destination and (origin, destination) not in missing_arcs
    )
    handling_map = handling or {}
    static = StaticProblemData(
        tuple(Center(center) for center in CENTERS),
        arcs,
        VEHICLES,
        mandatory_routes,
        tuple(
            HandlingCapacity(center, handling_map.get(center, 10_000.0))
            for center in CENTERS
        ),
        tuple(DailyTruckCapacity(center, 20) for center in CENTERS),
    )
    return ProblemData(static, demands)


def _policy_for(demands: tuple[DemandInstance, ...]):
    """Create a policy whose mandatory forecast window covers all test dates."""

    return make_policy(
        forecast_end_date=max(demand.demand_date for demand in demands)
    )


def _two_stop_trips(result) -> tuple:
    """Return only spot trips containing exactly two physical legs."""

    return tuple(
        trip for trip in result.plan.trips
        if trip.ownership_type == "spot" and len(trip.legs) == 2
    )


def _part_ids(trip) -> frozenset[str]:
    """Return unique atomic part IDs carried anywhere in one trip."""

    return frozenset(
        assignment.demand_part_id
        for leg in trip.legs
        for assignment in leg.assignments
    )


class HandlingLedgerTransactionTests(unittest.TestCase):
    """Verify forked scheduling is isolated until an explicit compatible commit."""

    def test_fork_is_atomic_and_commit_applies_only_successful_trial(self) -> None:
        """Rejected trial usage stays isolated; accepted usage commits once."""

        ledger = HandlingCapacityLedger(
            (HandlingCapacity("A", 100.0),),
            minutes_per_desi=0.01,
            desi_tolerance=1e-6,
        )
        trial = ledger.fork()
        trial.schedule("A", datetime(2026, 6, 29, 9), 60.0)
        self.assertEqual(ledger.snapshot(), ())
        ledger.commit_from(trial)
        self.assertEqual(ledger.snapshot()[0].used_desi, 60.0)


class GreedySameOriginTwoStopBuilderTests(unittest.TestCase):
    """Exercise eligibility, route choice, load decay, economics and validation."""

    def test_economical_pair_has_one_load_two_legs_and_two_exact_unloads(self) -> None:
        """Two destinations share one non-Tır while every part is delivered once."""

        demands = (_demand("D1", "B"), _demand("D2", "C"))
        result = GreedySameOriginTwoStopBuilder(
            _problem(demands, distances={("B", "C"): 10.0}),
            _policy_for(demands),
        ).build()
        trip = _two_stop_trips(result)[0]
        load, unload_one, unload_two = trip.handling_operations
        self.assertEqual((len(trip.legs), len(trip.handling_operations)), (2, 3))
        self.assertEqual(load.desi, 80.0)
        self.assertEqual((unload_one.desi, unload_two.desi), (40.0, 40.0))
        self.assertEqual(sum(item.desi for item in trip.legs[0].assignments), 80.0)
        self.assertEqual(sum(item.desi for item in trip.legs[1].assignments), 40.0)
        self.assertEqual(trip.legs[1].departure_at, unload_one.end_at)
        self.assertEqual(trip.vehicle_type, "Kamyon")
        self.assertNotEqual(trip.vehicle_type, "Tır")
        self.assertTrue(result.validation_report.is_valid)
        finals = Counter(
            item.demand_part_id
            for leg in trip.legs
            for item in leg.assignments
            if item.is_final_delivery
        )
        self.assertEqual(finals, Counter({"D1-1": 1, "D2-1": 1}))
        delivered_at = {
            line.demand_part_id: line.delivered_at
            for line in result.plan.reported_cost.sla_penalty_lines
        }
        self.assertEqual(delivered_at[unload_one.demand_part_ids[0]], unload_one.end_at)
        self.assertEqual(delivered_at[unload_two.demand_part_ids[0]], unload_two.end_at)

    def test_less_expensive_stop_order_and_ids_are_deterministic(self) -> None:
        """B-C short arc wins both runs with identical route, IDs and timestamps."""

        demands = (_demand("D1", "B"), _demand("D2", "C"))
        problem = _problem(
            demands,
            distances={("B", "C"): 10.0, ("C", "B"): 90.0},
        )
        first = GreedySameOriginTwoStopBuilder(
            problem, _policy_for(demands)
        ).build()
        second = GreedySameOriginTwoStopBuilder(
            problem, _policy_for(demands)
        ).build()
        self.assertEqual(
            tuple(leg.destination for leg in _two_stop_trips(first)[0].legs),
            ("B", "C"),
        )
        self.assertEqual(first.plan, second.plan)

    def test_sla_cost_can_override_the_shortest_distance_order(self) -> None:
        """Fast C-B is selected although slow B-C has much fewer kilometres."""

        demands = (_demand("D1", "B"), _demand("D2", "C"))
        problem = _problem(
            demands,
            distances={
                ("A", "B"): 10.0,
                ("B", "C"): 10.0,
                ("A", "C"): 100.0,
                ("C", "B"): 100.0,
            },
            travel_hours={
                ("A", "B"): 10.0,
                ("B", "C"): 10.0,
                ("A", "C"): 1.0,
                ("C", "B"): 1.0,
            },
            target_days=0,
        )
        result = GreedySameOriginTwoStopBuilder(
            problem, _policy_for(demands)
        ).build()
        destinations = tuple(
            leg.destination for leg in _two_stop_trips(result)[0].legs
        )
        self.assertEqual(destinations, ("C", "B"))

    def test_origin_date_and_same_destination_boundaries_are_preserved(self) -> None:
        """Only A-B/A-C pair; D-origin, next-day and same-O-D atoms stay direct."""

        demands = (
            _demand("D1", "B"),
            _demand("D2", "C"),
            _demand("D3", "B", origin="D"),
            _demand("D4", "E", demand_date=date(2026, 6, 30)),
            _demand("D5", "E", 20.0),
            _demand("D6", "E", 20.0),
        )
        result = GreedySameOriginTwoStopBuilder(
            _problem(demands, distances={("B", "C"): 10.0}),
            _policy_for(demands),
        ).build()
        paired = _part_ids(_two_stop_trips(result)[0])
        self.assertEqual(paired, frozenset({"D1-1", "D2-1"}))
        same_od = next(
            trip for trip in result.plan.trips
            if trip.ownership_type == "spot"
            and {"D5-1", "D6-1"}.issubset(_part_ids(trip))
        )
        self.assertEqual(len(same_od.legs), 1)

    def test_capacity_and_missing_interdestination_arcs_reject_pairs(self) -> None:
        """Oversized D pair and missing D-B/E link remain exact direct trips."""

        demands = (
            _demand("D1", "B"),
            _demand("D2", "C"),
            _demand("D3", "C", 70.0, origin="D"),
            _demand("D4", "E", 70.0, origin="D"),
            _demand("D5", "B", 20.0, origin="E"),
            _demand("D6", "D", 20.0, origin="E"),
        )
        result = GreedySameOriginTwoStopBuilder(
            _problem(
                demands,
                distances={("B", "C"): 10.0},
                missing_arcs=frozenset({("B", "D"), ("D", "B")}),
            ),
            _policy_for(demands),
        ).build()
        all_paired = set().union(*(_part_ids(trip) for trip in _two_stop_trips(result)))
        self.assertFalse({"D3-1", "D4-1", "D5-1", "D6-1"} & all_paired)
        self.assertTrue(result.validation_report.is_valid)

    def test_non_improving_pair_is_rejected_while_other_pair_succeeds(self) -> None:
        """A positive A pair survives; costly D-C/E remains two direct trips."""

        demands = (
            _demand("D1", "B"),
            _demand("D2", "E"),
            _demand("D3", "C", origin="D"),
            _demand("D4", "E", origin="D"),
        )
        for interdestination_distance in (500.0, 0.0):
            with self.subTest(interdestination_distance=interdestination_distance):
                distances = {
                    ("B", "E"): 10.0,
                    ("E", "B"): 10.0,
                    ("D", "C"): 5.0,
                    ("D", "E"): 5.0,
                    ("C", "E"): interdestination_distance,
                    ("E", "C"): interdestination_distance,
                }
                result = GreedySameOriginTwoStopBuilder(
                    _problem(demands, distances=distances),
                    _policy_for(demands),
                ).build()
                paired = set().union(
                    *(_part_ids(trip) for trip in _two_stop_trips(result))
                )
                self.assertTrue({"D1-1", "D2-1"}.issubset(paired))
                self.assertFalse({"D3-1", "D4-1"} & paired)

    def test_load_waits_for_latest_ready_and_each_bin_is_used_once(self) -> None:
        """A 17:00 atom delays one selected pair while a third bin stays direct."""

        demands = (
            _demand("D1", "B", demand_time=time(9)),
            _demand("D2", "C", demand_time=time(17)),
            _demand("D3", "D", demand_time=time(9)),
        )
        result = GreedySameOriginTwoStopBuilder(
            _problem(
                demands,
                distances={("B", "C"): 10.0, ("C", "B"): 10.0},
            ),
            _policy_for(demands),
        ).build()
        combined = _two_stop_trips(result)
        self.assertEqual(len(combined), 1)
        self.assertGreaterEqual(combined[0].handling_operations[0].start_at.time(), time(17))
        final_counts = Counter(
            item.demand_part_id
            for trip in result.plan.trips
            for leg in trip.legs
            for item in leg.assignments
            if item.is_final_delivery
        )
        self.assertEqual(final_counts, Counter({"D1-1": 1, "D2-1": 1, "D3-1": 1}))
        self.assertEqual(result.spot_vehicle_count, 2)

    def test_same_day_load_failure_falls_back_without_imaginary_usage(self) -> None:
        """A-origin 80 desi cannot pair in 60/day; D pair keeps 6D economical."""

        demands = (
            _demand("D1", "B"),
            _demand("D2", "C"),
            _demand("D3", "B", origin="D"),
            _demand("D4", "E", origin="D"),
        )
        result = GreedySameOriginTwoStopBuilder(
            _problem(
                demands,
                distances={("B", "E"): 10.0, ("E", "B"): 10.0},
                handling={"A": 60.0},
            ),
            _policy_for(demands),
        ).build()
        paired = set().union(*(_part_ids(trip) for trip in _two_stop_trips(result)))
        self.assertFalse({"D1-1", "D2-1"} & paired)
        usage = {
            (item.center, item.operation_date): item.used_desi
            for item in result.handling_usage
        }
        self.assertEqual(usage[("A", date(2026, 6, 29))], 40.0)
        self.assertEqual(usage[("A", date(2026, 6, 30))], 40.0)

    def test_destination_capacity_wait_enters_vehicle_envelope_cost(self) -> None:
        """Mandatory B unload fills day one; two-stop B unload waits to day two."""

        demands = (
            _demand("D-M", "B", origin="D"),
            _demand("D1", "B"),
            _demand("D2", "C"),
        )
        route = (MandatoryRentalRoute("D", "B", "Tır", 1),)
        problem = _problem(
            demands,
            distances={("B", "C"): 10.0, ("C", "B"): 10.0},
            handling={"B": 40.0},
            mandatory_routes=route,
        )
        result = GreedySameOriginTwoStopBuilder(
            problem, _policy_for(demands)
        ).build()
        trip = _two_stop_trips(result)[0]
        b_unload = next(op for op in trip.handling_operations if op.center == "B")
        self.assertEqual(b_unload.start_at.date(), date(2026, 6, 30))
        line = next(
            item for item in result.plan.reported_cost.vehicle_cost_lines
            if item.vehicle_id == trip.vehicle_id
        )
        self.assertGreater(line.usage_hours, 15.0)

    def test_mandatory_schedule_quota_and_part_balance_match_6b(self) -> None:
        """6D changes only spot routing; mandatory trips/quota and atoms stay exact."""

        demands = (
            _demand("D-M", "B", 30.0, origin="D"),
            _demand("D1", "B"),
            _demand("D2", "C"),
        )
        problem = _problem(
            demands,
            distances={("B", "C"): 10.0},
            mandatory_routes=(MandatoryRentalRoute("D", "B", "Tır", 1),),
        )
        policy = _policy_for(demands)
        base = GreedyMandatoryRentalFilledBuilder(problem, policy).build()
        result = GreedySameOriginTwoStopBuilder(problem, policy).build()
        base_mandatory = tuple(
            trip for trip in base.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        result_mandatory = tuple(
            trip for trip in result.plan.trips
            if trip.ownership_type == "mandatory_rented"
        )
        self.assertEqual(result.plan.demand_parts, base.plan.demand_parts)
        self.assertEqual(result_mandatory, base_mandatory)
        self.assertEqual(
            result.validation_report.metrics["truck_visits_by_center_day"],
            base.validation_report.metrics["truck_visits_by_center_day"],
        )
        self.assertLess(result.total_cost_tl, base.total_cost_tl)
        self.assertTrue(result.validation_report.is_valid)
        self.assertTrue(math.isfinite(result.total_cost_tl))


if __name__ == "__main__":
    unittest.main()
