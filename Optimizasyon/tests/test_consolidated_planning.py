"""Synthetic tests for strategy-local same-O-D/day FFD consolidation."""

from __future__ import annotations

from datetime import date, time
import math
import unittest

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandInstance,
    HandlingCapacity,
    ProblemData,
    StaticProblemData,
    TravelArc,
    VehicleType,
)
from src.planning import GreedySameODConsolidatedBuilder
from tests.solution_fixtures import make_policy


VEHICLES = (
    VehicleType("Tır", 200.0, 20.0, 3.0, 30.0, 4.0),
    VehicleType("Kamyon", 100.0, 10.0, 2.0, 16.0, 2.0),
    VehicleType("Hafif Kamyon", 70.0, 9.0, 1.8, 14.0, 1.8),
    VehicleType("Kamyonet", 50.0, 8.0, 1.5, 12.0, 1.5),
)


def _demand(
    demand_id: str,
    desi: float,
    demand_time: time,
    *,
    destination: str = "B",
    demand_date: date = date(2026, 6, 29),
) -> DemandInstance:
    """Create one A-origin synthetic demand for consolidation scenarios."""

    return DemandInstance(
        demand_id,
        demand_date,
        demand_time,
        "A",
        destination,
        desi,
        desi > 200.0,
    )


def _problem(
    demands: tuple[DemandInstance, ...],
    *,
    truck_quota: int = 10,
) -> ProblemData:
    """Build a three-centre directed problem with 100-desi non-Tır maximum.

    ``truck_quota`` sets every centre's daily Tır allowance.  Zero reproduces a
    network where no spot Tır may run, so bins stay within the largest non-Tır
    capacity.
    """

    travel_times = {vehicle.name: 1.0 for vehicle in VEHICLES}
    centers = ("A", "B", "C")
    arcs = tuple(
        TravelArc(origin, destination, 100.0, travel_times, 1)
        for origin in centers
        for destination in centers
        if origin != destination
    )
    static = StaticProblemData(
        centers=tuple(Center(center) for center in centers),
        travel_arcs=arcs,
        vehicle_types=VEHICLES,
        mandatory_rental_routes=(),
        handling_capacities=tuple(
            HandlingCapacity(center, 10_000.0) for center in centers
        ),
        daily_truck_capacities=tuple(
            DailyTruckCapacity(center, truck_quota) for center in centers
        ),
    )
    return ProblemData(static, demands)


class GreedySameODConsolidatedBuilderTests(unittest.TestCase):
    """Verify grouping, FFD packing, grouped operations and exact balances."""

    def test_morning_and_evening_parts_share_one_vehicle_after_17(self) -> None:
        """Same date/O-D 09:00 and 17:00 parts consolidate and wait for 17:00."""

        problem = _problem(
            (
                _demand("D1", 40.0, time(9)),
                _demand("D2", 50.0, time(17)),
            )
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        spot = _spot_trips(result.plan.trips)
        self.assertEqual(len(spot), 1)
        self.assertEqual(len(spot[0].legs[0].assignments), 2)
        self.assertGreaterEqual(spot[0].handling_operations[0].start_at.time(), time(17))
        self.assertTrue(result.validation_report.is_valid)

    def test_ffd_capacity_overflow_is_deterministic(self) -> None:
        """FFD packs 60+40 first and the second 60 into a stable second bin.

        Daily Tır quota is zero here, so bins stay within the 100-desi Kamyon
        capacity and the packing actually has to overflow.
        """

        problem = _problem(
            (
                _demand("D1", 60.0, time(9)),
                _demand("D2", 60.0, time(9)),
                _demand("D3", 40.0, time(9)),
            ),
            truck_quota=0,
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        groups = [
            tuple(item.demand_part_id for item in trip.legs[0].assignments)
            for trip in _spot_trips(result.plan.trips)
        ]
        self.assertEqual(groups, [("D1-1", "D3-1"), ("D2-1",)])
        self.assertEqual(
            [trip.vehicle_id for trip in _spot_trips(result.plan.trips)],
            ["GSOD-00001", "GSOD-00002"],
        )

    def test_spare_truck_quota_merges_bins_into_one_truck(self) -> None:
        """With Tır quota free, the same three parts ride one 160-desi Tır."""

        problem = _problem(
            (
                _demand("D1", 60.0, time(9)),
                _demand("D2", 60.0, time(9)),
                _demand("D3", 40.0, time(9)),
            ),
            truck_quota=10,
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        spot = _spot_trips(result.plan.trips)

        self.assertEqual(len(spot), 1)
        self.assertEqual(spot[0].vehicle_type, "Tır")
        self.assertEqual(
            tuple(item.demand_part_id for item in spot[0].legs[0].assignments),
            ("D1-1", "D2-1", "D3-1"),
        )
        self.assertTrue(result.validation_report.is_valid)

    def test_partial_truck_quota_spends_what_is_available(self) -> None:
        """Two Tır bins are wanted but one visit is free, so one Tır is used.

        Four 100-desi atoms pack into two 200-desi Tır bins.  With a single
        daily visit per centre only one of them may run as a Tır; the other is
        re-packed into non-Tır bins rather than the whole group giving up.
        """

        problem = _problem(
            (
                _demand("D1", 200.0, time(9)),
                _demand("D2", 200.0, time(9)),
            ),
            truck_quota=1,
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        spot = _spot_trips(result.plan.trips)
        truck_trips = [trip for trip in spot if trip.vehicle_type == "Tır"]

        self.assertEqual(len(truck_trips), 1)
        self.assertAlmostEqual(
            math.fsum(
                item.desi for item in truck_trips[0].legs[0].assignments
            ),
            200.0,
        )
        # The remaining 200 desi rides non-Tır vehicles.
        self.assertTrue(
            all(trip.vehicle_type != "Tır" for trip in spot if trip not in truck_trips)
        )
        self.assertTrue(result.validation_report.is_valid)

    def test_zero_truck_quota_never_reserves_a_truck(self) -> None:
        """A network without Tır quota keeps every bin inside Kamyon capacity."""

        problem = _problem(
            (
                _demand("D1", 200.0, time(9)),
                _demand("D2", 200.0, time(9)),
            ),
            truck_quota=0,
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        spot = _spot_trips(result.plan.trips)

        self.assertTrue(all(trip.vehicle_type != "Tır" for trip in spot))
        self.assertTrue(result.validation_report.is_valid)

    def test_different_destinations_never_share_a_vehicle(self) -> None:
        """Equal-date A-B and A-C parts remain in separate O-D groups."""

        problem = _problem(
            (
                _demand("D1", 40.0, time(9), destination="B"),
                _demand("D2", 40.0, time(9), destination="C"),
            )
        )
        result = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        routes = [
            (trip.legs[0].origin, trip.legs[0].destination)
            for trip in _spot_trips(result.plan.trips)
        ]
        self.assertEqual(routes, [("A", "B"), ("A", "C")])

    def test_grouped_handling_cost_and_demand_balances_are_exact(self) -> None:
        """One 90-desi vehicle has two assignments, grouped duration and one cost."""

        demands = (
            _demand("D1", 40.0, time(9)),
            _demand("D2", 50.0, time(17)),
        )
        result = GreedySameODConsolidatedBuilder(
            _problem(demands),
            make_policy(),
        ).build()
        trip = _spot_trips(result.plan.trips)[0]
        load, unload = trip.handling_operations
        self.assertEqual(load.desi, math.fsum((40.0, 50.0)))
        self.assertEqual(unload.desi, math.fsum((40.0, 50.0)))
        self.assertEqual((load.end_at - load.start_at).total_seconds(), 54.0)
        self.assertEqual((unload.end_at - unload.start_at).total_seconds(), 54.0)
        self.assertEqual(len(result.plan.reported_cost.vehicle_cost_lines), 1)
        part_by_demand = {
            part.demand_id: part.desi for part in result.plan.demand_parts
        }
        self.assertEqual(part_by_demand, {"D1": 40.0, "D2": 50.0})
        self.assertEqual(
            result.validation_report.metrics["delivered_desi"],
            math.fsum(demand.desi for demand in demands),
        )

    def test_repeated_build_has_identical_ids_groups_and_times(self) -> None:
        """Same inputs yield equal immutable plans despite runtime variation."""

        problem = _problem(
            (
                _demand("D1", 60.0, time(9)),
                _demand("D2", 40.0, time(17)),
                _demand("D3", 30.0, time(9)),
            )
        )
        first = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        second = GreedySameODConsolidatedBuilder(problem, make_policy()).build()
        self.assertEqual(first.plan, second.plan)


def _spot_trips(trips: tuple) -> tuple:
    """Return spot trips from a synthetic plan without assuming list order APIs."""

    return tuple(trip for trip in trips if trip.ownership_type == "spot")


if __name__ == "__main__":
    unittest.main()
