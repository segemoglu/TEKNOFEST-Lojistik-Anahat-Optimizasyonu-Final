"""Synthetic tests for greedy fixed-route mandatory-rental filling."""

from __future__ import annotations

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
    HandlingCapacityLedger,
)
from tests.solution_fixtures import make_policy


VEHICLES = (
    VehicleType("Tır", 100.0, 10.0, 2.0, 30.0, 4.0),
    VehicleType("Kamyon", 120.0, 9.0, 1.8, 16.0, 2.0),
    VehicleType("Kamyonet", 50.0, 8.0, 1.5, 12.0, 1.5),
)


def _demand(
    demand_id: str,
    desi: float,
    demand_time: time = time(9),
    *,
    demand_date: date = date(2026, 6, 29),
    destination: str = "B",
) -> DemandInstance:
    """Create one A-origin demand for mandatory-filling scenarios."""

    return DemandInstance(
        demand_id,
        demand_date,
        demand_time,
        "A",
        destination,
        desi,
        desi > 120.0,
    )


def _problem(
    demands: tuple[DemandInstance, ...],
    *,
    route_type: str = "Tır",
    route_count: int = 1,
    handling_a: float = 10_000.0,
    handling_b: float = 10_000.0,
    truck_quota: int = 10,
    tir_travel_hours: float = 1.0,
) -> ProblemData:
    """Build a three-centre problem with one configurable A-B mandatory route."""

    centers = ("A", "B", "C")
    travel_times = {
        "Tır": tir_travel_hours,
        "Kamyon": 1.0,
        "Kamyonet": 1.0,
    }
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
        mandatory_rental_routes=(
            MandatoryRentalRoute("A", "B", route_type, route_count),
        ),
        handling_capacities=(
            HandlingCapacity("A", handling_a),
            HandlingCapacity("B", handling_b),
            HandlingCapacity("C", 10_000.0),
        ),
        daily_truck_capacities=tuple(
            DailyTruckCapacity(center, truck_quota) for center in centers
        ),
    )
    return ProblemData(static, demands)


class HandlingDeadlineLedgerTests(unittest.TestCase):
    """Verify failed same-day reservations leave no imaginary desi usage."""

    def test_try_schedule_before_failure_is_atomic(self) -> None:
        """A failed second reservation does not alter the first 60-desi usage."""

        ledger = HandlingCapacityLedger(
            (HandlingCapacity("A", 100.0),),
            minutes_per_desi=0.01,
            desi_tolerance=1e-6,
        )
        first = ledger.try_schedule_before(
            "A",
            datetime(2026, 6, 29, 9),
            60.0,
            datetime(2026, 6, 30),
        )
        before_failure = ledger.snapshot()
        rejected = ledger.try_schedule_before(
            "A",
            datetime(2026, 6, 29, 17),
            60.0,
            datetime(2026, 6, 30),
        )
        self.assertIsNotNone(first)
        self.assertIsNone(rejected)
        self.assertEqual(ledger.snapshot(), before_failure)


class GreedyMandatoryRentalFilledBuilderTests(unittest.TestCase):
    """Verify eligibility, packing, timing, fallback and physical validation."""

    def test_same_route_morning_and_evening_parts_share_mandatory_vehicle(self) -> None:
        """Eligible 09:00/17:00 parts share grouped operations after readiness."""

        problem = _problem(
            (
                _demand("D1", 40.0, time(9)),
                _demand("D2", 60.0, time(17)),
            )
        )
        result = GreedyMandatoryRentalFilledBuilder(problem, make_policy()).build()
        loaded = _loaded_mandatory(result.plan.trips)
        self.assertEqual(len(loaded), 1)
        trip = loaded[0]
        self.assertEqual(len(trip.legs[0].assignments), 2)
        load, unload = trip.handling_operations
        self.assertGreaterEqual(load.start_at.time(), time(17))
        self.assertEqual(load.desi, math.fsum((40.0, 60.0)))
        self.assertEqual(unload.desi, math.fsum((40.0, 60.0)))
        self.assertEqual((load.end_at - load.start_at).total_seconds(), 60.0)
        self.assertLessEqual(load.desi, 100.0 + 1e-6)
        mandatory_cost_lines = [
            line for line in result.plan.reported_cost.vehicle_cost_lines
            if line.ownership_type == "mandatory_rented"
        ]
        self.assertEqual(len(mandatory_cost_lines), 1)
        self.assertTrue(result.validation_report.is_valid)

    def test_ineligible_route_date_and_capacity_parts_return_to_spot(self) -> None:
        """Different O-D/date and 110-desi Tır excess never enter mandatory load."""

        scenarios = (
            _demand("D-OD", 40.0, destination="C"),
            _demand("D-DATE", 40.0, demand_date=date(2026, 6, 30)),
            _demand("D-CAP", 110.0),
        )
        for demand in scenarios:
            with self.subTest(demand_id=demand.demand_id):
                result = GreedyMandatoryRentalFilledBuilder(
                    _problem((demand,)),
                    make_policy(),
                ).build()
                self.assertFalse(_loaded_mandatory(result.plan.trips))
                spot_ids = {
                    assignment.demand_part_id
                    for trip in _spot_trips(result.plan.trips)
                    for assignment in trip.legs[0].assignments
                }
                self.assertEqual(spot_ids, {f"{demand.demand_id}-1"})
                self.assertTrue(result.validation_report.is_valid)

    def test_multiple_boxes_fill_deterministically_and_empty_box_is_costed(self) -> None:
        """Three rentals carry 60+40, 60 and empty in stable sequence order."""

        problem = _problem(
            (
                _demand("D1", 60.0),
                _demand("D2", 60.0),
                _demand("D3", 40.0),
            ),
            route_count=3,
        )
        first = GreedyMandatoryRentalFilledBuilder(problem, make_policy()).build()
        second = GreedyMandatoryRentalFilledBuilder(problem, make_policy()).build()
        mandatory = _mandatory_trips(first.plan.trips)
        groups = [
            tuple(item.demand_part_id for item in trip.legs[0].assignments)
            for trip in mandatory
        ]
        self.assertEqual(groups, [("D1-1", "D3-1"), ("D2-1",), ()])
        self.assertEqual(first.plan, second.plan)
        self.assertTrue(
            all(
                math.fsum(item.desi for item in trip.legs[0].assignments)
                <= 100.0 + 1e-6
                for trip in mandatory
            )
        )
        cost_ids = {
            line.vehicle_id for line in first.plan.reported_cost.vehicle_cost_lines
        }
        self.assertTrue(all(trip.vehicle_id in cost_ids for trip in mandatory))
        self.assertEqual(first.mandatory_vehicle_count, 3)

    def test_unassigned_part_is_delivered_once_by_spot_with_exact_balance(self) -> None:
        """A 110-desi part bypasses Tır and has one final spot delivery only."""

        demand = _demand("D1", 110.0)
        result = GreedyMandatoryRentalFilledBuilder(
            _problem((demand,)),
            make_policy(),
        ).build()
        finals = [
            assignment
            for trip in result.plan.trips
            for leg in trip.legs
            for assignment in leg.assignments
            if assignment.is_final_delivery
        ]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].desi, demand.desi)
        self.assertEqual(result.plan.demand_parts[0].desi, demand.desi)
        self.assertEqual(result.validation_report.metrics["delivered_desi"], demand.desi)

    def test_same_day_capacity_failure_leaves_mandatory_empty_and_uses_spot(self) -> None:
        """Second 60-desi box fails atomically, stays empty and moves to spot."""

        result = GreedyMandatoryRentalFilledBuilder(
            _problem(
                (_demand("D1", 60.0), _demand("D2", 60.0)),
                route_count=2,
                handling_a=100.0,
            ),
            make_policy(),
        ).build()
        mandatory = _mandatory_trips(result.plan.trips)
        self.assertEqual(sum(bool(trip.legs[0].assignments) for trip in mandatory), 1)
        self.assertEqual(len(_spot_trips(result.plan.trips)), 1)
        usage = {
            (item.center, item.operation_date): item.used_desi
            for item in result.handling_usage
        }
        self.assertEqual(usage[("A", date(2026, 6, 29))], 60.0)
        self.assertEqual(usage[("A", date(2026, 6, 30))], 60.0)

    def test_short_lane_truck_carries_evening_cargo_and_arrives_same_day(self) -> None:
        """A one-hour Tır loads 17:00 cargo and still lands on its service day."""

        feasible = GreedyMandatoryRentalFilledBuilder(
            _problem((_demand("D1", 60.0, time(17)),)),
            make_policy(),
        ).build()
        loaded = _loaded_mandatory(feasible.plan.trips)[0]
        self.assertEqual(
            loaded.legs[0].arrival_at.date(),
            loaded.legs[0].departure_at.date(),
        )

    def test_saturated_destination_quota_sends_late_cargo_back_to_spot(self) -> None:
        """With no spare arrival quota the same-day guard still rejects the load.

        Quota one, consumed by the single mandatory Tır, leaves no room for the
        two arrivals a next-day route could put on one calendar day, so the
        eight-hour lane must refuse 17:00 cargo.
        """

        result = GreedyMandatoryRentalFilledBuilder(
            _problem(
                (_demand("D1", 60.0, time(17)),),
                tir_travel_hours=8.0,
                truck_quota=1,
            ),
            make_policy(),
        ).build()

        self.assertFalse(_loaded_mandatory(result.plan.trips))
        self.assertEqual(len(_spot_trips(result.plan.trips)), 1)
        self.assertTrue(result.validation_report.is_valid)

    def test_spare_destination_quota_lets_late_cargo_ride_into_next_day(self) -> None:
        """With headroom the mandatory Tır takes 17:00 cargo and arrives next day.

        This is the cargo that previously rode a paid spot vehicle departing at
        the same time and arriving at the same time.
        """

        result = GreedyMandatoryRentalFilledBuilder(
            _problem(
                (_demand("D1", 60.0, time(17)),),
                tir_travel_hours=8.0,
                truck_quota=10,
            ),
            make_policy(),
        ).build()

        loaded = _loaded_mandatory(result.plan.trips)
        self.assertTrue(loaded)
        leg = loaded[0].legs[0]
        self.assertGreater(leg.arrival_at.date(), leg.departure_at.date())
        self.assertEqual(_spot_trips(result.plan.trips), ())
        self.assertTrue(result.validation_report.is_valid)

    def test_mandatory_kamyon_does_not_consume_truck_quota(self) -> None:
        """Loaded Kamyon remains valid when every Tır quota is zero."""

        result = GreedyMandatoryRentalFilledBuilder(
            _problem(
                (_demand("D1", 60.0),),
                route_type="Kamyon",
                truck_quota=0,
            ),
            make_policy(),
        ).build()
        self.assertTrue(_loaded_mandatory(result.plan.trips))
        self.assertEqual(result.validation_report.metrics["truck_center_day_count"], 0)
        self.assertEqual(result.validation_report.metrics["peak_truck_quota_ratio"], 0.0)
        self.assertTrue(result.validation_report.is_valid)


def _mandatory_trips(trips: tuple) -> tuple:
    """Return mandatory trips in declared deterministic plan order."""

    return tuple(trip for trip in trips if trip.ownership_type == "mandatory_rented")


def _loaded_mandatory(trips: tuple) -> tuple:
    """Return mandatory trips carrying at least one demand assignment."""

    return tuple(
        trip for trip in _mandatory_trips(trips) if trip.legs[0].assignments
    )


def _spot_trips(trips: tuple) -> tuple:
    """Return remaining direct spot trips from a completed plan."""

    return tuple(trip for trip in trips if trip.ownership_type == "spot")


if __name__ == "__main__":
    unittest.main()
