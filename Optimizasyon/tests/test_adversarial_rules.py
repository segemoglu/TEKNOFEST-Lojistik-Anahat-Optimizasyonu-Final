"""Golden and adversarial checks for the audited competition rules A--F.

Every TL expectation in this module is a literal obtained by hand from the
fixture rates; no validator expectation is produced by ``CostCalculator``.
The scenarios exercise costing and validation only and never construct routes
algorithmically or invoke an optimizer.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta
import unittest

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandAssignment,
    DemandInstance,
    DemandPart,
    HandlingCapacity,
    HandlingOperation,
    MandatoryRentalRoute,
    ProblemData,
    StaticProblemData,
    TransportPlan,
    TravelArc,
    VehicleLeg,
    VehicleTrip,
    VehicleType,
)
from src.evaluation import CostCalculator, TransportPlanValidator

from tests.solution_fixtures import (
    make_delivery_trip,
    make_demand,
    make_empty_trip,
    make_policy,
    make_problem,
)


def error_codes(report) -> set[str]:
    """Return validator error codes for focused adversarial assertions."""

    return {issue.code for issue in report.errors}


def make_three_center_problem(
    *,
    demand_desi: float = 100.0,
    vehicle_capacity_desi: float = 20_000.0,
    handling_capacity_desi: float = 50_000.0,
) -> ProblemData:
    """Create a hand-auditable A-B-C network for intermediate-stop tests."""

    travel_times = {
        "Tır": 1.0,
        "Kamyon": 1.0,
        "Hafif Kamyon": 1.0,
        "Kamyonet": 1.0,
    }
    arcs = tuple(
        TravelArc(origin, destination, 100.0, travel_times, 1)
        for origin, destination in (
            ("A", "B"),
            ("B", "A"),
            ("B", "C"),
            ("C", "B"),
            ("A", "C"),
            ("C", "A"),
        )
    )
    vehicle_types = (
        VehicleType("Tır", vehicle_capacity_desi, 10.0, 2.0, 20.0, 3.0),
        VehicleType("Kamyon", vehicle_capacity_desi, 8.0, 1.0, 16.0, 2.0),
        VehicleType("Hafif Kamyon", vehicle_capacity_desi, 7.0, 1.0, 14.0, 2.0),
        VehicleType("Kamyonet", vehicle_capacity_desi, 6.0, 1.0, 12.0, 2.0),
    )
    static = StaticProblemData(
        centers=(Center("A"), Center("B"), Center("C")),
        travel_arcs=arcs,
        vehicle_types=vehicle_types,
        mandatory_rental_routes=(),
        handling_capacities=tuple(
            HandlingCapacity(center, handling_capacity_desi)
            for center in ("A", "B", "C")
        ),
        daily_truck_capacities=tuple(
            DailyTruckCapacity(center, 10) for center in ("A", "B", "C")
        ),
    )
    demand = DemandInstance(
        demand_id="D-AC",
        demand_date=date(2026, 6, 29),
        time_slot=time(9),
        origin="A",
        destination="C",
        desi=demand_desi,
    )
    return ProblemData(static=static, demands=(demand,))


def _make_intermediate_operations(
    *, overlap_reload: bool = False
) -> tuple[HandlingOperation, ...]:
    """Create the four exact one-minute operations for an A-B-C transfer."""

    load_a = HandlingOperation(
        "H-A-LOAD",
        "TIR-1",
        "A",
        "load",
        datetime(2026, 6, 29, 9, 0),
        datetime(2026, 6, 29, 9, 1),
        100.0,
        ("P-AC",),
    )
    unload_b = HandlingOperation(
        "H-B-UNLOAD",
        "TIR-1",
        "B",
        "unload",
        datetime(2026, 6, 29, 10, 1),
        datetime(2026, 6, 29, 10, 2),
        100.0,
        ("P-AC",),
    )
    reload_start = unload_b.end_at
    if overlap_reload:
        reload_start -= timedelta(seconds=1)
    load_b = HandlingOperation(
        "H-B-LOAD",
        "TIR-1",
        "B",
        "load",
        reload_start,
        reload_start + timedelta(minutes=1),
        100.0,
        ("P-AC",),
    )
    unload_c_start = load_b.end_at + timedelta(hours=1)
    unload_c = HandlingOperation(
        "H-C-UNLOAD",
        "TIR-1",
        "C",
        "unload",
        unload_c_start,
        unload_c_start + timedelta(minutes=1),
        100.0,
        ("P-AC",),
    )
    return load_a, unload_b, load_b, unload_c


def make_intermediate_plan(*, overlap_reload: bool = False) -> TransportPlan:
    """Create one A-B-C Tır with unload then reload at B.

    At 100 desi every handling operation lasts exactly 1 minute.  The valid
    plan starts B loading exactly when B unloading ends; the adversarial form
    starts it one second early.
    """

    part = DemandPart("P-AC", "D-AC", 100.0)
    first_assignment = DemandAssignment(
        "P-AC",
        100.0,
        loaded_at_origin=True,
        unloaded_at_destination=True,
        is_final_delivery=False,
    )
    final_assignment = DemandAssignment(
        "P-AC",
        100.0,
        loaded_at_origin=True,
        unloaded_at_destination=True,
        is_final_delivery=True,
    )
    load_a, unload_b, load_b, unload_c = _make_intermediate_operations(
        overlap_reload=overlap_reload
    )
    legs = (
        VehicleLeg(
            "L-AB",
            "A",
            "B",
            load_a.end_at,
            unload_b.start_at,
            (first_assignment,),
            load_a.operation_id,
            unload_b.operation_id,
        ),
        VehicleLeg(
            "L-BC",
            "B",
            "C",
            load_b.end_at,
            unload_c.start_at,
            (final_assignment,),
            load_b.operation_id,
            unload_c.operation_id,
        ),
    )
    trip = VehicleTrip(
        "TRIP-ABC",
        "TIR-1",
        "Tır",
        "spot",
        legs,
        (load_a, unload_b, load_b, unload_c),
    )
    return TransportPlan(demand_parts=(part,), trips=(trip,))


class HandlingGoldenTests(unittest.TestCase):
    """Audit proportional midnight capacity and grouped handling semantics."""

    def test_midnight_handling_splits_exactly_3000_and_7000_desi(self) -> None:
        """The 30/70-minute golden interval must allocate 3,000/7,000 desi."""

        validator = TransportPlanValidator(make_problem(demands=()), make_policy())
        operation = HandlingOperation(
            "H-MIDNIGHT",
            "V1",
            "A",
            "load",
            datetime(2026, 6, 29, 23, 30),
            datetime(2026, 6, 30, 1, 10),
            10_000.0,
        )

        allocation = validator._allocate_handling_by_day(operation)

        self.assertEqual(allocation[date(2026, 6, 29)], 3_000.0)
        self.assertEqual(allocation[date(2026, 6, 30)], 7_000.0)

    def test_grouped_shipments_share_one_operation_and_total_duration(self) -> None:
        """Two shipment parts use one 100-desi, one-minute operation window."""

        demands = (make_demand("D1", desi=40), make_demand("D2", desi=60))
        problem = make_problem(demands=demands)
        parts = (DemandPart("P1", "D1", 40), DemandPart("P2", "D2", 60))
        assignments = (
            DemandAssignment("P1", 40, is_final_delivery=True),
            DemandAssignment("P2", 60, is_final_delivery=True),
        )
        trip = make_delivery_trip(assignments=assignments)
        plan = TransportPlan(demand_parts=parts, trips=(trip,))

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        load = trip.handling_operations[0]
        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(load.demand_part_ids, ("P1", "P2"))
        self.assertEqual(load.end_at - load.start_at, timedelta(minutes=1))

    def test_intermediate_unload_and_reload_count_separately_without_wait(self) -> None:
        """B consumes 100 + 100 desi and equality of operation times is valid."""

        problem = make_three_center_problem()
        plan = make_intermediate_plan()

        report = TransportPlanValidator(problem, make_policy()).validate(plan)
        cost = CostCalculator(problem, make_policy()).calculate(plan)

        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(
            report.metrics["handling_usage_desi_by_center_day"]["B|2026-06-29"],
            200.0,
        )
        self.assertEqual(len(cost.vehicle_cost_lines), 1)
        self.assertEqual(cost.total_cost_tl, 660.0)

    def test_reload_starting_before_same_vehicle_unload_finishes_is_error(self) -> None:
        """One-second self-overlap at an intermediate center must be rejected."""

        report = TransportPlanValidator(
            make_three_center_problem(),
            make_policy(),
        ).validate(make_intermediate_plan(overlap_reload=True))

        self.assertIn("reload_started_before_unload_finished", error_codes(report))


class TruckQuotaGoldenTests(unittest.TestCase):
    """Audit vehicle-type filtering and discrete physical Tır visits."""

    def test_only_tir_consumes_quota(self) -> None:
        """Kamyon, Hafif Kamyon and Kamyonet add zero daily Tır visits."""

        problem = make_problem(
            demands=(),
            truck_quota_a=1,
            truck_quota_b=1,
        )
        trips = tuple(
            make_empty_trip(
                vehicle_id=f"V-{vehicle_type}",
                trip_id=f"T-{index}",
                vehicle_type=vehicle_type,
            )
            for index, vehicle_type in enumerate(
                ("Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"),
                start=1,
            )
        )

        report = TransportPlanValidator(problem, make_policy()).validate(
            TransportPlan(demand_parts=(), trips=trips)
        )

        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(
            report.metrics["truck_visits_by_center_day"],
            {"A|2026-06-29": 1, "B|2026-06-29": 1},
        )

    def test_departure_and_arrival_consume_their_own_calendar_dates(self) -> None:
        """A 29 June departure and 30 June arrival create two dated events."""

        validator = TransportPlanValidator(make_problem(demands=()), make_policy())
        leg = VehicleLeg(
            "L-CROSS",
            "A",
            "B",
            datetime(2026, 6, 29, 23, 30),
            datetime(2026, 6, 30, 0, 30),
        )

        events = validator._truck_visit_events((leg,))

        self.assertEqual(
            events,
            [("A", date(2026, 6, 29)), ("B", date(2026, 6, 30))],
        )

    def test_continuous_stop_counts_once_and_return_is_a_new_visit(self) -> None:
        """A-B-A-B produces two A visits and two B visits, not six events."""

        validator = TransportPlanValidator(make_problem(demands=()), make_policy())
        legs = (
            VehicleLeg(
                "L1",
                "A",
                "B",
                datetime(2026, 6, 29, 9),
                datetime(2026, 6, 29, 10),
            ),
            VehicleLeg(
                "L2",
                "B",
                "A",
                datetime(2026, 6, 29, 10, 30),
                datetime(2026, 6, 29, 11, 30),
            ),
            VehicleLeg(
                "L3",
                "A",
                "B",
                datetime(2026, 6, 29, 12),
                datetime(2026, 6, 29, 13),
            ),
        )

        events = validator._truck_visit_events(legs)

        self.assertEqual(Counter(center for center, _ in events), {"A": 2, "B": 2})

    def test_trip_boundary_does_not_double_count_stationary_tir(self) -> None:
        """Arrival then later departure under one vehicle ID is one B visit."""

        problem = make_problem(demands=(), truck_quota_a=2, truck_quota_b=1)
        trips = (
            make_empty_trip(vehicle_id="V1", trip_id="T1"),
            VehicleTrip(
                "T2",
                "V1",
                "Tır",
                "spot",
                (
                    VehicleLeg(
                        "L-T2",
                        "B",
                        "A",
                        datetime(2026, 6, 29, 10, 30),
                        datetime(2026, 6, 29, 11, 30),
                    ),
                ),
            ),
        )

        report = TransportPlanValidator(problem, make_policy()).validate(
            TransportPlan(demand_parts=(), trips=trips)
        )

        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(
            report.metrics["truck_visits_by_center_day"]["B|2026-06-29"],
            1,
        )

    def test_midnight_handling_never_fractionally_splits_truck_visit(self) -> None:
        """A continuous B visit from 23:30 to 01:10 remains one integer event."""

        validator = TransportPlanValidator(make_problem(demands=()), make_policy())
        legs = (
            VehicleLeg(
                "L1",
                "A",
                "B",
                datetime(2026, 6, 29, 22, 30),
                datetime(2026, 6, 29, 23, 30),
            ),
            VehicleLeg(
                "L2",
                "B",
                "A",
                datetime(2026, 6, 30, 1, 10),
                datetime(2026, 6, 30, 2, 10),
            ),
        )

        events = validator._truck_visit_events(legs)

        self.assertEqual(events.count(("B", date(2026, 6, 29))), 1)
        self.assertNotIn(("B", date(2026, 6, 30)), events)


class CostAndSlaGoldenTests(unittest.TestCase):
    """Cross-check physical vehicle and SLA formulas with literal TL totals."""

    def test_one_minute_late_costs_exactly_forty_tl_for_100_desi(self) -> None:
        """100 desi × 1 rounded hour × 0.4 TL equals 40 TL."""

        demand = make_demand(desi=100.0)
        problem = make_problem(demands=(demand,), vehicle_capacity=100.0)
        part = DemandPart("P1", "D1", 100.0)
        trip = make_delivery_trip(
            assignments=(DemandAssignment("P1", 100.0, is_final_delivery=True),),
            start_at=datetime(2026, 6, 30, 7, 59),
        )
        plan = TransportPlan(demand_parts=(part,), trips=(trip,))

        calculated = CostCalculator(problem, make_policy()).calculate(plan)
        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertEqual(calculated.sla_penalty_tl, 40.0)
        self.assertEqual(calculated.total_cost_tl, 380.0)
        self.assertEqual(report.metrics["computed_sla_penalty_tl"], 40.0)
        self.assertEqual(report.metrics["computed_total_cost_tl"], 380.0)

    def test_split_parts_use_independent_delay_rounding(self) -> None:
        """40 desi at +1 min and 60 desi at +61 min cost 16 + 48 = 64 TL."""

        demand = make_demand(desi=100.0)
        problem = make_problem(demands=(demand,), vehicle_capacity=100.0)
        parts = (DemandPart("P1", "D1", 40.0), DemandPart("P2", "D1", 60.0))
        trips = (
            make_delivery_trip(
                assignments=(DemandAssignment("P1", 40.0, is_final_delivery=True),),
                vehicle_id="V1",
                trip_id="T1",
                start_at=datetime(2026, 6, 30, 8, 0, 12),
            ),
            make_delivery_trip(
                assignments=(DemandAssignment("P2", 60.0, is_final_delivery=True),),
                vehicle_id="V2",
                trip_id="T2",
                start_at=datetime(2026, 6, 30, 8, 59, 48),
            ),
        )
        plan = TransportPlan(demand_parts=parts, trips=trips)

        calculated = CostCalculator(problem, make_policy()).calculate(plan)
        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertEqual(calculated.sla_penalty_tl, 64.0)
        self.assertEqual(calculated.total_cost_tl, 744.0)
        self.assertEqual(report.metrics["computed_sla_penalty_tl"], 64.0)


class MandatoryRentalAdversarialTests(unittest.TestCase):
    """Audit fixed route, empty cost, Tır quota and daily identity rules."""

    def test_empty_mandatory_tir_still_consumes_zero_center_quota(self) -> None:
        """A mandatory Tır departure is a visit even with zero carried desi."""

        problem = make_problem(demands=(), truck_quota_a=0)
        trip = make_empty_trip(ownership_type="mandatory_rented")
        report = TransportPlanValidator(problem, make_policy()).validate(
            TransportPlan(demand_parts=(), trips=(trip,))
        )

        self.assertIn("daily_truck_quota_exceeded", error_codes(report))
        self.assertEqual(
            report.metrics["truck_visits_by_center_day"]["A|2026-06-29"],
            1,
        )

    def test_daily_mandatory_schedule_applies_even_when_demand_is_empty(self) -> None:
        """An empty demand instance does not cancel the forecast-day fleet."""

        route = MandatoryRentalRoute("A", "B", "Tır", 1)
        problem = make_problem(demands=(), mandatory_routes=(route,))

        report = TransportPlanValidator(problem, make_policy()).validate(
            TransportPlan(demand_parts=(), trips=())
        )

        self.assertIn("mandatory_vehicle_missing", error_codes(report))

    def test_mandatory_vehicle_cannot_have_stop_or_return_leg(self) -> None:
        """A two-leg mandatory trip violates its single fixed O-D movement."""

        route = MandatoryRentalRoute("A", "B", "Tır", 1)
        problem = make_problem(mandatory_routes=(route,))
        delivery = make_delivery_trip(
            assignments=(DemandAssignment("P1", 40.0, is_final_delivery=True),),
            vehicle_id="V-SPOT",
            trip_id="T-SPOT",
        )
        mandatory = VehicleTrip(
            "T-MAND",
            "V-MAND",
            "Tır",
            "mandatory_rented",
            (
                VehicleLeg(
                    "L-M1",
                    "A",
                    "B",
                    datetime(2026, 6, 29, 7),
                    datetime(2026, 6, 29, 8),
                ),
                VehicleLeg(
                    "L-M2",
                    "B",
                    "A",
                    datetime(2026, 6, 29, 8),
                    datetime(2026, 6, 29, 9),
                ),
            ),
        )
        plan = TransportPlan(
            demand_parts=(DemandPart("P1", "D1", 40.0),),
            trips=(mandatory, delivery),
        )

        report = TransportPlanValidator(problem, make_policy()).validate(plan)

        self.assertIn("mandatory_route_mismatch", error_codes(report))

    def test_reused_mandatory_id_across_days_is_structured_error(self) -> None:
        """Two daily mandatory sefers may not be merged under one vehicle ID."""

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
        report = TransportPlanValidator(
            make_problem(demands=()),
            make_policy(),
        ).validate(TransportPlan(demand_parts=(), trips=trips))

        self.assertIn("mandatory_vehicle_id_reused", error_codes(report))


if __name__ == "__main__":
    unittest.main()
