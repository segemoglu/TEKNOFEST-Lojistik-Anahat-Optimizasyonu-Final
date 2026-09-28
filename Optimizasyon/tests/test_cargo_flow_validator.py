"""Adversarial tests for atomic cargo flow and physical vehicle continuity.

The fixtures describe complete leg and handling timelines by hand.  They call
only :class:`TransportPlanValidator`; no route generator, optimizer or cost
calculator is used to manufacture expected validation outcomes.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import unittest

from src.domain import (
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    ProblemData,
    TransportPlan,
    VehicleLeg,
    VehicleTrip,
)
from src.evaluation import TransportPlanValidator

from tests.solution_fixtures import make_policy
from tests.test_adversarial_rules import make_three_center_problem


PART_ID = "D-AC-1"
DEMAND_ID = "D-AC"
PART_DESI = 40.0
HANDLING_MINUTES_PER_DESI = 0.01
TRAVEL_DURATION = timedelta(hours=1)


def error_codes(plan: TransportPlan, problem: ProblemData | None = None) -> set[str]:
    """Validate ``plan`` and return its structured error-code set.

    Args:
        plan: Synthetic candidate whose flow is under test.
        problem: Optional three-centre input; defaults to one 40-desi A-C demand.

    Returns:
        Error codes emitted by ``TransportPlanValidator``.  Warnings such as a
        missing reported cost are intentionally excluded.
    """

    active_problem = problem or make_three_center_problem(demand_desi=PART_DESI)
    report = TransportPlanValidator(active_problem, make_policy()).validate(plan)
    return {issue.code for issue in report.errors}


def make_cargo_leg(
    *,
    leg_id: str,
    vehicle_id: str,
    origin: str,
    destination: str,
    departure_at: datetime,
    loaded: bool,
    unloaded: bool,
    final: bool,
    desi: float = PART_DESI,
    load_start_at: datetime | None = None,
) -> tuple[VehicleLeg, tuple[HandlingOperation, ...]]:
    """Build one one-hour leg with exact 0.01 minute/desi handling.

    The optional ``load_start_at`` exposes transfer-boundary cases.  When it
    is absent, loading ends exactly at departure.  Unloading always starts at
    arrival, so every non-adversarial timeline is unambiguous.
    """

    duration = timedelta(minutes=desi * HANDLING_MINUTES_PER_DESI)
    operations: list[HandlingOperation] = []
    load_id: str | None = None
    unload_id: str | None = None
    if loaded:
        load_id = f"H-LOAD-{leg_id}"
        load_start = load_start_at or departure_at - duration
        operations.append(
            HandlingOperation(
                load_id,
                vehicle_id,
                origin,
                "load",
                load_start,
                load_start + duration,
                desi,
                (PART_ID,),
            )
        )
    arrival_at = departure_at + TRAVEL_DURATION
    if unloaded:
        unload_id = f"H-UNLOAD-{leg_id}"
        operations.append(
            HandlingOperation(
                unload_id,
                vehicle_id,
                destination,
                "unload",
                arrival_at,
                arrival_at + duration,
                desi,
                (PART_ID,),
            )
        )
    assignment = DemandAssignment(
        PART_ID,
        desi,
        loaded_at_origin=loaded,
        unloaded_at_destination=unloaded,
        is_final_delivery=final,
    )
    leg = VehicleLeg(
        leg_id,
        origin,
        destination,
        departure_at,
        arrival_at,
        (assignment,),
        load_id,
        unload_id,
    )
    return leg, tuple(operations)


def make_empty_leg(
    leg_id: str,
    origin: str,
    destination: str,
    departure_at: datetime,
) -> VehicleLeg:
    """Build an explicit one-hour empty repositioning or movement leg."""

    return VehicleLeg(
        leg_id,
        origin,
        destination,
        departure_at,
        departure_at + TRAVEL_DURATION,
    )


def make_trip(
    trip_id: str,
    vehicle_id: str,
    *bundles: tuple[VehicleLeg, tuple[HandlingOperation, ...]],
) -> VehicleTrip:
    """Combine leg/operation bundles into one documented spot-vehicle trip."""

    return VehicleTrip(
        trip_id,
        vehicle_id,
        "Tır",
        "spot",
        tuple(bundle[0] for bundle in bundles),
        tuple(operation for bundle in bundles for operation in bundle[1]),
    )


def make_plan(*trips: VehicleTrip) -> TransportPlan:
    """Wrap one 40-desi atomic part and the supplied trips as a plan."""

    return TransportPlan(
        demand_parts=(DemandPart(PART_ID, DEMAND_ID, PART_DESI),),
        trips=trips,
    )


class CargoFlowAdversarialTests(unittest.TestCase):
    """Reject cargo creation/loss while preserving valid direct and transfer flow."""

    def test_final_delivery_without_ever_loading_is_invalid(self) -> None:
        """A final unload flag cannot create cargo that was never loaded."""

        leg = make_cargo_leg(
            leg_id="L-AC",
            vehicle_id="V1",
            origin="A",
            destination="C",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=False,
            unloaded=True,
            final=True,
        )

        self.assertIn(
            "cargo_flow_discontinuity",
            error_codes(make_plan(make_trip("T1", "V1", leg))),
        )

    def test_cargo_disappearing_after_first_leg_is_invalid(self) -> None:
        """Cargo kept on board must occur on the vehicle's immediate next leg."""

        first = make_cargo_leg(
            leg_id="L-AB",
            vehicle_id="V1",
            origin="A",
            destination="B",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=True,
            unloaded=False,
            final=False,
        )
        second_leg = make_empty_leg(
            "L-BC",
            "B",
            "C",
            datetime(2026, 6, 29, 10, 1),
        )
        second = (second_leg, ())

        self.assertIn(
            "cargo_flow_discontinuity",
            error_codes(make_plan(make_trip("T1", "V1", first, second))),
        )

    def test_cargo_appearing_on_second_leg_is_invalid(self) -> None:
        """A part cannot first occur at B without a previous load at A."""

        first = (
            make_empty_leg("L-AB", "A", "B", datetime(2026, 6, 29, 9)),
            (),
        )
        second = make_cargo_leg(
            leg_id="L-BC",
            vehicle_id="V1",
            origin="B",
            destination="C",
            departure_at=datetime(2026, 6, 29, 10),
            loaded=False,
            unloaded=True,
            final=True,
        )
        codes = error_codes(make_plan(make_trip("T1", "V1", first, second)))

        self.assertIn("cargo_flow_discontinuity", codes)
        self.assertIn("cargo_teleportation", codes)

    def test_part_cannot_reload_without_intermediate_unload(self) -> None:
        """A load flag on B-C is invalid while the part remained on V1."""

        first = make_cargo_leg(
            leg_id="L-AB",
            vehicle_id="V1",
            origin="A",
            destination="B",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=True,
            unloaded=False,
            final=False,
        )
        second = make_cargo_leg(
            leg_id="L-BC",
            vehicle_id="V1",
            origin="B",
            destination="C",
            departure_at=datetime(2026, 6, 29, 10, 1, 24),
            loaded=True,
            unloaded=True,
            final=True,
        )

        self.assertIn(
            "cargo_flow_discontinuity",
            error_codes(make_plan(make_trip("T1", "V1", first, second))),
        )

    def test_loading_before_vehicle_arrival_is_invalid(self) -> None:
        """Intermediate loading cannot overlap the incoming vehicle movement."""

        problem = make_three_center_problem(demand_desi=PART_DESI)
        demand_from_b = replace(problem.demands[0], origin="B")
        problem = ProblemData(problem.static, (demand_from_b,))
        first = (
            make_empty_leg("L-AB", "A", "B", datetime(2026, 6, 29, 9)),
            (),
        )
        second = make_cargo_leg(
            leg_id="L-BC",
            vehicle_id="V1",
            origin="B",
            destination="C",
            departure_at=datetime(2026, 6, 29, 10),
            loaded=True,
            unloaded=True,
            final=True,
        )

        self.assertIn(
            "handling_before_vehicle_arrival",
            error_codes(make_plan(make_trip("T1", "V1", first, second)), problem),
        )

    def test_same_part_cannot_split_across_vehicles_without_child_ids(self) -> None:
        """Two 20-desi assignments cannot silently divide one 40-desi part."""

        departure = datetime(2026, 6, 29, 9, 1)
        first = make_cargo_leg(
            leg_id="L1-AC",
            vehicle_id="V1",
            origin="A",
            destination="C",
            departure_at=departure,
            loaded=True,
            unloaded=True,
            final=True,
            desi=20.0,
        )
        second = make_cargo_leg(
            leg_id="L2-AC",
            vehicle_id="V2",
            origin="A",
            destination="C",
            departure_at=departure,
            loaded=True,
            unloaded=True,
            final=True,
            desi=20.0,
        )
        plan = make_plan(make_trip("T1", "V1", first), make_trip("T2", "V2", second))

        self.assertIn("assignment_desi_mismatch", error_codes(plan))

    def test_valid_direct_a_to_c_delivery_passes(self) -> None:
        """One load, one movement and one final unload form a valid chain."""

        direct = make_cargo_leg(
            leg_id="L-AC",
            vehicle_id="V1",
            origin="A",
            destination="C",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=True,
            unloaded=True,
            final=True,
        )

        self.assertEqual(error_codes(make_plan(make_trip("T1", "V1", direct))), set())

    def test_valid_same_vehicle_continuation_without_reload_passes(self) -> None:
        """Cargo may remain on V1 over contiguous A-B and B-C legs."""

        first = make_cargo_leg(
            leg_id="L-AB",
            vehicle_id="V1",
            origin="A",
            destination="B",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=True,
            unloaded=False,
            final=False,
        )
        second = make_cargo_leg(
            leg_id="L-BC",
            vehicle_id="V1",
            origin="B",
            destination="C",
            departure_at=datetime(2026, 6, 29, 10, 1),
            loaded=False,
            unloaded=True,
            final=True,
        )

        self.assertEqual(
            error_codes(make_plan(make_trip("T1", "V1", first, second))),
            set(),
        )

    def test_valid_vehicle_transfer_passes(self) -> None:
        """A B transfer may reload onto V2 after V1's unload is complete."""

        plan = self._make_transfer_plan(wait=timedelta(minutes=5))

        self.assertEqual(error_codes(plan), set())

    def test_transfer_load_equal_to_unload_end_passes(self) -> None:
        """Zero waiting time at B is valid when unload end equals load start."""

        plan = self._make_transfer_plan(wait=timedelta(0))

        self.assertEqual(error_codes(plan), set())

    @staticmethod
    def _make_transfer_plan(*, wait: timedelta) -> TransportPlan:
        """Build an A-B unload followed by a B-C reload on another vehicle."""

        first = make_cargo_leg(
            leg_id="L-AB",
            vehicle_id="V1",
            origin="A",
            destination="B",
            departure_at=datetime(2026, 6, 29, 9, 1),
            loaded=True,
            unloaded=True,
            final=False,
        )
        first_unload = first[1][-1]
        load_start = first_unload.end_at + wait
        handling_duration = timedelta(
            minutes=PART_DESI * HANDLING_MINUTES_PER_DESI
        )
        second = make_cargo_leg(
            leg_id="L-BC",
            vehicle_id="V2",
            origin="B",
            destination="C",
            departure_at=load_start + handling_duration,
            loaded=True,
            unloaded=True,
            final=True,
            load_start_at=load_start,
        )
        return make_plan(
            make_trip("T1", "V1", first),
            make_trip("T2", "V2", second),
        )


class PhysicalVehicleLocationTests(unittest.TestCase):
    """Ensure idle time cannot teleport one physical vehicle between trips."""

    def test_vehicle_ending_at_b_cannot_restart_at_a(self) -> None:
        """An explicit B-A empty leg is required before another A departure."""

        problem = make_three_center_problem(demand_desi=PART_DESI)
        problem_without_demands = ProblemData(problem.static, ())
        first = make_trip(
            "T1",
            "V1",
            (make_empty_leg("L-AB", "A", "B", datetime(2026, 6, 29, 9)), ()),
        )
        second = make_trip(
            "T2",
            "V1",
            (make_empty_leg("L-AC", "A", "C", datetime(2026, 6, 29, 11)), ()),
        )
        plan = TransportPlan(demand_parts=(), trips=(first, second))

        self.assertIn(
            "physical_vehicle_location_discontinuity",
            error_codes(plan, problem_without_demands),
        )


if __name__ == "__main__":
    unittest.main()
