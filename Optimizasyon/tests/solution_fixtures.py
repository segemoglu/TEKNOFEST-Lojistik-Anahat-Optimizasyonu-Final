"""Small documented builders shared by cost and plan-validator tests."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

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
from src.evaluation import EvaluationPolicy


def make_demand(
    demand_id: str = "D1",
    *,
    demand_date: date = date(2026, 6, 29),
    demand_time: time = time(9),
    desi: float = 40.0,
) -> DemandInstance:
    """Create one A-to-B demand for a focused synthetic scenario."""

    return DemandInstance(
        demand_id=demand_id,
        demand_date=demand_date,
        time_slot=demand_time,
        origin="A",
        destination="B",
        desi=desi,
        split_required=desi > 100.0,
    )


def make_problem(
    *,
    demands: tuple[DemandInstance, ...] | None = None,
    vehicle_capacity: float = 100.0,
    handling_capacity: float = 10_000.0,
    truck_quota_a: int = 10,
    truck_quota_b: int = 10,
    mandatory_routes: tuple[MandatoryRentalRoute, ...] = (),
) -> ProblemData:
    """Build a two-centre problem with simple one-hour directed travel."""

    static = StaticProblemData(
        centers=(Center("A"), Center("B")),
        travel_arcs=(
            TravelArc(
                "A",
                "B",
                100.0,
                {
                    "Tır": 1.0,
                    "Kamyon": 1.0,
                    "Hafif Kamyon": 1.0,
                    "Kamyonet": 1.0,
                },
                1,
            ),
            TravelArc(
                "B",
                "A",
                100.0,
                {
                    "Tır": 1.0,
                    "Kamyon": 1.0,
                    "Hafif Kamyon": 1.0,
                    "Kamyonet": 1.0,
                },
                1,
            ),
        ),
        vehicle_types=(
            VehicleType("Tır", vehicle_capacity, 10.0, 2.0, 20.0, 3.0),
            VehicleType("Kamyon", vehicle_capacity, 8.0, 1.0, 16.0, 2.0),
            VehicleType("Hafif Kamyon", vehicle_capacity, 7.0, 1.0, 14.0, 2.0),
            VehicleType("Kamyonet", vehicle_capacity, 6.0, 1.0, 12.0, 2.0),
        ),
        mandatory_rental_routes=mandatory_routes,
        handling_capacities=(
            HandlingCapacity("A", handling_capacity),
            HandlingCapacity("B", handling_capacity),
        ),
        daily_truck_capacities=(
            DailyTruckCapacity("A", truck_quota_a),
            DailyTruckCapacity("B", truck_quota_b),
        ),
    )
    return ProblemData(
        static=static,
        demands=demands if demands is not None else (make_demand(),),
    )


def make_policy(
    *,
    granularity: str = "vehicle",
    start_date: date = date(2026, 6, 29),
    forecast_end_date: date | None = None,
) -> EvaluationPolicy:
    """Create an evaluation policy with a short synthetic schedule horizon."""

    return EvaluationPolicy(
        hourly_cost_granularity=granularity,
        mandatory_schedule_start_date=start_date,
        mandatory_forecast_end_date=forecast_end_date or start_date,
    )


def make_empty_trip(
    *,
    vehicle_id: str = "V1",
    trip_id: str = "T1",
    start_at: datetime = datetime(2026, 6, 29, 9),
    duration: timedelta = timedelta(hours=1),
    ownership_type: str = "spot",
    vehicle_type: str = "Tır",
) -> VehicleTrip:
    """Create a costable empty movement with no handling operations."""

    leg = VehicleLeg(
        leg_id=f"L-{trip_id}",
        origin="A",
        destination="B",
        departure_at=start_at,
        arrival_at=start_at + duration,
        assignments=(),
    )
    return VehicleTrip(
        trip_id=trip_id,
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        ownership_type=ownership_type,
        legs=(leg,),
    )


def make_delivery_trip(
    *,
    assignments: tuple[DemandAssignment, ...],
    vehicle_id: str = "V1",
    trip_id: str = "T1",
    start_at: datetime = datetime(2026, 6, 29, 9),
    ownership_type: str = "spot",
    vehicle_type: str = "Tır",
) -> VehicleTrip:
    """Create a chronologically valid one-hour delivery with exact handling."""

    loaded_desi = sum(
        assignment.desi
        for assignment in assignments
        if assignment.loaded_at_origin
    )
    unloaded_desi = sum(
        assignment.desi
        for assignment in assignments
        if assignment.unloaded_at_destination
    )
    load_duration = timedelta(minutes=loaded_desi * 0.01)
    unload_duration = timedelta(minutes=unloaded_desi * 0.01)
    load_end = start_at + load_duration
    arrival_at = load_end + timedelta(hours=1)
    unload_end = arrival_at + unload_duration
    loaded_parts = tuple(
        assignment.demand_part_id
        for assignment in assignments
        if assignment.loaded_at_origin
    )
    unloaded_parts = tuple(
        assignment.demand_part_id
        for assignment in assignments
        if assignment.unloaded_at_destination
    )

    load = HandlingOperation(
        operation_id=f"H-LOAD-{trip_id}",
        vehicle_id=vehicle_id,
        center="A",
        operation_type="load",
        start_at=start_at,
        end_at=load_end,
        desi=loaded_desi,
        demand_part_ids=loaded_parts,
    )
    unload = HandlingOperation(
        operation_id=f"H-UNLOAD-{trip_id}",
        vehicle_id=vehicle_id,
        center="B",
        operation_type="unload",
        start_at=arrival_at,
        end_at=unload_end,
        desi=unloaded_desi,
        demand_part_ids=unloaded_parts,
    )
    leg = VehicleLeg(
        leg_id=f"L-{trip_id}",
        origin="A",
        destination="B",
        departure_at=load_end,
        arrival_at=arrival_at,
        assignments=assignments,
        loading_operation_id=load.operation_id,
        unloading_operation_id=unload.operation_id,
    )
    return VehicleTrip(
        trip_id=trip_id,
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        ownership_type=ownership_type,
        legs=(leg,),
        handling_operations=(load, unload),
    )


def make_complete_plan(
    *,
    demand_id: str = "D1",
    delivered_desi: float = 40.0,
    part_desi: float = 40.0,
    start_at: datetime = datetime(2026, 6, 29, 9),
    vehicle_id: str = "V1",
    trip_id: str = "T1",
    vehicle_type: str = "Tır",
) -> TransportPlan:
    """Create a one-part plan whose final delivered amount is configurable."""

    part = DemandPart("P1", demand_id, part_desi)
    assignment = DemandAssignment(
        demand_part_id=part.part_id,
        desi=delivered_desi,
        is_final_delivery=True,
    )
    trip = make_delivery_trip(
        assignments=(assignment,),
        vehicle_id=vehicle_id,
        trip_id=trip_id,
        start_at=start_at,
        vehicle_type=vehicle_type,
    )
    return TransportPlan(demand_parts=(part,), trips=(trip,))


__all__ = [
    "make_complete_plan",
    "make_delivery_trip",
    "make_demand",
    "make_empty_trip",
    "make_policy",
    "make_problem",
]
