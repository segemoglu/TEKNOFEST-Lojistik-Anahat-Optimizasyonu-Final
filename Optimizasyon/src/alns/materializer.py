"""Materializer to convert ALNSSolutionState back into an immutable TransportPlan.

This module provides the ALNSMaterializer class, which takes a lightweight state
and reconstructs the full VehicleTrip and HandlingOperation objects deterministically,
ready for CostCalculator and TransportPlanValidator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, time
import math

from src.domain import (
    ProblemData,
    TransportPlan,
    VehicleTrip,
    HandlingOperation,
    DemandAssignment,
    VehicleLeg,
)
from src.evaluation import EvaluationPolicy
from src.planning.handling_ledger import HandlingCapacityLedger
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop


class ALNSMaterializer:
    """Deterministically convert ALNS states into costable, valid TransportPlans."""

    def __init__(self, problem: ProblemData, policy: EvaluationPolicy) -> None:
        self.problem = problem
        self.policy = policy
        self._demands = {demand.demand_id: demand for demand in problem.demands}
        self._arcs = {
            (arc.origin, arc.destination): arc for arc in problem.static.travel_arcs
        }
        self._vehicle_types = {v.name: v for v in problem.static.vehicle_types}

    def materialize(self, state: ALNSSolutionState) -> TransportPlan:
        """Create a full TransportPlan from the state."""
        ledger = self._create_ledger()

        # 1. Register mandatory operations into the ledger.  These are fixed and
        # already scheduled, so they are recorded through the ledger's public
        # API rather than reserved: refusing or deferring them is not an option.
        for trip in state.mandatory_trips:
            for op in trip.handling_operations:
                ledger.record_completed(
                    op.center,
                    op.start_at,
                    op.end_at,
                    op.desi,
                )

        # 2. Re-schedule spot routes deterministically
        spot_trips = []
        for index, route in enumerate(state.spot_routes, start=1):
            trip = self._materialize_route(index, route, ledger)
            spot_trips.append(trip)

        # 3. Combine, extending the fixed schedule if the search pushed the plan
        # past the last day 6E had covered.
        continuation = self._mandatory_continuation(
            state.mandatory_trips,
            tuple(spot_trips),
        )
        all_trips = state.mandatory_trips + continuation + tuple(spot_trips)
        return TransportPlan(state.parts, all_trips)

    def _mandatory_continuation(
        self,
        mandatory_trips: tuple[VehicleTrip, ...],
        spot_trips: tuple[VehicleTrip, ...],
    ) -> tuple[VehicleTrip, ...]:
        """Add empty rentals for days the re-scheduled plan newly reaches.

        ``state.mandatory_trips`` is frozen at whatever 6E produced, but ALNS
        re-schedules every spot route and may finish later than 6E did.  The
        validator requires the fixed fleet to run on **every** day up to the
        plan's completion, so without this the candidate is rejected with
        ``mandatory_vehicle_missing`` even though it is otherwise legal.  That
        rejection is invisible while routes stay short and becomes dominant as
        soon as they span extra days — it discarded 712 of 747 audited
        candidates.

        The added vehicles are empty and cost real money, so a candidate that
        buys an extra day is now correctly priced rather than silently dropped.
        """

        completion_at = max(
            (
                operation.end_at
                for trip in mandatory_trips + spot_trips
                for operation in trip.handling_operations
                if operation.operation_type == "unload"
            ),
            default=None,
        )
        if completion_at is None:
            return ()
        covered_until = max(
            (leg.departure_at.date() for trip in mandatory_trips for leg in trip.legs),
            default=None,
        )
        if covered_until is None:
            return ()
        routes = sorted(
            self.problem.static.mandatory_rental_routes,
            key=lambda route: (route.origin, route.destination, route.vehicle_type),
        )
        trips: list[VehicleTrip] = []
        service_date = covered_until + timedelta(days=1)
        while service_date <= completion_at.date():
            departure_at = datetime.combine(service_date, time.min)
            for route_index, route in enumerate(routes, start=1):
                arc = self._arcs[(route.origin, route.destination)]
                arrival_at = departure_at + timedelta(
                    hours=arc.travel_hours_for(route.vehicle_type)
                )
                for sequence in range(1, route.vehicle_count + 1):
                    stem = (
                        f"ALNS-MAND-{service_date:%Y%m%d}"
                        f"-{route_index:02d}-{sequence:02d}"
                    )
                    trips.append(
                        VehicleTrip(
                            f"{stem}-TRIP",
                            stem,
                            route.vehicle_type,
                            "mandatory_rented",
                            (
                                VehicleLeg(
                                    f"{stem}-LEG",
                                    route.origin,
                                    route.destination,
                                    departure_at,
                                    arrival_at,
                                ),
                            ),
                        )
                    )
            service_date += timedelta(days=1)
        return tuple(trips)

    def _create_ledger(self) -> HandlingCapacityLedger:
        """Create a clean ledger based on problem constraints."""
        return HandlingCapacityLedger(
            self.problem.static.handling_capacities,
            self.policy.handling_minutes_per_desi,
            self.policy.desi_tolerance,
        )

    def _materialize_route(
        self, route_index: int, route: ALNSSpotRoute, ledger: HandlingCapacityLedger
    ) -> VehicleTrip:
        """Schedule one spot route sequentially and return a VehicleTrip."""
        stem = f"ALNS-SPOT-{route_index:05d}"
        
        current_time = None
        
        stop_unloads = []
        stop_loads = []
        stop_arrivals = []
        
        for stop_idx, stop in enumerate(route.stops):
            center = stop.center
            
            # Unload phase
            unload_op = None
            if stop.dropoffs:
                unload_desi = stop.dropoff_desi
                unload_interval = ledger.schedule(center, current_time, unload_desi)
                current_time = unload_interval[1]
                unload_op = HandlingOperation(
                    f"{stem}-UNLOAD-{stop_idx}",
                    stem,
                    center,
                    "unload",
                    unload_interval[0],
                    unload_interval[1],
                    unload_desi,
                    tuple(p.part_id for p in stop.dropoffs),
                )
            stop_unloads.append(unload_op)
            
            # Load phase
            load_op = None
            if stop.pickups:
                ready_at = max(
                    datetime.combine(self._demands[p.demand_id].demand_date, self._demands[p.demand_id].time_slot)
                    for p in stop.pickups
                )
                arrival_for_load = current_time if current_time is not None else ready_at
                arrival_for_load = max(arrival_for_load, ready_at)
                load_desi = stop.pickup_desi
                
                if stop_idx == 0 and len(route.stops) > 1:
                    deadline = datetime.combine(route.service_date + timedelta(days=1), time.min)
                    load_interval = ledger.try_schedule_before(
                        center, arrival_for_load, load_desi, deadline
                    )
                    if not load_interval:
                        # Fallback to standard scheduling which allows next-day carry-over, ensuring incumbent is always valid
                        load_interval = ledger.schedule(center, arrival_for_load, load_desi)
                else:
                    load_interval = ledger.schedule(center, arrival_for_load, load_desi)
                    
                current_time = load_interval[1]
                load_op = HandlingOperation(
                    f"{stem}-LOAD-{stop_idx}",
                    stem,
                    center,
                    "load",
                    load_interval[0],
                    load_interval[1],
                    load_desi,
                    tuple(p.part_id for p in stop.pickups),
                )
            stop_loads.append(load_op)
            
            # Travel to next stop
            if stop_idx < len(route.stops) - 1:
                next_stop = route.stops[stop_idx + 1]
                arc = self._arcs[(center, next_stop.center)]
                arrival_time = current_time + timedelta(hours=arc.travel_hours_for(route.vehicle_type))
                stop_arrivals.append(arrival_time)
                current_time = arrival_time
                
        # Create legs
        legs = []
        on_board = []
        for i in range(len(route.stops) - 1):
            stop = route.stops[i]
            next_stop = route.stops[i+1]
            
            for p in stop.dropoffs:
                if p in on_board:
                    on_board.remove(p)
            for p in stop.pickups:
                on_board.append(p)
                
            assignments = []
            for p in on_board:
                is_loaded_here = p in stop.pickups
                is_unloaded_next = p in next_stop.dropoffs
                assignments.append(
                    DemandAssignment(
                        p.part_id,
                        p.desi,
                        loaded_at_origin=is_loaded_here,
                        unloaded_at_destination=is_unloaded_next,
                        is_final_delivery=is_unloaded_next
                    )
                )
                
            load_op_id = stop_loads[i].operation_id if stop_loads[i] else None
            unload_op_id = stop_unloads[i+1].operation_id if stop_unloads[i+1] else None
            
            if stop_loads[i]:
                departure = stop_loads[i].end_at
            elif stop_unloads[i]:
                departure = stop_unloads[i].end_at
            elif i == 0:
                departure = datetime.combine(route.service_date, time.min)
            else:
                departure = stop_arrivals[i-1]
                
            legs.append(
                VehicleLeg(
                    f"{stem}-LEG-{i + 1}",
                    stop.center,
                    next_stop.center,
                    departure,
                    stop_arrivals[i],
                    tuple(assignments),
                    load_op_id,
                    unload_op_id
                )
            )
            
        all_ops = tuple(op for op in stop_unloads + stop_loads if op is not None)
        all_ops = tuple(sorted(all_ops, key=lambda o: o.start_at))
        
        return VehicleTrip(
            f"{stem}-TRIP",
            stem,
            route.vehicle_type,
            "spot",
            tuple(legs),
            all_ops
        )

    def extract_state(self, plan: TransportPlan) -> ALNSSolutionState:
        """Extract an ALNSSolutionState from an existing TransportPlan."""
        mandatory_trips = tuple(
            trip for trip in plan.trips if trip.ownership_type == "mandatory_rented"
        )
        spot_trips = tuple(
            trip for trip in plan.trips if trip.ownership_type == "spot"
        )

        parts_by_id = {part.part_id: part for part in plan.demand_parts}
        spot_routes = []

        for trip in spot_trips:
            service_date = trip.legs[0].departure_at.date()
            vehicle_type = trip.vehicle_type
            
            centers = [trip.legs[0].origin] + [leg.destination for leg in trip.legs]
            
            pickups = {c: [] for c in range(len(centers))}
            dropoffs = {c: [] for c in range(len(centers))}
            
            for i, leg in enumerate(trip.legs):
                for a in leg.assignments:
                    p = parts_by_id[a.demand_part_id]
                    if a.loaded_at_origin:
                        pickups[i].append(p)
                    if a.unloaded_at_destination:
                        dropoffs[i+1].append(p)
                        
            stops = []
            for i, center in enumerate(centers):
                stops.append(ALNSRouteStop(center, tuple(dropoffs[i]), tuple(pickups[i])))
                
            spot_routes.append(ALNSSpotRoute(service_date, tuple(stops), vehicle_type))

        return ALNSSolutionState(plan.demand_parts, mandatory_trips, tuple(spot_routes))

__all__ = ["ALNSMaterializer"]
