"""State invariant verification gate for ALNS."""

from __future__ import annotations

import math
from typing import Set, Dict, Tuple
from src.domain import ProblemData, DemandPart
from src.evaluation import EvaluationPolicy
from src.alns.state import ALNSSolutionState


def validate_alns_state(
    state: ALNSSolutionState,
    problem: ProblemData,
    policy: EvaluationPolicy,
) -> bool:
    """Validates structural, logical, and physical invariants of an ALNSSolutionState.
    
    Returns True if state satisfies all invariants, False otherwise.
    Does NOT mutate state.
    """
    if not isinstance(state, ALNSSolutionState):
        return False

    demand_map = {d.demand_id: d for d in problem.demands}
    vehicle_types = {v.name: v for v in problem.static.vehicle_types}

    # 1. Identify parts carried by mandatory trips
    mandatory_parts: set[DemandPart] = set()
    parts_by_id = {p.part_id: p for p in state.parts}

    for trip in state.mandatory_trips:
        for leg in trip.legs:
            for assignment in leg.assignments:
                part = parts_by_id.get(assignment.demand_part_id)
                if part is not None:
                    mandatory_parts.add(part)

    # 2. Identify parts that must be transported by spot routes
    required_spot_parts = set(state.parts) - mandatory_parts

    # 3. Track pickups and dropoffs for each spot part across all spot routes
    pickup_locations: dict[DemandPart, list[tuple[int, int, str]]] = {}
    dropoff_locations: dict[DemandPart, list[tuple[int, int, str]]] = {}

    for r_idx, route in enumerate(state.spot_routes):
        v_info = vehicle_types.get(route.vehicle_type)
        if v_info is None:
            return False
        max_capacity = v_info.capacity_desi + policy.desi_tolerance

        onboard_desi = 0.0

        for s_idx, stop in enumerate(route.stops):
            # Check stop pre-computed desis
            expected_dropoff_desi = sum(p.desi for p in stop.dropoffs)
            expected_pickup_desi = sum(p.desi for p in stop.pickups)

            if abs(stop.dropoff_desi - expected_dropoff_desi) > 1e-4:
                return False
            if abs(stop.pickup_desi - expected_pickup_desi) > 1e-4:
                return False

            # Update onboard load: first dropoffs, then pickups
            onboard_desi -= expected_dropoff_desi
            if onboard_desi < -1e-4:  # Cannot drop off what isn't loaded
                return False

            onboard_desi += expected_pickup_desi
            if onboard_desi > max_capacity + 1e-4:  # Capacity overload
                return False

            for part in stop.dropoffs:
                dropoff_locations.setdefault(part, []).append((r_idx, s_idx, stop.center))

            for part in stop.pickups:
                pickup_locations.setdefault(part, []).append((r_idx, s_idx, stop.center))

        # Check route end load
        if abs(onboard_desi) > 1e-4:  # Must be 0 at end of route
            return False

    # 4. Verify each required spot part:
    routed_parts = set(pickup_locations.keys()) | set(dropoff_locations.keys())

    # Every required spot part must be routed
    if routed_parts != required_spot_parts:
        return False

    for part in required_spot_parts:
        pickups = pickup_locations.get(part, [])
        dropoffs = dropoff_locations.get(part, [])

        if len(pickups) != 1 or len(dropoffs) != 1:
            return False

        p_r_idx, p_s_idx, p_center = pickups[0]
        d_r_idx, d_s_idx, d_center = dropoffs[0]

        demand = demand_map.get(part.demand_id)
        if demand is None:
            return False

        if p_center != demand.origin:
            return False

        if d_center != demand.destination:
            return False

        # Must be on the exact same route
        if p_r_idx != d_r_idx:
            return False

        # Pickup stop index must be strictly before dropoff stop index
        if p_s_idx >= d_s_idx:
            return False

    return True


__all__ = ["validate_alns_state"]
