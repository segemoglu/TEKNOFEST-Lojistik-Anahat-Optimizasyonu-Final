from __future__ import annotations

from typing import List, Tuple
import math

from src.domain import DemandPart, ProblemData
from src.evaluation import EvaluationPolicy
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.exceptions import RepairInfeasibleError

from collections.abc import Iterable, Iterator

# How many days past its ready date an atom may be offered an existing route.
# The competition allows cargo to wait — SLA is 1 or 2 days and a late delivery
# costs 0,40 TL/desi/hour — so a thin lane can send one full vehicle every other
# day instead of a half-empty one daily.  Measured on this dataset, 630 lane-days
# on 2-day-SLA lanes average only 1.604 desi and could pair up for free.
# Two days is the longest SLA, so beyond that every atom is guaranteed a penalty.
DEFERRABLE_SERVICE_DAYS = 2

def generate_insertion_candidates(
    stops: tuple[ALNSRouteStop, ...], origin: str, destination: str, part: DemandPart
) -> Iterator[tuple[ALNSRouteStop, ...]]:
    # Check if origin/dest are already in route
    origin_indices = [i for i, s in enumerate(stops) if s.center == origin]
    
    # 1. Generate all ways to add Origin
    o_ways = []
    if origin_indices:
        for i in origin_indices:
            stop_o = stops[i]
            merged_stop = ALNSRouteStop(origin, stop_o.dropoffs, stop_o.pickups + (part,), stop_o.dropoff_desi, stop_o.pickup_desi + part.desi)
            new_stops = list(stops[:i]) + [merged_stop] + list(stops[i+1:])
            o_ways.append((new_stops, i))
    else:
        for i in range(len(stops) + 1):
            new_stops = list(stops[:i]) + [ALNSRouteStop(origin, (), (part,), 0.0, part.desi)] + list(stops[i:])
            o_ways.append((new_stops, i))
            
    # 2. Add Destination AFTER Origin
    for o_stops, o_idx in o_ways:
        dest_indices = [j for j in range(o_idx, len(o_stops)) if o_stops[j].center == destination]
        
        if dest_indices:
            for j in dest_indices:
                if j < o_idx: continue # Destination must be after or same as origin
                # If j == o_idx, it means we dropoff at the SAME stop we picked up! 
                # Wait, can we dropoff and pickup at the same stop? Yes, if origin == destination, but that shouldn't happen.
                if j == o_idx and origin != destination: continue
                
                stop_d = o_stops[j]
                merged_stop = ALNSRouteStop(destination, stop_d.dropoffs + (part,), stop_d.pickups, stop_d.dropoff_desi + part.desi, stop_d.pickup_desi)
                new_stops = list(o_stops[:j]) + [merged_stop] + list(o_stops[j+1:])
                yield tuple(new_stops)
        else:
            for j in range(o_idx + 1, len(o_stops) + 1):
                new_stops = list(o_stops[:j]) + [ALNSRouteStop(destination, (part,), (), part.desi, 0.0)] + list(o_stops[j:])
                yield tuple(new_stops)


def select_best_vehicle_for_route(
    stops: tuple[ALNSRouteStop, ...],
    service_date: datetime.date,
    proxy: ProxyEvaluator,
    policy: EvaluationPolicy,
) -> tuple[float, str | None]:
    """Select the lowest-cost vehicle for a sequence of stops that satisfies capacity constraints.
    
    Evaluates all suitable vehicle types where capacity >= max_load.
    Returns (best_cost, best_vehicle_name) or (inf, None) if no feasible vehicle exists.
    Tie-breaking: lowest cost first, then alphabetical vehicle name.
    """
    for i in range(1, len(stops)):
        if (stops[i-1].center, stops[i].center) not in proxy.arcs:
            return float('inf'), None

    max_load = 0.0
    current_load = 0.0
    for s in stops:
        current_load = current_load - s.dropoff_desi + s.pickup_desi
        if current_load > max_load:
            max_load = current_load

    vehicles_list = getattr(proxy, 'sorted_vehicles', None)
    if not isinstance(vehicles_list, (list, tuple)):
        vehicles = getattr(proxy, 'vehicles', {})
        if isinstance(vehicles, dict):
            vehicles_list = sorted(vehicles.values(), key=lambda x: getattr(x, 'name', str(x)))
        else:
            vehicles_list = []

    # Tır is the cheapest type on medium and long arcs, so without this screen
    # the sweep keeps selecting it for centres whose daily visit quota is zero
    # or fully consumed by mandatory rentals, and the exact audit discards the
    # whole candidate.
    truck_type = policy.truck_vehicle_type
    truck_allowed = proxy.can_use_truck(
        service_date,
        {stop.center for stop in stops},
    )

    suitable_vehicles = [
        v for v in vehicles_list
        if hasattr(v, 'capacity_desi')
        and v.capacity_desi + policy.desi_tolerance >= max_load
        and (truck_allowed or getattr(v, 'name', None) != truck_type)
    ]
    if not suitable_vehicles:
        return float('inf'), None

    best_cost = float('inf')
    best_vehicle_name = None

    for v in suitable_vehicles:
        v_name = getattr(v, 'name', str(v))
        cost = proxy.evaluate_route(service_date, stops, v_name)
        if not math.isinf(cost):
            if cost < best_cost:
                best_cost = cost
                best_vehicle_name = v_name
            elif cost == best_cost and (best_vehicle_name is None or v_name < best_vehicle_name):
                best_vehicle_name = v_name

    return best_cost, best_vehicle_name


def score_insertion_candidates(
    candidates: Iterable[tuple[ALNSRouteStop, ...]], 
    service_date: datetime.date, 
    proxy: ProxyEvaluator,
    policy: EvaluationPolicy
) -> tuple[float, ALNSSpotRoute | None]:
    best_cost = float('inf')
    best_route = None
    
    for stops in candidates:
        cost, vehicle_name = select_best_vehicle_for_route(stops, service_date, proxy, policy)
        if not math.isinf(cost) and cost < best_cost:
            best_cost = cost
            best_route = ALNSSpotRoute(service_date, stops, vehicle_name)
                
    return best_cost, best_route

from src.alns.route_index import RouteIndex

def greedy_insertion(
    state: ALNSSolutionState,
    unassigned: list[tuple[DemandPart, ...]],
    problem: ProblemData,
    policy: EvaluationPolicy,
    proxy: ProxyEvaluator,
) -> ALNSSolutionState:
    """Insert unassigned bins by scoring all candidate positions with ProxyEvaluator."""
    import datetime
    spot_routes = list(state.spot_routes)
    
    unassigned_parts = []
    for bin_parts in unassigned:
        unassigned_parts.extend(bin_parts)
    unassigned_parts.sort(key=lambda p: (p.desi, p.demand_id, p.part_id), reverse=True)
    
    # Initialize spatial-temporal index for spot routes
    index = RouteIndex(spot_routes)

    for part in unassigned_parts:
        demand = proxy.demands[part.demand_id]
        service_date = demand.demand_date
        
        best_cost_delta = float('inf')
        best_route_idx = -1
        best_new_route = None

        origin = demand.origin
        destination = demand.destination
        
        # 1. Try to insert into an existing route
        candidate_indices = index.get_candidate_indices(
            service_date, origin, destination, DEFERRABLE_SERVICE_DAYS
        )
        for route_idx in candidate_indices:
            route = spot_routes[route_idx]

            old_cost = proxy.evaluate_route(
                route.service_date, route.stops, route.vehicle_type
            )
            if math.isinf(old_cost):
                continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            # Score against the route's own day, not the atom's.  They differ as
            # soon as deferral is allowed, and rebuilding the route on the atom's
            # date would silently move every other atom already on it.
            new_cost, new_route = score_insertion_candidates(
                candidates, route.service_date, proxy, policy
            )
            
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_route_idx = route_idx
                    best_new_route = new_route
                
        # 2. Try creating a new direct route
        new_stop_tuple = (
            ALNSRouteStop(origin, (), (part,)),
            ALNSRouteStop(destination, (part,), ())
        )
        direct_cost, direct_route = score_insertion_candidates([new_stop_tuple], service_date, proxy, policy)
        
        if direct_cost < best_cost_delta:
            best_cost_delta = direct_cost
            best_route_idx = -1
            best_new_route = direct_route
            
        # 3. Apply the best insertion and update index
        if best_route_idx >= 0:
            spot_routes[best_route_idx] = best_new_route
            index.update_route(best_route_idx, best_new_route)
        elif best_new_route is not None and not math.isinf(best_cost_delta):
            spot_routes.append(best_new_route)
            index.append_route(best_new_route)
        else:
            raise RepairInfeasibleError(f"No feasible insertion for part {part.part_id}")
                
    # Final sweep to upgrade/downgrade vehicle types
    for i, route in enumerate(spot_routes):
        _, best_v_name = select_best_vehicle_for_route(route.stops, route.service_date, proxy, policy)
        if best_v_name and best_v_name != route.vehicle_type:
            spot_routes[i] = ALNSSpotRoute(route.service_date, route.stops, best_v_name)
            
    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(spot_routes))


def regret_2_insertion(
    state: ALNSSolutionState,
    unassigned: list[tuple[DemandPart, ...]],
    problem: ProblemData,
    policy: EvaluationPolicy,
    proxy: 'ProxyEvaluator',
) -> ALNSSolutionState:
    """Insert unassigned bins using a faster Static Regret-2 heuristic."""
    import datetime
    spot_routes = list(state.spot_routes)
    
    unassigned_parts = []
    for bin_parts in unassigned:
        unassigned_parts.extend(bin_parts)
    
    part_regrets = []
    
    # Initialize spatial-temporal index for spot routes
    index = RouteIndex(spot_routes)

    for part in unassigned_parts:
        demand = proxy.demands[part.demand_id]
        origin = demand.origin
        destination = demand.destination
        service_date = demand.demand_date
        
        best_cost = float('inf')
        second_best_cost = float('inf')
        
        # 1. Try existing routes
        candidate_indices = index.get_candidate_indices(
            service_date, origin, destination, DEFERRABLE_SERVICE_DAYS
        )
        for r_idx in candidate_indices:
            route = spot_routes[r_idx]

            old_cost = proxy.evaluate_route(
                route.service_date, route.stops, route.vehicle_type
            )
            if math.isinf(old_cost):
                continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            new_cost, _ = score_insertion_candidates(
                candidates, route.service_date, proxy, policy
            )
            
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost:
                    second_best_cost = best_cost
                    best_cost = delta
                elif delta < second_best_cost:
                    second_best_cost = delta
        
        # 2. Try new direct route
        new_stop_tuple = (
            ALNSRouteStop(origin, (), (part,)),
            ALNSRouteStop(destination, (part,), ())
        )
        direct_cost, _ = select_best_vehicle_for_route(new_stop_tuple, service_date, proxy, policy)
        if not math.isinf(direct_cost):
            if direct_cost < best_cost:
                second_best_cost = best_cost
                best_cost = direct_cost
            elif direct_cost < second_best_cost:
                second_best_cost = direct_cost
            
        regret = second_best_cost - best_cost if not math.isinf(second_best_cost) else float('inf')
        part_regrets.append((regret, part))
        
    part_regrets.sort(key=lambda x: (x[0], x[1].demand_id, x[1].part_id), reverse=True)
    
    for _, part in part_regrets:
        demand = proxy.demands[part.demand_id]
        origin = demand.origin
        destination = demand.destination
        service_date = demand.demand_date
        
        best_cost_delta = float('inf')
        best_route_idx = -1
        best_new_route = None
        
        # 1. Try existing routes
        candidate_indices = index.get_candidate_indices(
            service_date, origin, destination, DEFERRABLE_SERVICE_DAYS
        )
        for i in candidate_indices:
            route = spot_routes[i]

            old_cost = proxy.evaluate_route(
                route.service_date, route.stops, route.vehicle_type
            )
            if math.isinf(old_cost):
                continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            new_cost, new_route = score_insertion_candidates(
                candidates, route.service_date, proxy, policy
            )
            
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_route_idx = i
                    best_new_route = new_route
                    
        # 2. Try direct route in placement phase
        new_stop_tuple = (
            ALNSRouteStop(origin, (), (part,)),
            ALNSRouteStop(destination, (part,), ())
        )
        direct_cost, direct_route = score_insertion_candidates([new_stop_tuple], service_date, proxy, policy)
        if direct_cost < best_cost_delta:
            best_cost_delta = direct_cost
            best_route_idx = -1
            best_new_route = direct_route

        if best_route_idx >= 0:
            spot_routes[best_route_idx] = best_new_route
            index.update_route(best_route_idx, best_new_route)
        elif best_new_route is not None and not math.isinf(best_cost_delta):
            spot_routes.append(best_new_route)
            index.append_route(best_new_route)
        else:
            raise RepairInfeasibleError(f"No feasible insertion for part {part.part_id}")
                
    # Final sweep to upgrade/downgrade vehicle types
    for i, route in enumerate(spot_routes):
        _, best_v_name = select_best_vehicle_for_route(route.stops, route.service_date, proxy, policy)
        if best_v_name and best_v_name != route.vehicle_type:
            spot_routes[i] = ALNSSpotRoute(route.service_date, route.stops, best_v_name)
            
    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(spot_routes))

__all__ = ["greedy_insertion", "regret_2_insertion", "select_best_vehicle_for_route"]
