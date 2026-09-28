"""ALNS destroy operators."""

from __future__ import annotations

import random
from typing import Tuple, List

from src.domain import DemandPart
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop


def random_removal(
    state: ALNSSolutionState,
    n: int,
    rng: random.Random,
) -> tuple[ALNSSolutionState, list[tuple[DemandPart, ...]]]:
    """Remove n random parts from the state and return them as unassigned bins."""
    spot_routes = list(state.spot_routes)
    unassigned: list[tuple[DemandPart, ...]] = []
    
    all_parts = []
    for r_idx, route in enumerate(spot_routes):
        for stop in route.stops:
            for part in stop.pickups:
                all_parts.append((r_idx, part))
                
    if n > len(all_parts):
        n = len(all_parts)
        
    to_remove = set(rng.sample(all_parts, n))
    parts_to_remove = {part for _, part in to_remove}
    
    new_spot_routes = []
    for r_idx, route in enumerate(spot_routes):
        new_stops = []
        for stop in route.stops:
            new_dropoffs = tuple(p for p in stop.dropoffs if p not in parts_to_remove)
            new_pickups = tuple(p for p in stop.pickups if p not in parts_to_remove)
            if new_dropoffs or new_pickups:
                new_stops.append(ALNSRouteStop(stop.center, new_dropoffs, new_pickups))
                
        if new_stops:
            new_spot_routes.append(
                ALNSSpotRoute(route.service_date, tuple(new_stops), route.vehicle_type)
            )
            
    for part in parts_to_remove:
        unassigned.append((part,))
        
    unassigned.sort(key=lambda x: (x[0].demand_id, x[0].part_id))
            
    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(new_spot_routes)), unassigned


def worst_removal(
    state: ALNSSolutionState,
    n: int,
    rng: random.Random,
    problem: 'ProblemData',
    policy: 'EvaluationPolicy',
    proxy: 'ProxyEvaluator',
) -> tuple[ALNSSolutionState, list[tuple[DemandPart, ...]]]:
    """Remove parts that contribute the most to the route cost (worst removal)."""
    spot_routes = list(state.spot_routes)
    unassigned: list[tuple[DemandPart, ...]] = []
    
    part_savings = []
    for r_idx, route in enumerate(spot_routes):
        old_cost = proxy.evaluate_route(route.service_date, route.stops, route.vehicle_type)
        
        route_parts = []
        for stop in route.stops:
            route_parts.extend(stop.pickups)
            
        if len(route_parts) == 1:
            part_savings.append((old_cost, r_idx, route_parts[0]))
        else:
            for part in route_parts:
                new_stops = []
                for stop in route.stops:
                    new_dropoffs = tuple(p for p in stop.dropoffs if p != part)
                    new_pickups = tuple(p for p in stop.pickups if p != part)
                    if new_dropoffs or new_pickups:
                        new_stops.append(ALNSRouteStop(stop.center, new_dropoffs, new_pickups))
                if not new_stops:
                    saving = old_cost
                else:
                    new_cost = proxy.evaluate_route(route.service_date, tuple(new_stops), route.vehicle_type)
                    saving = old_cost - new_cost
                part_savings.append((saving, r_idx, part))
                
    part_savings.sort(key=lambda x: (x[0], x[2].demand_id, x[2].part_id), reverse=True)
    if n > len(part_savings):
        n = len(part_savings)
        
    to_remove = set()
    p = 3.0
    while len(to_remove) < n and part_savings:
        idx = int((rng.random() ** p) * len(part_savings))
        saving, r_idx, part = part_savings.pop(idx)
        to_remove.add(part)
        
    new_spot_routes = []
    for r_idx, route in enumerate(spot_routes):
        new_stops = []
        for stop in route.stops:
            new_dropoffs = tuple(p for p in stop.dropoffs if p not in to_remove)
            new_pickups = tuple(p for p in stop.pickups if p not in to_remove)
            if new_dropoffs or new_pickups:
                new_stops.append(ALNSRouteStop(stop.center, new_dropoffs, new_pickups))
        if new_stops:
            new_spot_routes.append(
                ALNSSpotRoute(route.service_date, tuple(new_stops), route.vehicle_type)
            )
            
    for part in to_remove:
        unassigned.append((part,))
        
    unassigned.sort(key=lambda x: (x[0].demand_id, x[0].part_id))
        
    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(new_spot_routes)), unassigned

def shaw_removal(
    state: ALNSSolutionState,
    n: int,
    rng: random.Random,
    problem: 'ProblemData',
    policy: 'EvaluationPolicy',
    proxy: 'ProxyEvaluator',
    sample_size: int = 64,
) -> tuple[ALNSSolutionState, list[tuple[DemandPart, ...]]]:
    """Remove parts that are related (similar) to each other (Shaw removal with bounded random sampling)."""
    spot_routes = list(state.spot_routes)
    unassigned: list[tuple[DemandPart, ...]] = []
    
    all_parts = []
    for r_idx, route in enumerate(spot_routes):
        for stop in route.stops:
            for part in stop.pickups:
                all_parts.append((route, part))
                
    if n >= len(all_parts):
        for _, part in all_parts:
            unassigned.append((part,))
        unassigned.sort(key=lambda x: (x[0].demand_id, x[0].part_id))
        return ALNSSolutionState(state.parts, state.mandatory_trips, ()), unassigned
        
    if not all_parts:
        return state, unassigned
        
    seed = rng.choice(all_parts)
    to_remove_set = {seed[1]}
    removed_list = [seed]
    remaining = [p for p in all_parts if p[1] not in to_remove_set]
    
    p = 3.0
    while len(to_remove_set) < n and remaining:
        i_node = rng.choice(removed_list)
        route_i, part_i = i_node
        demand_i = proxy.demands[part_i.demand_id]
        
        k = min(sample_size, len(remaining))
        candidate_sample = rng.sample(remaining, k)
        
        relatedness_scores = []
        for j_node in candidate_sample:
            route_j, part_j = j_node
            demand_j = proxy.demands[part_j.demand_id]
            
            arc = proxy.arcs.get((demand_i.destination, demand_j.destination))
            dist = arc.distance_km if arc else 10000.0
            
            origin_penalty = 0.0 if demand_i.origin == demand_j.origin else 1000.0
            date_penalty = 0.0 if route_i.service_date == route_j.service_date else 1000.0
            
            r_score = dist + origin_penalty + date_penalty
            relatedness_scores.append((r_score, j_node))
            
        relatedness_scores.sort(key=lambda x: (x[0], x[1][1].demand_id, x[1][1].part_id))
        idx = int((rng.random() ** p) * len(relatedness_scores))
        _, selected_node = relatedness_scores[idx]
        
        to_remove_set.add(selected_node[1])
        removed_list.append(selected_node)
        remaining.remove(selected_node)
        
    new_spot_routes = []
    for r_idx, route in enumerate(spot_routes):
        new_stops = []
        for stop in route.stops:
            new_dropoffs = tuple(p for p in stop.dropoffs if p not in to_remove_set)
            new_pickups = tuple(p for p in stop.pickups if p not in to_remove_set)
            if new_dropoffs or new_pickups:
                new_stops.append(ALNSRouteStop(stop.center, new_dropoffs, new_pickups))
        if new_stops:
            new_spot_routes.append(
                ALNSSpotRoute(route.service_date, tuple(new_stops), route.vehicle_type)
            )
            
    for part in to_remove_set:
        unassigned.append((part,))
        
    unassigned.sort(key=lambda x: (x[0].demand_id, x[0].part_id))
        
    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(new_spot_routes)), unassigned

__all__ = ["random_removal", "worst_removal", "shaw_removal"]
