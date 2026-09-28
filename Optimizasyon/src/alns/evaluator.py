from __future__ import annotations

import math
from collections import OrderedDict
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.alns.state import ALNSSolutionState, ALNSRouteStop
from src.alns.invariants import validate_alns_state

class ProxyEvaluator:
    """O(1) / O(stops) approximation of route costs without full materialization.
    
    Caches route evaluations using memoization. Used to rapidly score greedy insertions
    and drive the simulated annealing acceptance loop.
    """
    
    def __init__(
        self, 
        problem: ProblemData, 
        policy: EvaluationPolicy, 
        initial_state: ALNSSolutionState,
        cache_maxsize: int = 100_000,
    ):
        self.problem = problem
        self.policy = policy
        self.cache_maxsize = cache_maxsize
        
        self.vehicles = {v.name: v for v in problem.static.vehicle_types}
        
        unique_types = {}
        for v in self.vehicles.values():
            if v.name not in unique_types or unique_types[v.name].capacity_desi > v.capacity_desi:
                unique_types[v.name] = v
        self.sorted_vehicles = sorted(unique_types.values(), key=lambda v: (v.capacity_desi, v.name))
        
        self.arcs = {(a.origin, a.destination): a for a in problem.static.travel_arcs}
        self.demands = {d.demand_id: d for d in problem.demands}
        
        # Precompute datetimes to avoid massive overhead in hot loops
        self.part_ready_times = {}
        self.part_deadlines = {}
        for d in problem.demands:
            ready_time = datetime.combine(d.demand_date, d.time_slot)
            self.part_ready_times[d.demand_id] = ready_time.timestamp()
            
            arc = self.arcs.get((d.origin, d.destination))
            if arc:
                deadline = ready_time + timedelta(days=arc.target_delivery_days)
            else:
                deadline = ready_time
            self.part_deadlines[d.demand_id] = deadline.timestamp()
                
        # Setup Capacities and base usage
        self.handling_capacities = {
            hc.center: hc.max_daily_desi
            for hc in problem.static.handling_capacities
        }
        self.truck_capacities = {
            c.center: c.max_daily_truck_visits
            for c in problem.static.daily_truck_capacities
        }
        
        self.truck_vehicle_type = policy.truck_vehicle_type
        self.mandatory_truck_usage = {}
        self.mandatory_handling_usage = {}

        for trip in initial_state.mandatory_trips:
            if trip.vehicle_type == policy.truck_vehicle_type:
                for leg in trip.legs:
                    dep_key = (leg.origin, leg.departure_at.date())
                    arr_key = (leg.destination, leg.arrival_at.date())
                    self.mandatory_truck_usage[dep_key] = self.mandatory_truck_usage.get(dep_key, 0) + 1
                    self.mandatory_truck_usage[arr_key] = self.mandatory_truck_usage.get(arr_key, 0) + 1
            
            for op in trip.handling_operations:
                hk = (op.center, op.start_at.date())
                self.mandatory_handling_usage[hk] = self.mandatory_handling_usage.get(hk, 0.0) + op.desi
                
        self._route_cost_cache: OrderedDict[Tuple, float] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _make_cache_key(
        self,
        service_date: date,
        stops: tuple[ALNSRouteStop, ...],
        vehicle_name: str,
    ) -> tuple:
        """Create a lightweight, hashable cache key composed strictly of primitive values."""
        return (service_date.isoformat(), vehicle_name, tuple(s.stop_key for s in stops))

    def _cache_put(self, key: tuple, value: float) -> float:
        """Store evaluated cost in LRU cache, guaranteeing len(cache) <= cache_maxsize."""
        if key in self._route_cost_cache:
            self._route_cost_cache.move_to_end(key)
        self._route_cost_cache[key] = value
        if len(self._route_cost_cache) > self.cache_maxsize:
            self._route_cost_cache.popitem(last=False)
        return value

    def clear_cache(self) -> None:
        """Drop every memoized route cost.

        The cache holds up to ``cache_maxsize`` entries and is worthless once a
        search finishes, so a caller running several searches in one process
        releases it here instead of carrying the memory into the next run.
        """

        self._route_cost_cache.clear()

    def cache_info(self) -> dict[str, int]:
        """Return cache statistics for diagnostics and testing."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "currsize": len(self._route_cost_cache),
            "maxsize": self.cache_maxsize,
        }

    def evaluate_route(
        self, 
        service_date: date, 
        stops: tuple[ALNSRouteStop, ...], 
        vehicle_name: str
    ) -> float:
        """Returns the approximate cost in TL of a single spot route."""
        cache_key = self._make_cache_key(service_date, stops, vehicle_name)
        
        if cache_key in self._route_cost_cache:
            self._route_cost_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._route_cost_cache[cache_key]
            
        self._cache_misses += 1
        vehicle = self.vehicles[vehicle_name]
        
        current_load = 0.0
        total_distance = 0.0
        sla_penalty = 0.0
        
        current_time_sec = None
        
        for idx, stop in enumerate(stops):
            unload_desi = stop.dropoff_desi
            load_desi = stop.pickup_desi
            
            if current_load - unload_desi + load_desi > vehicle.capacity_desi + self.policy.desi_tolerance:
                return self._cache_put(cache_key, float('inf'))
                
            # Travel
            if idx > 0:
                prev_stop = stops[idx - 1]
                arc = self.arcs.get((prev_stop.center, stop.center))
                if not arc:
                    return self._cache_put(cache_key, float('inf'))  # No path
                total_distance += arc.distance_km
                current_time_sec += arc.travel_hours_for(vehicle_name) * 3600.0
                
            # Unload handling
            if unload_desi > 0:
                current_time_sec += unload_desi * self.policy.handling_minutes_per_desi * 60.0
                for part in stop.dropoffs:
                    deadline_sec = self.part_deadlines.get(part.demand_id)
                    if not deadline_sec:
                        return self._cache_put(cache_key, float('inf'))
                    if current_time_sec > deadline_sec:
                        delay_minutes = (current_time_sec - deadline_sec) / 60.0
                        delay_hours_ceil = math.ceil(delay_minutes / 60.0)
                        sla_penalty += delay_hours_ceil * part.desi * self.policy.sla_penalty_per_desi_hour_tl

            # Load waiting and handling
            if stop.pickups:
                max_ready_at = max(self.part_ready_times[p.demand_id] for p in stop.pickups)
                
                if current_time_sec is None:
                    current_time_sec = max_ready_at
                elif current_time_sec < max_ready_at:
                    current_time_sec = max_ready_at
                    
                current_time_sec += load_desi * self.policy.handling_minutes_per_desi * 60.0
            
            if current_time_sec is None and not stop.pickups:
                current_time_sec = datetime.combine(service_date, datetime.min.time()).timestamp()
                
            current_load = current_load - unload_desi + load_desi
            
        if current_time_sec is None:
            return self._cache_put(cache_key, float('inf'))

        first_ready_time = max(self.part_ready_times[p.demand_id] for p in stops[0].pickups) if stops[0].pickups else datetime.combine(service_date, datetime.min.time()).timestamp()
        
        total_hours = (current_time_sec - first_ready_time) / 3600.0
        billable_hours = math.ceil(total_hours)
        
        cost = (
            billable_hours * vehicle.spot_hourly_cost_tl 
            + total_distance * vehicle.spot_cost_per_km_tl
            + sla_penalty
        )
        
        return self._cache_put(cache_key, cost)

    def can_use_truck(self, service_date: date, centers) -> bool:
        """Report whether a Tır could visit every centre on this service date.

        Args:
            service_date: The route's calendar day.
            centers: Centres the route would stop at.

        Returns:
            ``True`` only when every centre still has at least one Tır visit
            free after the fixed mandatory schedule is subtracted.

        Why this exists:
            ``repair.select_best_vehicle_for_route`` picks the cheapest vehicle
            type, and Tır wins on medium and long arcs.  Without this screen the
            repair operators keep proposing Tır routes into centres whose daily
            quota is zero or already fully consumed by mandatory rentals, and
            every such candidate is thrown away by the exact audit.  Only
            mandatory usage is subtracted here, so the screen is optimistic and
            cheap; :meth:`state_capacity_violation` applies the exact check.
        """

        for center in centers:
            quota = self.truck_capacities.get(center)
            if quota is None:
                return False
            if quota - self.mandatory_truck_usage.get((center, service_date), 0) < 1:
                return False
        return True

    def stop_dates(
        self,
        service_date: date,
        stops: tuple[ALNSRouteStop, ...],
        vehicle_name: str,
    ) -> tuple[date, ...]:
        """Return the calendar day the vehicle is at each stop.

        The walk mirrors :meth:`evaluate_route` exactly — wait for cargo to be
        ready, handle, travel — so a route that spans midnight reports the days
        it really touches.  Charging every stop to ``service_date`` instead was
        wrong for any route longer than a day and made the search propose
        candidates the validator then threw out for
        ``daily_truck_quota_exceeded``.
        """

        current_time_sec: float | None = None
        result: list[date] = []
        for index, stop in enumerate(stops):
            if index > 0:
                arc = self.arcs.get((stops[index - 1].center, stop.center))
                if arc is None or current_time_sec is None:
                    result.append(service_date)
                    continue
                current_time_sec += arc.travel_hours_for(vehicle_name) * 3600.0
            if stop.pickups:
                ready = max(self.part_ready_times[p.demand_id] for p in stop.pickups)
                current_time_sec = (
                    ready if current_time_sec is None
                    else max(current_time_sec, ready)
                )
            if current_time_sec is None:
                current_time_sec = datetime.combine(
                    service_date, datetime.min.time()
                ).timestamp()
            result.append(datetime.fromtimestamp(current_time_sec).date())
            current_time_sec += (
                (stop.dropoff_desi + stop.pickup_desi)
                * self.policy.handling_minutes_per_desi
                * 60.0
            )
        return tuple(result)

    def truck_quota_violation(self, state: ALNSSolutionState) -> str | None:
        """Return the daily Tır quota a whole state would break, or ``None``.

        The per-route cost model cannot see this constraint because the quota is
        shared across every route touching a centre on a given day.  Mandatory
        trips are counted first: their schedule is fixed and already saturates
        several centres (İstanbul, Yalova, Tekirdağ, Balıkesir).

        Each spot route stop counts as one visit, matching the validator's rule
        that a continuous unload-and-reload at one centre is a single visit and
        that leaving and returning starts a new one.  Timing is approximated by
        the route's ``service_date`` because the lightweight state carries no
        per-stop timestamps; ``TransportPlanValidator`` re-checks the exact
        per-day placement during the exact audit.

        Handling capacity is deliberately **not** checked here.  Which calendar
        day absorbs a handling operation is decided by
        ``HandlingCapacityLedger`` during materialization, and that ledger never
        reserves onto a full day — it defers to the next one.  A materialized
        plan therefore cannot exceed handling capacity, while judging it from
        ``service_date`` alone would falsely reject valid states: measured on the
        6E plan that approximation reported 16 over-capacity centre-days even
        though the validator measured peak utilisation at 99,77%.
        """

        truck_usage = dict(self.mandatory_truck_usage)
        for route in state.spot_routes:
            if route.vehicle_type != self.truck_vehicle_type:
                continue
            days = self.stop_dates(
                route.service_date, route.stops, route.vehicle_type
            )
            for stop, day in zip(route.stops, days, strict=True):
                quota = self.truck_capacities.get(stop.center)
                if quota is None:
                    return "unknown_truck_capacity_center"
                key = (stop.center, day)
                visits = truck_usage.get(key, 0) + 1
                if visits > quota:
                    return "daily_truck_quota_exceeded"
                truck_usage[key] = visits
        return None

    def evaluate_state(self, state: ALNSSolutionState) -> float:
        """Sums the proxy costs of all spot routes in the state."""
        if not validate_alns_state(state, self.problem, self.policy):
            return float('inf')
        if self.truck_quota_violation(state) is not None:
            return float('inf')

        total = 0.0
        for route in state.spot_routes:
            cost = self.evaluate_route(
                route.service_date,
                route.stops,
                route.vehicle_type
            )
            if math.isinf(cost):
                return float('inf')
            total += cost

        return total

__all__ = ["ProxyEvaluator"]
