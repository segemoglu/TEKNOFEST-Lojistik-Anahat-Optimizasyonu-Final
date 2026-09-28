"""Unit tests for RouteIndex spatial-temporal lookup in ALNS repair operators."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime

from src.alns.route_index import RouteIndex
from src.alns.repair import greedy_insertion, regret_2_insertion
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.evaluator import validate_alns_state
from src.domain import DemandPart


def make_100_routes(service_date: datetime.date) -> list[ALNSSpotRoute]:
    routes = []
    # 3 target routes containing CenterA or CenterB
    routes.append(ALNSSpotRoute(service_date, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("TargetX", (), ())), "Kamyonet"))
    routes.append(ALNSSpotRoute(service_date, (ALNSRouteStop("TargetY", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet"))
    routes.append(ALNSSpotRoute(service_date, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet"))

    # 97 unrelated routes
    for i in range(97):
        routes.append(
            ALNSSpotRoute(
                service_date,
                (ALNSRouteStop(f"UnrelatedO_{i}", (), ()), ALNSRouteStop(f"UnrelatedD_{i}", (), ())),
                "Kamyonet",
            )
        )
    return routes


class RepairRouteIndexTests(unittest.TestCase):
    """Test suite for RouteIndex spatial-temporal lookups and repair operator integration."""

    def test_route_index_filters_unrelated_routes(self):
        """Out of 100 routes, RouteIndex must return only the 3 routes matching origin/destination/date."""
        sdate = datetime.date(2026, 7, 1)
        routes = make_100_routes(sdate)

        index = RouteIndex(routes)
        candidate_indices = index.get_candidate_indices(sdate, "CenterA", "CenterB")

        total_routes = len(routes)
        evaluated_routes = len(candidate_indices)

        print("\n--- Route Index Evaluation Metrics ---")
        print(f"Total Routes in State       : {total_routes}")
        print(f"Evaluated Candidate Routes  : {evaluated_routes}")
        print(f"Bypassed Unrelated Routes   : {total_routes - evaluated_routes}\n")

        self.assertEqual(total_routes, 100)
        self.assertEqual(evaluated_routes, 3)
        self.assertEqual(candidate_indices, [0, 1, 2])

    def test_route_index_matches_brute_force_reference(self):
        """Candidate indices from RouteIndex must match brute-force prefilter linear scan identically."""
        sdate = datetime.date(2026, 7, 1)
        routes = make_100_routes(sdate)
        index = RouteIndex(routes)

        origin, destination = "CenterA", "CenterB"

        brute_force_indices = [
            i
            for i, r in enumerate(routes)
            if r.service_date == sdate
            and (
                any(s.center == origin for s in r.stops)
                or any(s.center == destination for s in r.stops)
            )
        ]

        indexed_indices = index.get_candidate_indices(sdate, origin, destination)
        self.assertEqual(indexed_indices, brute_force_indices)

    def test_route_index_updates_on_route_replacement(self):
        """Updating a route removes old center keys and indexes new center keys correctly."""
        sdate = datetime.date(2026, 7, 1)
        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet")
        r1 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterX", (), ()), ALNSRouteStop("CenterY", (), ())), "Kamyonet")

        index = RouteIndex([r0, r1])

        # Replace route 0 with new centers CenterC -> CenterD
        new_r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterC", (), ()), ALNSRouteStop("CenterD", (), ())), "Kamyonet")
        index.update_route(0, new_r0)

        # Old centers CenterA/CenterB must no longer return route 0
        self.assertEqual(index.get_candidate_indices(sdate, "CenterA", "CenterB"), [])

        # New centers CenterC/CenterD must return route 0
        self.assertEqual(index.get_candidate_indices(sdate, "CenterC", "CenterD"), [0])

    def test_route_index_appends_new_direct_route(self):
        """Appending a new direct route adds it to the index at the next available index."""
        sdate = datetime.date(2026, 7, 1)
        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet")
        index = RouteIndex([r0])

        new_route = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterNewO", (), ()), ALNSRouteStop("CenterNewD", (), ())), "Kamyonet")
        assigned_idx = index.append_route(new_route)

        self.assertEqual(assigned_idx, 1)
        self.assertEqual(index.get_candidate_indices(sdate, "CenterNewO", "CenterNewD"), [1])

    def test_greedy_and_regret2_return_invariant_valid_state(self):
        """greedy_insertion and regret_2_insertion with RouteIndex return valid states."""
        sdate = datetime.date(2026, 7, 1)
        p1 = DemandPart(part_id="P1", demand_id="D1", desi=10.0)
        p2 = DemandPart(part_id="P2", demand_id="D2", desi=10.0)

        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("OriginA", (), ()), ALNSRouteStop("DestB", (), ())), "Kamyonet")
        state = ALNSSolutionState(parts=(p1, p2), mandatory_trips=(), spot_routes=(r0,))
        unassigned = [(p1,), (p2,)]

        d1 = MagicMock(demand_id="D1", origin="OriginA", destination="DestB", demand_date=sdate)
        d2 = MagicMock(demand_id="D2", origin="OriginA", destination="DestB", demand_date=sdate)

        proxy = MagicMock()
        proxy.demands = {"D1": d1, "D2": d2}
        proxy.arcs = {("OriginA", "DestB"): 10.0}
        v = MagicMock(capacity_desi=1000.0)
        v.name = "Kamyonet"
        proxy.vehicles = {"Kamyonet": v}
        proxy.evaluate_route.return_value = 100.0

        problem = MagicMock()
        problem.demands = [d1, d2]
        problem.static.vehicle_types = [v]
        policy = MagicMock(desi_tolerance=0.0)

        res_greedy = greedy_insertion(state, unassigned, problem, policy, proxy)
        self.assertTrue(validate_alns_state(res_greedy, problem, policy))

        res_regret = regret_2_insertion(state, unassigned, problem, policy, proxy)
        self.assertTrue(validate_alns_state(res_regret, problem, policy))

    def test_deterministic_results(self):
        """Repeated calls to greedy and regret-2 with RouteIndex produce identical states."""
        sdate = datetime.date(2026, 7, 1)
        p1 = DemandPart(part_id="P1", demand_id="D1", desi=10.0)
        p2 = DemandPart(part_id="P2", demand_id="D2", desi=10.0)

        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("OriginA", (), ()), ALNSRouteStop("DestB", (), ())), "Kamyonet")
        state = ALNSSolutionState(parts=(p1, p2), mandatory_trips=(), spot_routes=(r0,))
        unassigned = [(p1,), (p2,)]

        d1 = MagicMock(origin="OriginA", destination="DestB", demand_date=sdate)
        d2 = MagicMock(origin="OriginA", destination="DestB", demand_date=sdate)

        proxy = MagicMock()
        proxy.demands = {"D1": d1, "D2": d2}
        proxy.arcs = {("OriginA", "DestB"): 10.0}
        v = MagicMock(name="Kamyonet", capacity_desi=1000.0)
        proxy.vehicles = {"Kamyonet": v}
        proxy.evaluate_route.return_value = 100.0

        problem = MagicMock()
        policy = MagicMock(desi_tolerance=0.0)

        g1 = greedy_insertion(state, unassigned, problem, policy, proxy)
        g2 = greedy_insertion(state, unassigned, problem, policy, proxy)
        self.assertEqual(g1.canonical_tuple(), g2.canonical_tuple())

        r1 = regret_2_insertion(state, unassigned, problem, policy, proxy)
        r2 = regret_2_insertion(state, unassigned, problem, policy, proxy)
        self.assertEqual(r1.canonical_tuple(), r2.canonical_tuple())

    def test_indexed_vs_brute_force_greedy_insertion_update_and_append(self):
        """Indexed greedy_insertion matches brute-force linear-scan on route updates and appends."""
        sdate = datetime.date(2026, 7, 1)

        p1 = DemandPart(part_id="P1", demand_id="D1", desi=10.0)
        p2 = DemandPart(part_id="P2", demand_id="D2", desi=10.0)
        p3 = DemandPart(part_id="P3", demand_id="D3", desi=10.0)
        p4 = DemandPart(part_id="P4", demand_id="D4", desi=10.0)

        # Initial route r0: CenterA -> CenterB
        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet")
        initial_state = ALNSSolutionState(parts=(p1, p2, p3, p4), mandatory_trips=(), spot_routes=(r0,))
        unassigned = [(p1,), (p2,), (p3,), (p4,)]

        d1 = MagicMock(demand_id="D1", origin="CenterA", destination="CenterC", demand_date=sdate)
        d2 = MagicMock(demand_id="D2", origin="CenterC", destination="CenterB", demand_date=sdate)
        d3 = MagicMock(demand_id="D3", origin="CenterX", destination="CenterY", demand_date=sdate)
        d4 = MagicMock(demand_id="D4", origin="CenterX", destination="CenterY", demand_date=sdate)

        proxy = MagicMock()
        proxy.demands = {"D1": d1, "D2": d2, "D3": d3, "D4": d4}
        proxy.arcs = {
            ("CenterA", "CenterB"): 10.0,
            ("CenterA", "CenterC"): 5.0,
            ("CenterC", "CenterB"): 5.0,
            ("CenterX", "CenterY"): 10.0,
        }
        v = MagicMock(capacity_desi=1000.0)
        v.name = "Kamyonet"
        proxy.vehicles = {"Kamyonet": v}
        proxy.evaluate_route.return_value = 100.0

        problem = MagicMock()
        problem.demands = [d1, d2, d3, d4]
        problem.static.vehicle_types = [v]
        policy = MagicMock(desi_tolerance=0.0)

        # 1. Indexed Production Repair
        state_indexed = greedy_insertion(initial_state, unassigned, problem, policy, proxy)

        # 2. Reference Brute-Force Repair
        state_brute = brute_force_greedy_insertion(initial_state, unassigned, problem, policy, proxy)

        # 3. Assert full equivalence
        self.assertEqual(state_indexed.canonical_tuple(), state_brute.canonical_tuple())
        self.assertEqual(len(state_indexed.spot_routes), len(state_brute.spot_routes))

    def test_indexed_vs_brute_force_regret_2_insertion_update_and_append(self):
        """Indexed regret_2_insertion matches brute-force linear-scan on route updates and appends."""
        sdate = datetime.date(2026, 7, 1)

        p1 = DemandPart(part_id="P1", demand_id="D1", desi=10.0)
        p2 = DemandPart(part_id="P2", demand_id="D2", desi=10.0)
        p3 = DemandPart(part_id="P3", demand_id="D3", desi=10.0)
        p4 = DemandPart(part_id="P4", demand_id="D4", desi=10.0)

        # Initial route r0: CenterA -> CenterB
        r0 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet")
        initial_state = ALNSSolutionState(parts=(p1, p2, p3, p4), mandatory_trips=(), spot_routes=(r0,))
        unassigned = [(p1,), (p2,), (p3,), (p4,)]

        d1 = MagicMock(demand_id="D1", origin="CenterA", destination="CenterC", demand_date=sdate)
        d2 = MagicMock(demand_id="D2", origin="CenterC", destination="CenterB", demand_date=sdate)
        d3 = MagicMock(demand_id="D3", origin="CenterX", destination="CenterY", demand_date=sdate)
        d4 = MagicMock(demand_id="D4", origin="CenterX", destination="CenterY", demand_date=sdate)

        proxy = MagicMock()
        proxy.demands = {"D1": d1, "D2": d2, "D3": d3, "D4": d4}
        proxy.arcs = {
            ("CenterA", "CenterB"): 10.0,
            ("CenterA", "CenterC"): 5.0,
            ("CenterC", "CenterB"): 5.0,
            ("CenterX", "CenterY"): 10.0,
        }
        v = MagicMock(capacity_desi=1000.0)
        v.name = "Kamyonet"
        proxy.vehicles = {"Kamyonet": v}
        proxy.evaluate_route.return_value = 100.0

        problem = MagicMock()
        problem.demands = [d1, d2, d3, d4]
        problem.static.vehicle_types = [v]
        policy = MagicMock(desi_tolerance=0.0)

        # 1. Indexed Production Repair
        state_indexed = regret_2_insertion(initial_state, unassigned, problem, policy, proxy)

        # 2. Reference Brute-Force Repair
        state_brute = brute_force_regret_2_insertion(initial_state, unassigned, problem, policy, proxy)

        # 3. Assert full equivalence
        self.assertEqual(state_indexed.canonical_tuple(), state_brute.canonical_tuple())
        self.assertEqual(len(state_indexed.spot_routes), len(state_brute.spot_routes))


def brute_force_greedy_insertion(state, unassigned, problem, policy, proxy):
    import math
    from src.alns.repair import generate_insertion_candidates, score_insertion_candidates, select_best_vehicle_for_route
    from src.alns.exceptions import RepairInfeasibleError
    spot_routes = list(state.spot_routes)
    unassigned_parts = [p for bin_parts in unassigned for p in bin_parts]
    unassigned_parts.sort(key=lambda p: (p.desi, p.demand_id, p.part_id), reverse=True)
    route_center_maps = [{stop.center for stop in route.stops} for route in spot_routes]

    for part in unassigned_parts:
        demand = proxy.demands[part.demand_id]
        service_date, origin, destination = demand.demand_date, demand.origin, demand.destination
        best_cost_delta, best_route_idx, best_new_route = float('inf'), -1, None

        for route_idx, route in enumerate(spot_routes):
            if route.service_date != service_date: continue
            if origin not in route_center_maps[route_idx] and destination not in route_center_maps[route_idx]: continue
            old_cost = proxy.evaluate_route(route.service_date, route.stops, route.vehicle_type)
            if math.isinf(old_cost): continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            new_cost, new_route = score_insertion_candidates(candidates, service_date, proxy, policy)
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_route_idx = route_idx
                    best_new_route = new_route

        new_stop_tuple = (ALNSRouteStop(origin, (), (part,)), ALNSRouteStop(destination, (part,), ()))
        direct_cost, direct_route = score_insertion_candidates([new_stop_tuple], service_date, proxy, policy)
        if direct_cost < best_cost_delta:
            best_cost_delta = direct_cost
            best_route_idx = -1
            best_new_route = direct_route

        if best_route_idx >= 0:
            spot_routes[best_route_idx] = best_new_route
            route_center_maps[best_route_idx] = {stop.center for stop in best_new_route.stops}
        elif best_new_route is not None and not math.isinf(best_cost_delta):
            spot_routes.append(best_new_route)
            route_center_maps.append({stop.center for stop in best_new_route.stops})
        else:
            raise RepairInfeasibleError(f"No feasible insertion for part {part.part_id}")

    for i, route in enumerate(spot_routes):
        _, best_v_name = select_best_vehicle_for_route(route.stops, route.service_date, proxy, policy)
        if best_v_name and best_v_name != route.vehicle_type:
            spot_routes[i] = ALNSSpotRoute(route.service_date, route.stops, best_v_name)

    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(spot_routes))


def brute_force_regret_2_insertion(state, unassigned, problem, policy, proxy):
    import math
    from src.alns.repair import generate_insertion_candidates, score_insertion_candidates, select_best_vehicle_for_route
    from src.alns.exceptions import RepairInfeasibleError
    spot_routes = list(state.spot_routes)
    unassigned_parts = [p for bin_parts in unassigned for p in bin_parts]
    part_regrets = []
    route_center_maps = [{stop.center for stop in route.stops} for route in spot_routes]

    for part in unassigned_parts:
        demand = proxy.demands[part.demand_id]
        origin, destination, service_date = demand.origin, demand.destination, demand.demand_date
        best_cost, second_best_cost = float('inf'), float('inf')

        for r_idx, route in enumerate(spot_routes):
            if route.service_date != service_date: continue
            if origin not in route_center_maps[r_idx] and destination not in route_center_maps[r_idx]: continue
            old_cost = proxy.evaluate_route(route.service_date, route.stops, route.vehicle_type)
            if math.isinf(old_cost): continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            new_cost, _ = score_insertion_candidates(candidates, service_date, proxy, policy)
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost:
                    second_best_cost = best_cost
                    best_cost = delta
                elif delta < second_best_cost:
                    second_best_cost = delta

        new_stop_tuple = (ALNSRouteStop(origin, (), (part,)), ALNSRouteStop(destination, (part,), ()))
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
        origin, destination, service_date = demand.origin, demand.destination, demand.demand_date
        best_cost_delta, best_route_idx, best_new_route = float('inf'), -1, None

        for i, route in enumerate(spot_routes):
            if route.service_date != service_date: continue
            route_centers = {stop.center for stop in route.stops}
            if origin not in route_centers and destination not in route_centers: continue
            old_cost = proxy.evaluate_route(route.service_date, route.stops, route.vehicle_type)
            if math.isinf(old_cost): continue

            candidates = generate_insertion_candidates(route.stops, origin, destination, part)
            new_cost, new_route = score_insertion_candidates(candidates, service_date, proxy, policy)
            if not math.isinf(new_cost):
                delta = new_cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_route_idx = i
                    best_new_route = new_route

        new_stop_tuple = (ALNSRouteStop(origin, (), (part,)), ALNSRouteStop(destination, (part,), ()))
        direct_cost, direct_route = score_insertion_candidates([new_stop_tuple], service_date, proxy, policy)
        if direct_cost < best_cost_delta:
            best_cost_delta = direct_cost
            best_route_idx = -1
            best_new_route = direct_route

        if best_route_idx >= 0:
            spot_routes[best_route_idx] = best_new_route
            route_center_maps[best_route_idx] = {stop.center for stop in best_new_route.stops}
        elif best_new_route is not None and not math.isinf(best_cost_delta):
            spot_routes.append(best_new_route)
            route_center_maps.append({stop.center for stop in best_new_route.stops})
        else:
            raise RepairInfeasibleError(f"No feasible insertion for part {part.part_id}")

    for i, route in enumerate(spot_routes):
        _, best_v_name = select_best_vehicle_for_route(route.stops, route.service_date, proxy, policy)
        if best_v_name and best_v_name != route.vehicle_type:
            spot_routes[i] = ALNSSpotRoute(route.service_date, route.stops, best_v_name)

    return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(spot_routes))


if __name__ == "__main__":
    unittest.main()
