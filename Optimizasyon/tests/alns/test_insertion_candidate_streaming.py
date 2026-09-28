"""Unit tests for streaming insertion candidate generation and scoring in ALNS repair."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime
import inspect
from collections.abc import Iterator, Iterable

from src.alns.repair import (
    generate_insertion_candidates,
    score_insertion_candidates,
)
from src.alns.state import ALNSRouteStop, ALNSSpotRoute
from src.domain import DemandPart


def make_sample_stops() -> tuple[ALNSRouteStop, ...]:
    p_existing = DemandPart(part_id="P_EXISTING", demand_id="D000", desi=50.0)
    stop1 = ALNSRouteStop("CenterA", (), (p_existing,), 0.0, 50.0)
    stop2 = ALNSRouteStop("CenterB", (), (), 0.0, 0.0)
    stop3 = ALNSRouteStop("CenterC", (p_existing,), (), 50.0, 0.0)
    return (stop1, stop2, stop3)


class InsertionCandidateStreamingTests(unittest.TestCase):
    """Test suite for lazy, streaming insertion candidate generation and scoring."""

    def setUp(self):
        self.part = DemandPart(part_id="P_NEW", demand_id="D001", desi=100.0)
        self.stops = make_sample_stops()

    def test_generate_insertion_candidates_returns_iterator(self):
        """generate_insertion_candidates must return a generator/iterator, not an eager list."""
        gen = generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part)
        self.assertFalse(isinstance(gen, list), "Should not return an eager list")
        self.assertTrue(
            inspect.isgenerator(gen) or isinstance(gen, Iterator),
            "Must return a generator or Iterator instance",
        )

    def test_candidates_valid_pickup_dropoff_order(self):
        """Every generated candidate must have pickup_index <= dropoff_index."""
        gen = generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part)
        candidates = list(gen)
        self.assertGreater(len(candidates), 0)

        for candidate_stops in candidates:
            pickup_indices = [
                i for i, s in enumerate(candidate_stops) if any(p.part_id == "P_NEW" for p in s.pickups)
            ]
            dropoff_indices = [
                i for i, s in enumerate(candidate_stops) if any(p.part_id == "P_NEW" for p in s.dropoffs)
            ]
            self.assertTrue(len(pickup_indices) > 0, "Part must be picked up")
            self.assertTrue(len(dropoff_indices) > 0, "Part must be dropped off")

            p_idx = pickup_indices[0]
            d_idx = dropoff_indices[0]
            self.assertLessEqual(
                p_idx,
                d_idx,
                f"Pickup index ({p_idx}) must be <= dropoff index ({d_idx})",
            )

    def test_lazy_evaluation_does_not_precalculate_all(self):
        """Generator must yield candidates lazily item by item when next() is called."""
        gen = generate_insertion_candidates(self.stops, "NewOrigin", "NewDest", self.part)

        # Retrieve first item
        first_item = next(gen)
        self.assertIsInstance(first_item, tuple)

        # Retrieve second item
        second_item = next(gen)
        self.assertIsInstance(second_item, tuple)
        self.assertNotEqual(first_item, second_item)

    def test_streaming_scorer_matches_reference_list_scorer(self):
        """Streaming score_insertion_candidates yields identical cost and route to eager list evaluation."""
        proxy = MagicMock()
        proxy.arcs = {
            ("CenterA", "CenterB"): 10.0,
            ("CenterB", "CenterC"): 10.0,
            ("CenterA", "CenterC"): 15.0,
        }
        vehicle = MagicMock()
        vehicle.name = "Kamyonet"
        vehicle.capacity_desi = 1000.0
        proxy.vehicles = {"Kamyonet": vehicle}
        proxy.evaluate_route.return_value = 500.0

        policy = MagicMock()
        policy.desi_tolerance = 0.0

        service_date = datetime.date(2026, 7, 1)

        # 1. Generator evaluation
        gen = generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part)
        cost_gen, route_gen = score_insertion_candidates(gen, service_date, proxy, policy)

        # 2. List evaluation (Reference)
        candidates_list = list(generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part))
        cost_list, route_list = score_insertion_candidates(candidates_list, service_date, proxy, policy)

        self.assertEqual(cost_gen, cost_list)
        self.assertEqual(route_gen, route_list)

    def test_deterministic_candidate_order(self):
        """Multiple iterations over generate_insertion_candidates produce identical candidate sequences."""
        gen1 = generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part)
        gen2 = generate_insertion_candidates(self.stops, "CenterA", "CenterC", self.part)

        list1 = list(gen1)
        list2 = list(gen2)

        self.assertEqual(len(list1), len(list2))
        self.assertEqual(list1, list2)

    def test_fixture_three_stop_candidates_completeness(self):
        """Verify that all valid insertion positions are generated for a 3-stop route."""
        gen = generate_insertion_candidates(self.stops, "NewOrigin", "NewDest", self.part)
        candidates = list(gen)

        # New origin can be inserted before 0, 1, 2, or 3 -> 4 positions.
        # New dest can be inserted after new origin position -> multiple positions.
        self.assertGreaterEqual(len(candidates), 6)
        for c in candidates:
            self.assertIsInstance(c, tuple)

    def test_regret_2_insertion_streaming_versus_eager_reference(self):
        """Streaming regret_2_insertion yields identical regret, part selection, route, and tie-break to eager list reference."""
        from src.alns.repair import regret_2_insertion, select_best_vehicle_for_route
        from src.alns.state import ALNSSolutionState

        p1 = DemandPart(part_id="P1", demand_id="D1", desi=10.0)
        p2 = DemandPart(part_id="P2", demand_id="D2", desi=10.0)
        p3 = DemandPart(part_id="P3", demand_id="D3", desi=10.0)

        sdate = datetime.date(2026, 7, 1)

        r1 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterA", (), ()), ALNSRouteStop("CenterB", (), ())), "Kamyonet")
        r2 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterC", (), ()), ALNSRouteStop("CenterD", (), ())), "Kamyonet")
        r3 = ALNSSpotRoute(sdate, (ALNSRouteStop("CenterE", (), ()), ALNSRouteStop("CenterF", (), ())), "Kamyonet")

        initial_state = ALNSSolutionState(parts=(p1, p2, p3), mandatory_trips=(), spot_routes=(r1, r2, r3))
        unassigned = [(p1,), (p2,), (p3,)]

        # Mock demands
        d1 = MagicMock(origin="CenterA", destination="CenterB", demand_date=sdate)
        d2 = MagicMock(origin="CenterC", destination="CenterD", demand_date=sdate)
        d3 = MagicMock(origin="CenterE", destination="CenterF", demand_date=sdate)

        proxy = MagicMock()
        proxy.demands = {"D1": d1, "D2": d2, "D3": d3}
        proxy.arcs = {
            ("CenterA", "CenterB"): 10.0,
            ("CenterC", "CenterD"): 10.0,
            ("CenterE", "CenterF"): 10.0,
        }
        v = MagicMock()
        v.name = "Kamyonet"
        v.capacity_desi = 1000.0
        proxy.vehicles = {"Kamyonet": v}

        # Mock evaluate_route to yield 3 distinct insertion costs per part:
        # P1: Best=100, 2nd=500, 3rd=600 -> Regret=400
        # P2: Best=100, 2nd=200, 3rd=300 -> Regret=100
        # P3: Best=100, 2nd=150, 3rd=250 -> Regret=50
        def custom_evaluate(date, stops, vehicle_type):
            part_ids = [p.part_id for s in stops for p in (s.pickups + s.dropoffs)]
            if "P1" in part_ids:
                if any(s.center == "CenterA" for s in stops):
                    return 100.0
                elif any(s.center == "CenterC" for s in stops):
                    return 500.0
                else:
                    return 600.0
            elif "P2" in part_ids:
                if any(s.center == "CenterC" for s in stops):
                    return 100.0
                elif any(s.center == "CenterA" for s in stops):
                    return 200.0
                else:
                    return 300.0
            elif "P3" in part_ids:
                if any(s.center == "CenterE" for s in stops):
                    return 100.0
                elif any(s.center == "CenterA" for s in stops):
                    return 150.0
                else:
                    return 250.0
            return 1000.0

        proxy.evaluate_route.side_effect = custom_evaluate

        policy = MagicMock()
        policy.desi_tolerance = 0.0

        # Eager reference implementation of regret_2_insertion (converts generator to eager list)
        def eager_regret_2_insertion(state, unassigned, problem, policy, proxy):
            import math
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
                    route_centers = route_center_maps[r_idx]
                    if origin not in route_centers and destination not in route_centers: continue
                    old_cost = proxy.evaluate_route(route.service_date, route.stops, route.vehicle_type)
                    if math.isinf(old_cost): continue

                    # EAGER LIST CONVERSION REFERENCE
                    candidates = list(generate_insertion_candidates(route.stops, origin, destination, part))
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

                    # EAGER LIST CONVERSION REFERENCE
                    candidates = list(generate_insertion_candidates(route.stops, origin, destination, part))
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
                elif best_new_route is not None and not math.isinf(best_cost_delta):
                    spot_routes.append(best_new_route)

            for i, route in enumerate(spot_routes):
                _, best_v_name = select_best_vehicle_for_route(route.stops, route.service_date, proxy, policy)
                if best_v_name and best_v_name != route.vehicle_type:
                    spot_routes[i] = ALNSSpotRoute(route.service_date, route.stops, best_v_name)

            return ALNSSolutionState(state.parts, state.mandatory_trips, tuple(spot_routes))

        # 1. Run Production Streaming Implementation
        state_streaming = regret_2_insertion(initial_state, unassigned, MagicMock(), policy, proxy)

        # 2. Run Reference Eager Implementation
        state_eager = eager_regret_2_insertion(initial_state, unassigned, MagicMock(), policy, proxy)

        # 3. Verify exact equivalence
        self.assertEqual(len(state_streaming.spot_routes), len(state_eager.spot_routes))
        for r_stream, r_eager in zip(state_streaming.spot_routes, state_eager.spot_routes):
            self.assertEqual(r_stream.service_date, r_eager.service_date)
            self.assertEqual(r_stream.vehicle_type, r_eager.vehicle_type)
            self.assertEqual(len(r_stream.stops), len(r_eager.stops))
            for s_str, s_eag in zip(r_stream.stops, r_eager.stops):
                self.assertEqual(s_str.center, s_eag.center)
                self.assertEqual(s_str.pickups, s_eag.pickups)
                self.assertEqual(s_str.dropoffs, s_eag.dropoffs)


if __name__ == "__main__":
    unittest.main()
