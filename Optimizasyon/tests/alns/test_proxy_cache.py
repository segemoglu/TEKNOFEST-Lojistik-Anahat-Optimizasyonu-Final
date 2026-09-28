"""Synthetic unit tests for ProxyEvaluator LRU cache bounds and primitive key representation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime

from src.alns.evaluator import ProxyEvaluator
from src.alns.state import ALNSSolutionState, ALNSRouteStop
from src.domain import DemandPart, VehicleType, TravelArc, ProblemData, StaticProblemData, DemandInstance, Center
from src.evaluation import EvaluationPolicy


def make_problem_and_policy() -> tuple[ProblemData, EvaluationPolicy, ALNSSolutionState]:
    v_kamyonet = VehicleType("Kamyonet", capacity_desi=500.0, rented_hourly_cost_tl=100.0, rented_cost_per_km_tl=10.0, spot_hourly_cost_tl=100.0, spot_cost_per_km_tl=10.0)
    v_kamyon = VehicleType("Kamyon", capacity_desi=1500.0, rented_hourly_cost_tl=200.0, rented_cost_per_km_tl=20.0, spot_hourly_cost_tl=200.0, spot_cost_per_km_tl=20.0)
    vehicles = [v_kamyonet, v_kamyon]

    arc1 = TravelArc("OriginA", "DestB", distance_km=100.0, travel_hours_by_vehicle={"Kamyonet": 2.0, "Kamyon": 2.0}, target_delivery_days=1)
    arc2 = TravelArc("DestB", "DestC", distance_km=50.0, travel_hours_by_vehicle={"Kamyonet": 1.0, "Kamyon": 1.0}, target_delivery_days=1)
    arcs = [arc1, arc2]

    centers = [Center("OriginA"), Center("DestB"), Center("DestC")]
    static = StaticProblemData(centers=centers, vehicle_types=vehicles, travel_arcs=arcs, handling_capacities=[], daily_truck_capacities=[], mandatory_rental_routes=[])

    demands = [
        DemandInstance("D001", datetime.date(2026, 7, 1), datetime.time(8, 0), "OriginA", "DestB", 100.0),
        DemandInstance("D002", datetime.date(2026, 7, 1), datetime.time(8, 0), "OriginA", "DestC", 200.0),
    ]

    problem = ProblemData(static=static, demands=demands)
    policy = EvaluationPolicy(handling_minutes_per_desi=0.1, desi_tolerance=10.0, sla_penalty_per_desi_hour_tl=50.0)
    dummy_state = ALNSSolutionState(parts=(), mandatory_trips=(), spot_routes=())
    return problem, policy, dummy_state


def make_route_stop(center: str, pickups: list[DemandPart], dropoffs: list[DemandPart]) -> ALNSRouteStop:
    pickup_desi = sum(p.desi for p in pickups)
    dropoff_desi = sum(p.desi for p in dropoffs)
    return ALNSRouteStop(
        center=center,
        pickups=tuple(pickups),
        dropoffs=tuple(dropoffs),
        pickup_desi=pickup_desi,
        dropoff_desi=dropoff_desi,
    )


class ProxyCacheTests(unittest.TestCase):
    """Test suite for ProxyEvaluator LRU caching bounds, hit/miss tracking, and primitive keys."""

    def setUp(self):
        self.problem, self.policy, self.initial_state = make_problem_and_policy()
        self.date = datetime.date(2026, 7, 1)

    def test_feasible_routes_respect_cache_maxsize(self):
        """Evaluating 10 feasible routes with cache_maxsize=3 keeps size <= 3."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=3)

        for i in range(10):
            part = DemandPart(part_id=f"PART-{i}", demand_id="D001", desi=10.0 + i)
            stop1 = make_route_stop("OriginA", pickups=[part], dropoffs=[])
            stop2 = make_route_stop("DestB", pickups=[], dropoffs=[part])
            evaluator.evaluate_route(self.date, (stop1, stop2), "Kamyonet")

        self.assertLessEqual(len(evaluator._route_cost_cache), 3)
        self.assertEqual(evaluator.cache_info()["currsize"], len(evaluator._route_cost_cache))

    def test_infeasible_routes_respect_cache_maxsize(self):
        """Evaluating 10 infeasible routes (capacity exceeded / no path) with cache_maxsize=3 keeps size <= 3."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=3)

        for i in range(10):
            # Over capacity part (> 500 desi capacity + 10 tolerance)
            part = DemandPart(part_id=f"OVER-{i}", demand_id="D001", desi=1000.0 + i)
            stop1 = make_route_stop("OriginA", pickups=[part], dropoffs=[])
            stop2 = make_route_stop("DestB", pickups=[], dropoffs=[part])
            cost = evaluator.evaluate_route(self.date, (stop1, stop2), "Kamyonet")
            self.assertEqual(cost, float("inf"))

        self.assertLessEqual(len(evaluator._route_cost_cache), 3)

    def test_mixed_early_return_routes_respect_cache_maxsize(self):
        """Mixing feasible and infeasible (no path / capacity / SLA) routes preserves cache_maxsize."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=3)

        for i in range(12):
            if i % 3 == 0:
                # Feasible
                part = DemandPart(part_id=f"P-{i}", demand_id="D001", desi=20.0)
                stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestB", [], [part]))
            elif i % 3 == 1:
                # Infeasible (No path: OriginA -> DestC directly without arc)
                part = DemandPart(part_id=f"P-{i}", demand_id="D002", desi=20.0)
                stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestC", [], [part]))
            else:
                # Infeasible (Over capacity)
                part = DemandPart(part_id=f"P-{i}", demand_id="D001", desi=2000.0)
                stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestB", [], [part]))

            evaluator.evaluate_route(self.date, stops, "Kamyonet")

        self.assertLessEqual(len(evaluator._route_cost_cache), 3)

    def test_cache_hit_refreshes_lru_order(self):
        """Cache hit moves key to end (most recently used); the least recently used key is evicted when capacity is reached."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=3)

        part1 = DemandPart(part_id="P1", demand_id="D001", desi=10.0)
        part2 = DemandPart(part_id="P2", demand_id="D001", desi=20.0)
        part3 = DemandPart(part_id="P3", demand_id="D001", desi=30.0)
        part4 = DemandPart(part_id="P4", demand_id="D001", desi=40.0)

        stops1 = (make_route_stop("OriginA", [part1], []), make_route_stop("DestB", [], [part1]))
        stops2 = (make_route_stop("OriginA", [part2], []), make_route_stop("DestB", [], [part2]))
        stops3 = (make_route_stop("OriginA", [part3], []), make_route_stop("DestB", [], [part3]))
        stops4 = (make_route_stop("OriginA", [part4], []), make_route_stop("DestB", [], [part4]))

        # Fill cache with 3 routes: R1, R2, R3
        evaluator.evaluate_route(self.date, stops1, "Kamyonet")
        evaluator.evaluate_route(self.date, stops2, "Kamyonet")
        evaluator.evaluate_route(self.date, stops3, "Kamyonet")

        # Hit R1 -> Moves R1 to most recently used. Order becomes: R2, R3, R1
        evaluator.evaluate_route(self.date, stops1, "Kamyonet")

        # Now add R4 -> Should evict R2 (which is now least recently used)
        evaluator.evaluate_route(self.date, stops4, "Kamyonet")

        # Check keys present in cache
        keys_in_cache = list(evaluator._route_cost_cache.keys())
        self.assertEqual(len(keys_in_cache), 3)

        # R1, R3, R4 must be present; R2 must NOT be in cache
        key1 = evaluator._make_cache_key(self.date, stops1, "Kamyonet")
        key2 = evaluator._make_cache_key(self.date, stops2, "Kamyonet")
        key3 = evaluator._make_cache_key(self.date, stops3, "Kamyonet")
        key4 = evaluator._make_cache_key(self.date, stops4, "Kamyonet")

        self.assertIn(key1, evaluator._route_cost_cache)
        self.assertNotIn(key2, evaluator._route_cost_cache)
        self.assertIn(key3, evaluator._route_cost_cache)
        self.assertIn(key4, evaluator._route_cost_cache)

    def test_eviction_removes_least_recently_used_entry(self):
        """Without hit, evaluating 4 items in maxsize=3 evicts item 1 (oldest)."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=3)

        stops_list = []
        for i in range(4):
            part = DemandPart(part_id=f"P-{i}", demand_id="D001", desi=10.0 * (i + 1))
            stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestB", [], [part]))
            stops_list.append(stops)
            evaluator.evaluate_route(self.date, stops, "Kamyonet")

        key0 = evaluator._make_cache_key(self.date, stops_list[0], "Kamyonet")
        key3 = evaluator._make_cache_key(self.date, stops_list[3], "Kamyonet")

        self.assertNotIn(key0, evaluator._route_cost_cache)
        self.assertIn(key3, evaluator._route_cost_cache)

    def test_subsequent_call_returns_cached_value(self):
        """Evaluating the same route twice hits cache and avoids recomputation."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=10)
        part = DemandPart(part_id="P-SAME", demand_id="D001", desi=50.0)
        stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestB", [], [part]))

        cost1 = evaluator.evaluate_route(self.date, stops, "Kamyonet")
        cost2 = evaluator.evaluate_route(self.date, stops, "Kamyonet")

        self.assertEqual(cost1, cost2)
        info = evaluator.cache_info()
        self.assertEqual(info["hits"], 1)
        self.assertEqual(info["misses"], 1)

    def test_cache_key_contains_only_primitive_types(self):
        """Keys stored in _route_cost_cache must contain only primitive types and no domain objects."""
        evaluator = ProxyEvaluator(self.problem, self.policy, self.initial_state, cache_maxsize=10)
        part = DemandPart(part_id="P-PRIM", demand_id="D001", desi=50.0)
        stops = (make_route_stop("OriginA", [part], []), make_route_stop("DestB", [], [part]))

        evaluator.evaluate_route(self.date, stops, "Kamyonet")

        def assert_primitives(obj):
            if isinstance(obj, tuple):
                for elem in obj:
                    assert_primitives(elem)
            else:
                self.assertIsInstance(obj, (str, int, float, bool, type(None)))
                self.assertNotIsInstance(obj, (DemandPart, ALNSRouteStop))

        for key in evaluator._route_cost_cache.keys():
            assert_primitives(key)


if __name__ == "__main__":
    unittest.main()
