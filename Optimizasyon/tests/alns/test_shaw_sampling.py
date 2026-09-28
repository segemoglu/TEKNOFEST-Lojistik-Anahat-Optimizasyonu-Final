"""Unit tests for bounded random sampling in Shaw removal destroy operator."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import random
import datetime

from src.alns.destroy import shaw_removal
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.domain import DemandPart


def make_large_state(num_parts: int = 1000) -> tuple[ALNSSolutionState, MagicMock]:
    """Helper to create a synthetic state with N parts across spot routes."""
    sdate = datetime.date(2026, 7, 1)
    parts = []
    routes = []

    proxy = MagicMock()
    proxy.demands = {}
    proxy.arcs = {}

    parts_per_route = 10
    num_routes = num_parts // parts_per_route

    all_parts_list = []

    for r_idx in range(num_routes):
        route_parts = []
        for p_idx in range(parts_per_route):
            pid = r_idx * parts_per_route + p_idx
            part = DemandPart(part_id=f"P_{pid:04d}", demand_id=f"D_{pid:04d}", desi=10.0)
            parts.append(part)
            route_parts.append(part)

            demand = MagicMock()
            demand.origin = f"Origin_{r_idx}"
            demand.destination = f"Dest_{r_idx}"
            demand.demand_date = sdate
            proxy.demands[f"D_{pid:04d}"] = demand

        stops = (
            ALNSRouteStop(f"Origin_{r_idx}", (), tuple(route_parts)),
            ALNSRouteStop(f"Dest_{r_idx}", tuple(route_parts), ()),
        )
        routes.append(ALNSSpotRoute(sdate, stops, "Kamyonet"))

    state = ALNSSolutionState(parts=tuple(parts), mandatory_trips=(), spot_routes=tuple(routes))
    return state, proxy


class ShawSamplingTests(unittest.TestCase):
    """Test suite for shaw_removal bounded random sampling, determinism, and complexity."""

    def test_shaw_sampling_exact_removal_count(self):
        """shaw_removal removes exactly min(n, part_count) unique parts."""
        state, proxy = make_large_state(num_parts=100)
        rng = random.Random(42)

        new_state, unassigned = shaw_removal(state, n=15, rng=rng, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=16)

        self.assertEqual(len(unassigned), 15)
        removed_part_ids = {u[0].part_id for u in unassigned}
        self.assertEqual(len(removed_part_ids), 15, "All removed parts must be unique")

    def test_shaw_sampling_no_leaks_and_no_missing_parts(self):
        """Removed parts must be purged from spot routes without leaking or losing remaining parts."""
        state, proxy = make_large_state(num_parts=50)
        rng = random.Random(42)

        new_state, unassigned = shaw_removal(state, n=10, rng=rng, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=16)
        removed_parts = {u[0] for u in unassigned}

        remaining_parts_in_state = set()
        for route in new_state.spot_routes:
            for stop in route.stops:
                remaining_parts_in_state.update(stop.pickups)
                remaining_parts_in_state.update(stop.dropoffs)

        # 1. No removed part should remain in new_state
        self.assertEqual(len(remaining_parts_in_state & removed_parts), 0, "No removed part should remain in new state")

        # 2. Total parts must equal original state parts
        self.assertEqual(len(remaining_parts_in_state) + len(removed_parts), 50)

    def test_shaw_sampling_determinism_same_seed(self):
        """Same seed must produce 100% identical state and unassigned list."""
        state, proxy = make_large_state(num_parts=100)

        rng1 = random.Random(12345)
        state1, unassigned1 = shaw_removal(state, n=20, rng=rng1, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=32)

        rng2 = random.Random(12345)
        state2, unassigned2 = shaw_removal(state, n=20, rng=rng2, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=32)

        self.assertEqual(state1.canonical_tuple(), state2.canonical_tuple())
        self.assertEqual(unassigned1, unassigned2)

    def test_shaw_sampling_different_seeds(self):
        """Different seeds produce different selections."""
        state, proxy = make_large_state(num_parts=100)

        rng1 = random.Random(111)
        _, unassigned1 = shaw_removal(state, n=20, rng=rng1, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=16)

        rng2 = random.Random(999)
        _, unassigned2 = shaw_removal(state, n=20, rng=rng2, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=16)

        self.assertNotEqual(unassigned1, unassigned2)

    def test_shaw_sampling_complexity_bounded(self):
        """For 1,000 parts, n=100, sample_size=32, relatedness evaluations must be bounded ~3,168 (vs ~94,550 unsampled)."""
        state, proxy = make_large_state(num_parts=1000)
        rng = random.Random(42)

        class CountingDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.count = 0

            def get(self, key, default=None):
                self.count += 1
                return super().get(key, default)

        counting_arcs = CountingDict(proxy.arcs)
        proxy.arcs = counting_arcs

        n = 100
        sample_size = 32
        new_state, unassigned = shaw_removal(state, n=n, rng=rng, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=sample_size)

        evaluation_count = counting_arcs.count
        max_expected_evaluations = (n - 1) * sample_size
        unsampled_evaluations = sum(1000 - i for i in range(1, n))

        print("\n--- Shaw Sampling Complexity Metrics ---")
        print(f"Total Parts in State            : 1,000")
        print(f"Requested Removal Count (n)     : {n}")
        print(f"Sample Size                     : {sample_size}")
        print(f"Measured Relatedness Evaluations : {evaluation_count}")
        print(f"Max Bounded Expected             : {max_expected_evaluations}")
        print(f"Unsampled Full Scan Evaluations : {unsampled_evaluations}\n")

        self.assertEqual(len(unassigned), n)
        self.assertLessEqual(evaluation_count, max_expected_evaluations)
        self.assertLess(evaluation_count, 3300)

    def test_shaw_sampling_edge_case_n_greater_than_parts(self):
        """n >= part_count removes all parts and leaves empty spot routes."""
        state, proxy = make_large_state(num_parts=10)
        rng = random.Random(42)

        new_state, unassigned = shaw_removal(state, n=20, rng=rng, problem=MagicMock(), policy=MagicMock(), proxy=proxy, sample_size=64)

        self.assertEqual(len(unassigned), 10)
        self.assertEqual(new_state.spot_routes, ())

    def test_shaw_sampling_edge_case_empty_state(self):
        """Empty state returns identical state and empty unassigned list."""
        empty_state = ALNSSolutionState(parts=(), mandatory_trips=(), spot_routes=())
        rng = random.Random(42)

        new_state, unassigned = shaw_removal(empty_state, n=10, rng=rng, problem=MagicMock(), policy=MagicMock(), proxy=MagicMock(), sample_size=64)

        self.assertEqual(new_state, empty_state)
        self.assertEqual(unassigned, [])


if __name__ == "__main__":
    unittest.main()
