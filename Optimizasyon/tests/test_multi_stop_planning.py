"""Synthetic tests for bounded three/four-stop cheapest-insertion 6E."""

from __future__ import annotations

from collections import Counter
import unittest

from src.planning import (
    BaselineBuildError,
    GreedySameOriginMultiStopBuilder,
    GreedySameOriginTwoStopBuilder,
)
from tests.test_two_stop_planning import _demand, _policy_for, _problem


def _chain_distances(destinations: tuple[str, ...]) -> dict[tuple[str, str], float]:
    """Return short forward arcs for one synthetic destination chain."""

    return {
        (origin, destination): 10.0
        for origin, destination in zip(
            destinations[:-1],
            destinations[1:],
            strict=True,
        )
    }


def _multi_stop_trips(result) -> tuple:
    """Return only spot trips containing at least three delivery legs."""

    return tuple(
        trip
        for trip in result.plan.trips
        if trip.ownership_type == "spot" and len(trip.legs) >= 3
    )


class GreedySameOriginMultiStopBuilderTests(unittest.TestCase):
    """Verify insertion, load decay, stop cap, economics and determinism."""

    def test_three_destinations_form_one_shrinking_load_route(self) -> None:
        """One load and three exact unloads beat 6D and pass the validator."""

        demands = (
            _demand("D1", "B", 30.0),
            _demand("D2", "C", 30.0),
            _demand("D3", "D", 30.0),
        )
        problem = _problem(
            demands,
            distances=_chain_distances(("B", "C", "D")),
        )
        policy = _policy_for(demands)
        two_stop = GreedySameOriginTwoStopBuilder(problem, policy).build()
        result = GreedySameOriginMultiStopBuilder(
            problem,
            policy,
        ).build_from_two_stop(two_stop)
        trip = _multi_stop_trips(result)[0]

        self.assertEqual(tuple(leg.destination for leg in trip.legs), ("B", "C", "D"))
        self.assertEqual(len(trip.handling_operations), 4)
        self.assertEqual(
            [sum(item.desi for item in leg.assignments) for leg in trip.legs],
            [90.0, 60.0, 30.0],
        )
        self.assertEqual(trip.legs[1].loading_operation_id, None)
        self.assertEqual(trip.legs[2].loading_operation_id, None)
        self.assertLess(result.total_cost_tl, two_stop.total_cost_tl)
        self.assertTrue(result.validation_report.is_valid)
        finals = Counter(
            item.demand_part_id
            for leg in trip.legs
            for item in leg.assignments
            if item.is_final_delivery
        )
        self.assertEqual(finals, Counter({"D1-1": 1, "D2-1": 1, "D3-1": 1}))

    def test_four_stop_cap_is_reached_but_never_exceeded(self) -> None:
        """Four compatible destinations share one route with exact load decay."""

        destinations = ("B", "C", "D", "E")
        demands = tuple(
            _demand(f"D{index}", destination, 20.0)
            for index, destination in enumerate(destinations, start=1)
        )
        problem = _problem(
            demands,
            distances=_chain_distances(destinations),
        )
        policy = _policy_for(demands)
        result = GreedySameOriginMultiStopBuilder(problem, policy).build()
        trip = _multi_stop_trips(result)[0]

        self.assertEqual(len(trip.legs), 4)
        self.assertEqual(tuple(leg.destination for leg in trip.legs), destinations)
        self.assertEqual(
            [sum(item.desi for item in leg.assignments) for leg in trip.legs],
            [80.0, 60.0, 40.0, 20.0],
        )
        self.assertTrue(all(len(item.legs) <= 4 for item in result.plan.trips))

    def test_capacity_blocked_third_stop_keeps_6d_as_safe_fallback(self) -> None:
        """A 130-desi triple cannot exceed the 120-desi non-Tır maximum."""

        demands = (
            _demand("D1", "B", 50.0),
            _demand("D2", "C", 50.0),
            _demand("D3", "D", 30.0),
        )
        problem = _problem(
            demands,
            distances=_chain_distances(("B", "C", "D")),
        )
        policy = _policy_for(demands)
        two_stop = GreedySameOriginTwoStopBuilder(problem, policy).build()

        with self.assertRaises(BaselineBuildError) as captured:
            GreedySameOriginMultiStopBuilder(
                problem,
                policy,
            ).build_from_two_stop(two_stop)

        self.assertIn("multi_stop_no_strict_saving", captured.exception.error_codes)
        self.assertTrue(two_stop.validation_report.is_valid)

    def test_same_input_produces_identical_routes_ids_and_times(self) -> None:
        """Cheapest insertion remains deterministic across complete rebuilds."""

        destinations = ("B", "C", "D", "E")
        demands = tuple(
            _demand(f"D{index}", destination, 20.0)
            for index, destination in enumerate(destinations, start=1)
        )
        problem = _problem(
            demands,
            distances=_chain_distances(destinations),
        )
        policy = _policy_for(demands)

        first = GreedySameOriginMultiStopBuilder(problem, policy).build()
        second = GreedySameOriginMultiStopBuilder(problem, policy).build()

        self.assertEqual(first.plan, second.plan)


if __name__ == "__main__":
    unittest.main()
