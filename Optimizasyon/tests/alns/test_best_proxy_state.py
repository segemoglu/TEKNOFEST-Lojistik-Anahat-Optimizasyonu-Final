"""Synthetic unit tests for ALNSEngine best proxy state tracking."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime

from src.alns.engine import ALNSEngine
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.domain import DemandPart


def make_dummy_state(part_id: str, cost_tag: str) -> ALNSSolutionState:
    part = DemandPart(part_id=part_id, demand_id="D001", desi=100.0)
    stop1 = ALNSRouteStop("OriginA", (), (part,))
    stop2 = ALNSRouteStop("DestB", (part,), ())
    route = ALNSSpotRoute(datetime.date(2026, 7, 1), (stop1, stop2), "Kamyonet")
    return ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))


class BestProxyStateEngineTests(unittest.TestCase):
    """Test suite for best proxy state tracking in ALNSEngine."""

    def test_engine_materializes_best_proxy_state_not_current_state(self):
        """When search finishes at state C, final materialization evaluates best proxy state B."""
        state_A = make_dummy_state("PART-A", "1000")
        state_B = make_dummy_state("PART-B", "500")
        state_C = make_dummy_state("PART-C", "800")

        # Initial plan and cost
        initial_plan = MagicMock(name="InitialPlan")
        initial_cost = 2000.0

        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_A,
            initial_plan=initial_plan,
            initial_cost=initial_cost,
            seed=42,
        )

        # Mock ProxyEvaluator costs: A=1000, B=500, C=800
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: {
            state_A: 1000.0,
            state_B: 500.0,
            state_C: 800.0,
        }[st])

        # Step 1: destroy & repair returns state_B
        # Step 2: destroy & repair returns state_C
        step_results = [
            (make_dummy_state("DESTROYED-1", "0"), [make_dummy_state("PART-B", "500")]),
            (make_dummy_state("DESTROYED-2", "0"), [make_dummy_state("PART-C", "800")]),
        ]

        def mock_destroy(st, n, rng, *args):
            return step_results[0][0], []

        # Simulated operator sequence: iteration 1 -> candidate_B, iteration 2 -> candidate_C
        repair_call_count = 0
        def mock_repair(st, unassigned, *args):
            nonlocal repair_call_count
            repair_call_count += 1
            if repair_call_count == 1:
                return state_B
            return state_C

        engine.destroy_ops = [mock_destroy]
        engine.repair_ops = [mock_repair]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        # Mock materializer, validator, calculator
        plan_B = MagicMock(name="PlanB")
        engine.materializer.materialize = MagicMock(return_value=plan_B)
        
        valid_report = MagicMock(is_valid=True)
        engine.validator.validate = MagicMock(return_value=valid_report)
        
        calculated_cost_B = MagicMock(total_cost_tl=450.0)
        engine.calculator.calculate = MagicMock(return_value=calculated_cost_B)

        # Force SA to accept state_C in iteration 2
        engine.rng.random = MagicMock(return_value=0.0001)

        # Run 2 iterations
        best_plan, best_cost = engine.run(max_iterations=2, start_temperature=10000.0)

        # Verify engine's current state ended as state_C
        self.assertEqual(engine.current_state, state_C)

        # VERIFY CRITICAL BEHAVIOR: materializer was called with state_B and state_B won as best plan
        engine.materializer.materialize.assert_any_call(state_B)
        self.assertEqual(best_plan, plan_B)
        self.assertEqual(best_cost, 450.0)

    def test_exact_b_fails_validation_returns_initial_incumbent(self):
        """If exact evaluation of best proxy state B fails validation, initial plan is returned."""
        state_A = make_dummy_state("PART-A", "1000")
        state_B = make_dummy_state("PART-B", "500")

        initial_plan = MagicMock(name="InitialPlan")
        initial_cost = 2000.0

        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_A,
            initial_plan=initial_plan,
            initial_cost=initial_cost,
            seed=42,
        )

        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: {
            state_A: 1000.0,
            state_B: 500.0,
        }.get(st, 1000.0))

        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_B]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        plan_B = MagicMock(name="PlanB")
        engine.materializer.materialize = MagicMock(return_value=plan_B)

        # Validation fails for plan_B
        invalid_report = MagicMock(is_valid=False, errors=["Over capacity"])
        engine.validator.validate = MagicMock(return_value=invalid_report)

        best_plan, best_cost = engine.run(max_iterations=1)

        # Fallback to initial plan
        self.assertEqual(best_plan, initial_plan)
        self.assertEqual(best_cost, 2000.0)


if __name__ == "__main__":
    unittest.main()
