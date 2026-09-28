"""Synthetic unit tests for ALNSEngine exact audit error safety and original incumbent fallback."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime

from src.alns.engine import ALNSEngine
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.domain import DemandPart


def make_dummy_state(part_id: str) -> ALNSSolutionState:
    part = DemandPart(part_id=part_id, demand_id="D001", desi=100.0)
    stop1 = ALNSRouteStop("OriginA", (), (part,))
    stop2 = ALNSRouteStop("DestB", (part,), ())
    route = ALNSSpotRoute(datetime.date(2026, 7, 1), (stop1, stop2), "Kamyonet")
    return ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))


class ExactFallbackTests(unittest.TestCase):
    """Test suite for error-safe exact auditing and original plan fallback in ALNSEngine."""

    def setUp(self):
        self.state_init = make_dummy_state("PART-INIT")
        self.initial_plan = MagicMock(name="InitialPlan")
        self.initial_cost = 10000.0

        self.problem = MagicMock(name="ProblemData")
        self.policy = MagicMock(name="EvaluationPolicy")

    def test_materializer_exception_falls_back_to_original_plan(self):
        """When materializer raises an Exception, candidate is rejected and original plan is returned."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CAND")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        # Materializer raises ValueError
        engine.materializer.materialize = MagicMock(side_effect=ValueError("Invalid state structure"))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_validator_invalid_falls_back_to_original_plan(self):
        """When validator returns is_valid=False, candidate is rejected and original plan is returned."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CAND")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=False, errors=["Over capacity"]))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_validator_exception_falls_back_to_original_plan(self):
        """When validator raises an Exception, candidate is rejected and original plan is returned."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CAND")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(side_effect=RuntimeError("Validator error"))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_calculator_exception_falls_back_to_original_plan(self):
        """When calculator raises an Exception, candidate is rejected and original plan is returned."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CAND")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(side_effect=ZeroDivisionError("Calculator math error"))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_non_finite_exact_cost_falls_back_to_original_plan(self):
        """When exact cost is NaN or inf, candidate is rejected and original plan is returned."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CAND")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=float("nan")))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_valid_and_cheaper_candidate_replaces_original_plan(self):
        """When candidate is valid and exact cost is lower than initial, new plan replaces initial plan."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-CHEAPER")
        new_plan = MagicMock(name="NewCheaperPlan")

        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=new_plan)
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=4500.0))

        best_plan, best_cost = engine.run(max_iterations=1)

        self.assertIs(best_plan, new_plan)
        self.assertEqual(best_cost, 4500.0)

    def test_faulty_candidate_does_not_block_subsequent_valid_candidate(self):
        """A faulty candidate in step 1 does not block a valid, cheaper candidate in step 2."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_faulty = make_dummy_state("PART-FAULTY")
        state_valid = make_dummy_state("PART-VALID")
        valid_plan = MagicMock(name="ValidPlan")

        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 4000.0 if st in (state_faulty, state_valid) else 10000.0)

        step_count = 0
        def mock_repair(st, unassigned, *args):
            nonlocal step_count
            step_count += 1
            return state_faulty if step_count == 1 else state_valid

        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [mock_repair]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        def mock_materialize(st):
            if st == state_faulty:
                raise ValueError("Faulty state materialization failed")
            return valid_plan

        engine.materializer.materialize = MagicMock(side_effect=mock_materialize)
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=3500.0))

        best_plan, best_cost = engine.run(max_iterations=2)

        self.assertIs(best_plan, valid_plan)
        self.assertEqual(best_cost, 3500.0)

    def test_fallback_plan_object_identity(self):
        """Fallback plan returned by engine is the exact same object instance as initial_plan (using 'is')."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        # Force state evaluation to return non-improving costs
        engine.proxy.evaluate_state = MagicMock(return_value=12000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: make_dummy_state("PART-EXPENSIVE")]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        best_plan, best_cost = engine.run(max_iterations=1)

        # Check object identity using assertIs
        self.assertIs(best_plan, self.initial_plan)
        self.assertEqual(best_cost, self.initial_cost)

    def test_keyboard_interrupt_is_not_masked(self):
        """KeyboardInterrupt raised during exact audit is NOT caught or masked by _audit_state_exact."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-INTERRUPT")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        # Materializer raises KeyboardInterrupt
        engine.materializer.materialize = MagicMock(side_effect=KeyboardInterrupt("SIGINT"))

        with self.assertRaises(KeyboardInterrupt):
            engine.run(max_iterations=1)

    def test_system_exit_is_not_masked(self):
        """SystemExit raised during exact audit is NOT caught or masked by _audit_state_exact."""
        engine = ALNSEngine(
            problem=self.problem,
            policy=self.policy,
            initial_state=self.state_init,
            initial_plan=self.initial_plan,
            initial_cost=self.initial_cost,
            seed=42,
        )

        state_cand = make_dummy_state("PART-EXIT")
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 5000.0 if st == state_cand else 10000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_cand]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        # Validator raises SystemExit
        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(side_effect=SystemExit(1))

        with self.assertRaises(SystemExit):
            engine.run(max_iterations=1)


if __name__ == "__main__":
    unittest.main()
