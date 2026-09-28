"""Synthetic unit tests for ALNS candidate archive and exact audit policy."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import datetime

from src.alns.engine import ALNSEngine
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.candidate_archive import CandidateArchive
from src.domain import DemandPart


def make_dummy_state(part_id: str) -> ALNSSolutionState:
    part = DemandPart(part_id=part_id, demand_id="D001", desi=100.0)
    stop1 = ALNSRouteStop("OriginA", (), (part,))
    stop2 = ALNSRouteStop("DestB", (part,), ())
    route = ALNSSpotRoute(datetime.date(2026, 7, 1), (stop1, stop2), "Kamyonet")
    return ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))


class ExactCandidateArchiveTests(unittest.TestCase):
    """Test suite for CandidateArchive and exact-audit policy in ALNSEngine."""

    def test_archive_respects_capacity_limit(self):
        """CandidateArchive capacity K limit is strictly enforced."""
        archive = CandidateArchive(capacity=3)
        for i in range(10):
            st = make_dummy_state(f"PART-{i}")
            archive.add(st, proxy_cost=1000.0 + i * 10)

        self.assertEqual(len(archive), 3)
        candidates = archive.get_candidates()
        # Should keep top 3 lowest proxy costs (1000.0, 1010.0, 1020.0)
        self.assertEqual([c[0] for c in candidates], [1000.0, 1010.0, 1020.0])

    def test_mandatory_synthetic_scenario_b_wins_over_a(self):
        """Scenario: A is proxy-best (10.0M, exact 13.9M), B is archive candidate (10.1M, exact 13.8M). Final result is B."""
        state_init = make_dummy_state("PART-INIT")
        state_A = make_dummy_state("PART-A")
        state_B = make_dummy_state("PART-B")

        initial_plan = MagicMock(name="InitialPlan")
        initial_cost = 15000000.0

        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_init,
            initial_plan=initial_plan,
            initial_cost=initial_cost,
            archive_capacity=8,
            exact_check_interval=25,
            seed=42,
        )

        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: {
            state_init: 12000000.0,
            state_A: 10000000.0,
            state_B: 10100000.0,
        }.get(st, 12000000.0))

        step_counter = 0
        def mock_destroy(st, n, rng, *args):
            return st, []

        def mock_repair(st, unassigned, *args):
            nonlocal step_counter
            step_counter += 1
            if step_counter == 1:
                return state_A
            return state_B

        engine.destroy_ops = [mock_destroy]
        engine.repair_ops = [mock_repair]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        plan_A = MagicMock(name="PlanA")
        plan_B = MagicMock(name="PlanB")

        def mock_materialize(st):
            if st == state_A:
                return plan_A
            if st == state_B:
                return plan_B
            return MagicMock(name="PlanInit")

        engine.materializer.materialize = MagicMock(side_effect=mock_materialize)
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))

        def mock_calculate(plan):
            if plan == plan_A:
                return MagicMock(total_cost_tl=13900000.0)
            if plan == plan_B:
                return MagicMock(total_cost_tl=13800000.0)
            return MagicMock(total_cost_tl=15000000.0)

        engine.calculator.calculate = MagicMock(side_effect=mock_calculate)

        # Run 2 iterations
        best_plan, best_cost = engine.run(max_iterations=2)

        # B is non-proxy-record candidate in archive, but wins in exact audit (13.8M vs 13.9M)
        self.assertEqual(best_plan, plan_B)
        self.assertEqual(best_cost, 13800000.0)

    def test_proxy_best_invalid_second_candidate_valid_wins(self):
        """When proxy-best candidate is invalid, the second valid candidate in archive is chosen."""
        state_init = make_dummy_state("PART-INIT")
        state_A = make_dummy_state("PART-A")
        state_B = make_dummy_state("PART-B")

        initial_plan = MagicMock(name="InitialPlan")
        initial_cost = 15000000.0

        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_init,
            initial_plan=initial_plan,
            initial_cost=initial_cost,
            archive_capacity=8,
            exact_check_interval=25,
            seed=42,
        )

        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: {
            state_init: 12000000.0,
            state_A: 9000000.0,
            state_B: 9500000.0,
        }.get(st, 12000000.0))

        step_counter = 0
        def mock_destroy(st, n, rng, *args):
            return st, []

        def mock_repair(st, unassigned, *args):
            nonlocal step_counter
            step_counter += 1
            if step_counter == 1:
                return state_A
            return state_B

        engine.destroy_ops = [mock_destroy]
        engine.repair_ops = [mock_repair]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        plan_A = MagicMock(name="PlanA")
        plan_B = MagicMock(name="PlanB")

        engine.materializer.materialize = MagicMock(side_effect=lambda st: plan_A if st == state_A else plan_B)

        def mock_validate(plan):
            if plan == plan_A:
                return MagicMock(is_valid=False, errors=["Over SLA limit"])
            return MagicMock(is_valid=True)

        engine.validator.validate = MagicMock(side_effect=mock_validate)
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=14000000.0))

        best_plan, best_cost = engine.run(max_iterations=2)

        # Plan A is invalid, plan B is valid and accepted
        self.assertEqual(best_plan, plan_B)
        self.assertEqual(best_cost, 14000000.0)

    def test_duplicate_states_not_re_audited(self):
        """Duplicate candidate states are not re-audited multiple times."""
        state_init = make_dummy_state("PART-INIT")
        state_A = make_dummy_state("PART-A")

        initial_plan = MagicMock(name="InitialPlan")
        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_init,
            initial_plan=initial_plan,
            initial_cost=15000000.0,
            archive_capacity=8,
            exact_check_interval=1,
            seed=42,
        )

        engine.proxy.evaluate_state = MagicMock(return_value=10000000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_A]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        plan_A = MagicMock(name="PlanA")
        engine.materializer.materialize = MagicMock(return_value=plan_A)
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=14000000.0))

        # Run 5 iterations where repair always produces state_A
        engine.run(max_iterations=5)

        # materialize should be called ONCE for state_A (because signature is cached in audited_signatures)
        self.assertEqual(engine.materializer.materialize.call_count, 1)

    def test_periodic_exact_check_interval_gate(self):
        """Exact audit validator is not called every iteration, but at exact_check_interval iterations."""
        state_init = make_dummy_state("PART-INIT")
        initial_plan = MagicMock(name="InitialPlan")
        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_init,
            initial_plan=initial_plan,
            initial_cost=15000000.0,
            archive_capacity=8,
            exact_check_interval=10, # Check every 10 iterations
            seed=42,
        )

        # Return worse proxy cost for candidates so proxy record trigger doesn't fire
        engine.proxy.evaluate_state = MagicMock(side_effect=lambda st: 12000000.0 if st == state_init else 14000000.0)
        engine.best_ever_proxy_cost = 12000000.0
        
        counter = 0
        def mock_repair(st, unassigned, *args):
            nonlocal counter
            counter += 1
            return make_dummy_state(f"PART-{counter}")

        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [mock_repair]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=14000000.0))

        # Run 5 iterations (less than exact_check_interval=10, so periodic audit won't fire during loop)
        engine.run(max_iterations=5)

        # Confirm that validator was not called on every loop iteration (only final sweep for archive)
        self.assertLessEqual(engine.validator.validate.call_count, 6)

    def test_no_improvement_preserves_initial_incumbent(self):
        """If no candidate improves the initial plan cost, initial plan and cost are preserved."""
        state_init = make_dummy_state("PART-INIT")
        initial_plan = MagicMock(name="InitialPlan")
        initial_cost = 10000000.0

        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state_init,
            initial_plan=initial_plan,
            initial_cost=initial_cost,
            archive_capacity=8,
            exact_check_interval=25,
            seed=42,
        )

        state_expensive = make_dummy_state("PART-EXPENSIVE")
        engine.proxy.evaluate_state = MagicMock(return_value=12000000.0)
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: state_expensive]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(return_value=MagicMock(is_valid=True))
        engine.calculator.calculate = MagicMock(return_value=MagicMock(total_cost_tl=12000000.0))

        best_plan, best_cost = engine.run(max_iterations=2)

        self.assertEqual(best_plan, initial_plan)
        self.assertEqual(best_cost, initial_cost)


if __name__ == "__main__":
    unittest.main()
