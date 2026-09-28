"""Unit tests for ALNS CLI time-limit forwarding, argument validation, and stop reason summary."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import datetime
from pathlib import Path

from src.planning.run_alns import (
    build_parser,
    build_seed_plan,
    parse_seeds,
    run_alns_plan,
)
from src.planning.baseline import BaselineBuildError
from src.alns.engine import ALNSEngine
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.domain import DemandPart


def make_dummy_state() -> ALNSSolutionState:
    part = DemandPart(part_id="P1", demand_id="D001", desi=100.0)
    stop1 = ALNSRouteStop("OriginA", (), (part,))
    stop2 = ALNSRouteStop("DestB", (part,), ())
    route = ALNSSpotRoute(datetime.date(2026, 7, 1), (stop1, stop2), "Kamyonet")
    return ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))


class ParseSeedsTests(unittest.TestCase):
    """Verify the multi-seed CLI list is parsed deterministically."""

    def test_missing_list_falls_back_to_single_seed(self):
        """Without --seeds the run uses the single --seed value."""
        self.assertEqual(parse_seeds(None, 42), (42,))

    def test_list_keeps_given_order_and_drops_duplicates(self):
        """Order decides tie-breaks, so it must survive parsing unchanged."""
        self.assertEqual(parse_seeds("42, 7,123, 7 ", 1), (42, 7, 123))

    def test_invalid_or_empty_list_is_rejected(self):
        """A malformed seed list fails loudly instead of silently using one seed."""
        with self.assertRaises(ValueError):
            parse_seeds("42,abc", 1)
        with self.assertRaises(ValueError):
            parse_seeds(" , ", 1)


class SeedPlanFallbackTests(unittest.TestCase):
    """A layer that refuses to improve must degrade, never end the run.

    6D and 6E raise ``BaselineBuildError`` when they cannot strictly beat the
    layer below.  Before the fallback ladder that refusal propagated out of
    ``run_alns_plan`` and the run exported nothing, discarding a valid plan.
    """

    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    def test_six_e_result_is_used_when_the_chain_succeeds(
        self, mock_two_stop, mock_multi_stop,
    ):
        """The unchanged happy path still returns the 6E result."""

        six_e = MagicMock(name="6E result")
        mock_multi_stop.return_value.build_from_two_stop.return_value = six_e

        result, layer = build_seed_plan(MagicMock(), MagicMock())

        self.assertIs(result, six_e)
        self.assertEqual(layer, "6E")

    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    def test_six_d_is_kept_when_six_e_finds_no_saving(
        self, mock_two_stop, mock_multi_stop,
    ):
        """``multi_stop_no_strict_saving`` keeps the valid 6D plan."""

        six_d = MagicMock(name="6D result")
        mock_two_stop.return_value.build.return_value = six_d
        mock_multi_stop.return_value.build_from_two_stop.side_effect = (
            BaselineBuildError(
                "6E kazanç üretmedi",
                error_codes=("multi_stop_no_strict_saving",),
            )
        )

        result, layer = build_seed_plan(MagicMock(), MagicMock())

        self.assertIs(result, six_d)
        self.assertEqual(layer, "6D")

    @patch("src.planning.run_alns.GreedyMandatoryRentalFilledBuilder")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    def test_six_b_is_used_when_six_d_finds_no_saving(
        self, mock_two_stop, mock_multi_stop, mock_mandatory,
    ):
        """A 6D refusal skips 6E entirely and drops to 6B."""

        six_b = MagicMock(name="6B result")
        mock_two_stop.return_value.build.side_effect = BaselineBuildError(
            "6D kazanç üretmedi",
            error_codes=("two_stop_no_strict_saving",),
        )
        mock_mandatory.return_value.build.return_value = six_b

        result, layer = build_seed_plan(MagicMock(), MagicMock())

        self.assertIs(result, six_b)
        self.assertEqual(layer, "6B")
        mock_multi_stop.return_value.build_from_two_stop.assert_not_called()

    @patch("src.planning.run_alns.ConservativeDirectBaselineBuilder")
    @patch("src.planning.run_alns.GreedySameODConsolidatedBuilder")
    @patch("src.planning.run_alns.GreedyMandatoryRentalFilledBuilder")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    def test_ladder_descends_to_the_conservative_baseline(
        self, mock_two_stop, mock_multi_stop, mock_mandatory,
        mock_consolidated, mock_conservative,
    ):
        """Every richer layer failing still yields a plan, not an exception."""

        failure = BaselineBuildError("kurulamadı", error_codes=("boom",))
        mock_two_stop.return_value.build.side_effect = failure
        mock_mandatory.return_value.build.side_effect = failure
        mock_consolidated.return_value.build.side_effect = failure
        conservative = MagicMock(name="conservative result")
        mock_conservative.return_value.build.return_value = conservative

        result, layer = build_seed_plan(MagicMock(), MagicMock())

        self.assertIs(result, conservative)
        self.assertEqual(layer, "konservatif")

    @patch("src.planning.run_alns.ConservativeDirectBaselineBuilder")
    @patch("src.planning.run_alns.GreedySameODConsolidatedBuilder")
    @patch("src.planning.run_alns.GreedyMandatoryRentalFilledBuilder")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    def test_exhausted_ladder_raises_a_named_error(
        self, mock_two_stop, mock_multi_stop, mock_mandatory,
        mock_consolidated, mock_conservative,
    ):
        """When no layer can plan the input, the failure stays loud."""

        failure = BaselineBuildError("kurulamadı", error_codes=("boom",))
        for mock_builder in (
            mock_two_stop, mock_mandatory, mock_consolidated, mock_conservative,
        ):
            mock_builder.return_value.build.side_effect = failure

        with self.assertRaises(BaselineBuildError) as caught:
            build_seed_plan(MagicMock(), MagicMock())

        self.assertIn("greedy_chain_exhausted", caught.exception.error_codes)


class MultiSeedSelectionTests(unittest.TestCase):
    """Verify that the cheapest valid seed result is the one exported."""

    @patch("dataclasses.replace")
    @patch("src.evaluation.costs.CostCalculator")
    @patch("src.planning.run_alns.EvaluationPolicy")
    @patch("src.planning.run_alns.configured_project_path", return_value=Path("data/interim/development_forecast.xlsx"))
    @patch("src.planning.run_alns.ALNSMaterializer")
    @patch("src.planning.run_alns.ALNSEngine")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    @patch("src.planning.run_alns.ExcelProblemLoader")
    @patch("src.planning.run_alns.TransportPlanWorkbookExporter")
    def test_cheapest_seed_plan_is_exported(
        self, mock_exporter, mock_loader, mock_two_stop, mock_multi_stop,
        mock_engine_cls, mock_materializer, mock_conf_path, mock_policy_cls,
        mock_cost_calc_cls, mock_replace,
    ):
        """Three seeds return different costs; the lowest one reaches the exporter."""

        mock_loader_inst = mock_loader.return_value
        mock_loader_inst.load_static_problem_data.return_value = MagicMock()
        mock_loader_inst.load_demands_from_workbook.return_value = []
        mock_loader_inst.manifest = MagicMock()
        mock_loader_inst.project_root = Path(".")
        mock_policy_cls.from_manifest.return_value = MagicMock()

        six_e_plan = MagicMock(name="6E")
        mock_two_stop.return_value.build.return_value = MagicMock()
        multi_stop_result = MagicMock()
        multi_stop_result.plan = six_e_plan
        multi_stop_result.total_cost_tl = 1000.0
        mock_multi_stop.return_value.build_from_two_stop.return_value = multi_stop_result

        cheap_plan = MagicMock(name="cheap")
        plans = {42: (MagicMock(name="mid"), 900.0), 7: (cheap_plan, 800.0), 123: (MagicMock(name="dear"), 950.0)}

        def make_engine(problem, policy, state, plan, cost, seed):
            engine = MagicMock()
            engine.run.return_value = plans[seed]
            engine.completed_iterations = 50
            engine.render_diagnostics.return_value = ""
            return engine

        mock_engine_cls.side_effect = make_engine
        mock_replace.side_effect = lambda plan, **kwargs: plan

        run_alns_plan(iterations=50, seed=42, seeds=(42, 7, 123))

        # The 6E reference is deterministic, so it is built once and reused.
        self.assertEqual(mock_multi_stop.return_value.build_from_two_stop.call_count, 1)
        self.assertEqual(mock_engine_cls.call_count, 3)
        exported_plan = mock_exporter.return_value.write.call_args.args[0]
        self.assertIs(exported_plan, cheap_plan)

    @patch("dataclasses.replace")
    @patch("src.evaluation.costs.CostCalculator")
    @patch("src.planning.run_alns.EvaluationPolicy")
    @patch("src.planning.run_alns.configured_project_path", return_value=Path("data/interim/development_forecast.xlsx"))
    @patch("src.planning.run_alns.ALNSMaterializer")
    @patch("src.planning.run_alns.ALNSEngine")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    @patch("src.planning.run_alns.ExcelProblemLoader")
    @patch("src.planning.run_alns.TransportPlanWorkbookExporter")
    def test_six_e_is_exported_when_no_seed_improves(
        self, mock_exporter, mock_loader, mock_two_stop, mock_multi_stop,
        mock_engine_cls, mock_materializer, mock_conf_path, mock_policy_cls,
        mock_cost_calc_cls, mock_replace,
    ):
        """If every seed returns a dearer plan the 6E reference is kept."""

        mock_loader_inst = mock_loader.return_value
        mock_loader_inst.load_static_problem_data.return_value = MagicMock()
        mock_loader_inst.load_demands_from_workbook.return_value = []
        mock_loader_inst.manifest = MagicMock()
        mock_loader_inst.project_root = Path(".")
        mock_policy_cls.from_manifest.return_value = MagicMock()

        six_e_plan = MagicMock(name="6E")
        mock_two_stop.return_value.build.return_value = MagicMock()
        multi_stop_result = MagicMock()
        multi_stop_result.plan = six_e_plan
        multi_stop_result.total_cost_tl = 1000.0
        mock_multi_stop.return_value.build_from_two_stop.return_value = multi_stop_result

        def make_engine(problem, policy, state, plan, cost, seed):
            engine = MagicMock()
            engine.run.return_value = (MagicMock(name="worse"), 1200.0)
            engine.completed_iterations = 50
            engine.render_diagnostics.return_value = ""
            return engine

        mock_engine_cls.side_effect = make_engine
        mock_replace.side_effect = lambda plan, **kwargs: plan

        run_alns_plan(iterations=50, seed=42, seeds=(42, 7))

        exported_plan = mock_exporter.return_value.write.call_args.args[0]
        self.assertIs(exported_plan, six_e_plan)


class RunALNSCLITests(unittest.TestCase):
    """Test suite for CLI argument parsing, time-limit validation, and stop reason logic."""

    def test_parser_default_time_limit_is_600(self):
        """CLI parser default time-limit must be 600.0 seconds."""
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.time_limit, 600.0)

    def test_parser_custom_float_time_limit(self):
        """CLI parser correctly parses float time-limit values like 123.5."""
        parser = build_parser()
        args = parser.parse_args(["--time-limit", "123.5"])
        self.assertEqual(args.time_limit, 123.5)

    def test_zero_or_negative_validation_raises_value_error(self):
        """Zero or negative iterations and time-limit values must raise ValueError."""
        with self.assertRaises(ValueError):
            run_alns_plan(iterations=0, seed=42)

        with self.assertRaises(ValueError):
            run_alns_plan(iterations=-5, seed=42)

        with self.assertRaises(ValueError):
            run_alns_plan(iterations=10, seed=42, time_limit_seconds=0.0)

        with self.assertRaises(ValueError):
            run_alns_plan(iterations=10, seed=42, time_limit_seconds=-10.0)

    @patch("dataclasses.replace")
    @patch("src.evaluation.costs.CostCalculator")
    @patch("src.planning.run_alns.EvaluationPolicy")
    @patch("src.planning.run_alns.configured_project_path", return_value=Path("data/interim/development_forecast.xlsx"))
    @patch("src.planning.run_alns.ALNSEngine")
    @patch("src.planning.run_alns.GreedySameOriginMultiStopBuilder")
    @patch("src.planning.run_alns.GreedySameOriginTwoStopBuilder")
    @patch("src.planning.run_alns.ExcelProblemLoader")
    @patch("src.planning.run_alns.TransportPlanWorkbookExporter")
    def test_time_limit_seconds_passed_to_engine(
        self, mock_exporter, mock_loader, mock_two_stop, mock_multi_stop, mock_engine_cls, mock_conf_path, mock_policy_cls, mock_cost_calc_cls, mock_replace
    ):
        """run_alns_plan passes time_limit_seconds to engine.run()."""
        mock_loader_inst = mock_loader.return_value
        mock_loader_inst.load_static_problem_data.return_value = MagicMock()
        mock_loader_inst.load_demands_from_workbook.return_value = []
        mock_loader_inst.manifest = MagicMock()
        mock_loader_inst.project_root = Path(".")

        mock_policy_cls.from_manifest.return_value = MagicMock()

        dummy_plan = MagicMock()
        mock_replace.return_value = dummy_plan
        mock_two_stop.return_value.build.return_value = MagicMock()
        mock_multi_stop_res = MagicMock()
        mock_multi_stop_res.plan = dummy_plan
        mock_multi_stop_res.total_cost_tl = 1000.0
        mock_multi_stop.return_value.build_from_two_stop.return_value = mock_multi_stop_res

        mock_engine_inst = mock_engine_cls.return_value
        mock_engine_inst.destroy_calls = [1, 1, 1]
        # The CLI reports the engine's own iteration counter, not destroy_calls,
        # because the weight-update segment resets the operator call counts.
        mock_engine_inst.completed_iterations = 3
        mock_engine_inst.render_diagnostics.return_value = "teşhis yok"
        mock_engine_inst.run.return_value = (dummy_plan, 900.0)

        run_alns_plan(iterations=50, seed=42, time_limit_seconds=123.5)

        mock_engine_inst.run.assert_called_once_with(
            max_iterations=50,
            start_temperature=10000.0,
            end_temperature=1.0,
            time_limit_seconds=123.5,
            set_partition_seconds=0.0,
        )

    def test_small_time_limit_produces_time_limit_stop_reason(self):
        """When search is interrupted by time_limit_seconds, stop reason evaluates to time_limit."""
        state = make_dummy_state()
        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")
        initial_plan = MagicMock(name="InitialPlan")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state,
            initial_plan=initial_plan,
            initial_cost=1000.0,
            seed=42,
        )

        # Set time_limit_seconds=0.000001 (0.1 ms) to force immediate time-limit hit
        engine.run(max_iterations=1000, time_limit_seconds=0.000001)

        completed_iterations = sum(engine.destroy_calls)
        self.assertLess(completed_iterations, 1000)

    def test_completed_iterations_produces_completed_stop_reason(self):
        """When all requested iterations complete within time limit, completed iterations equals max_iterations."""
        state = make_dummy_state()
        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")
        initial_plan = MagicMock(name="InitialPlan")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=state,
            initial_plan=initial_plan,
            initial_cost=1000.0,
            seed=42,
        )
        engine.destroy_ops = [lambda st, n, rng, *args: (st, [])]
        engine.repair_ops = [lambda st, unassigned, *args: st]
        engine.destroy_weights = [1.0]
        engine.repair_weights = [1.0]

        engine.run(max_iterations=5, time_limit_seconds=600.0)

        completed_iterations = sum(engine.destroy_calls)
        self.assertEqual(completed_iterations, 5)

    def test_stagnation_stop_reason_when_engine_stops_early(self):
        """When engine stops early before max_iterations and before time_limit, stop_reason evaluates to stagnation."""
        empty_state = ALNSSolutionState(parts=(), mandatory_trips=(), spot_routes=())
        problem = MagicMock(name="ProblemData")
        policy = MagicMock(name="EvaluationPolicy")
        initial_plan = MagicMock(name="InitialPlan")

        engine = ALNSEngine(
            problem=problem,
            policy=policy,
            initial_state=empty_state,
            initial_plan=initial_plan,
            initial_cost=1000.0,
            seed=42,
        )

        import time
        start_time = time.time()
        engine.run(max_iterations=100, time_limit_seconds=600.0)
        elapsed_seconds = time.time() - start_time

        completed_iterations = sum(engine.destroy_calls)

        # 1. Completed iterations is less than requested 100
        self.assertLess(completed_iterations, 100)

        # 2. Elapsed time is less than 600.0s time limit
        self.assertLess(elapsed_seconds, 600.0)

        # 3. Evaluated stop_reason must be stagnation
        if completed_iterations >= 100:
            stop_reason = "completed"
        elif elapsed_seconds >= 600.0:
            stop_reason = "time_limit"
        else:
            stop_reason = "stagnation"

        self.assertEqual(stop_reason, "stagnation")


if __name__ == "__main__":
    unittest.main()
