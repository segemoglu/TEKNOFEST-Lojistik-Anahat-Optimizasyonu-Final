"""Calibration and exact audit tests for ProxyEvaluator on real forecast data."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.alns.engine import ALNSEngine
from src.alns.evaluator import ProxyEvaluator
from src.alns.materializer import ALNSMaterializer
from src.alns.state import ALNSSolutionState
from src.domain import ProblemData
from src.evaluation import CostCalculator, EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import GreedySameOriginMultiStopBuilder, GreedySameOriginTwoStopBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ProxyCalibrationTests(unittest.TestCase):
    """Test suite for ProxyEvaluator numerical calibration against exact spot costs."""

    @classmethod
    def setUpClass(cls):
        loader = ExcelProblemLoader(project_root=REPOSITORY_ROOT)
        cls.static = loader.load_static_problem_data()
        forecast_path = configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=REPOSITORY_ROOT,
        )
        cls.demands = loader.load_demands_from_workbook(forecast_path, static_data=cls.static)
        cls.problem = ProblemData(cls.static, cls.demands)
        cls.policy = EvaluationPolicy.from_manifest(loader.manifest)

        # Build 6E Plan
        base_plan = GreedySameOriginTwoStopBuilder(cls.problem, cls.policy).build()
        result_6E = GreedySameOriginMultiStopBuilder(cls.problem, cls.policy).build_from_two_stop(base_plan)
        cls.plan_6E = result_6E.plan

        cls.materializer = ALNSMaterializer(cls.problem, cls.policy)
        cls.state_6E = cls.materializer.extract_state(cls.plan_6E)

    def test_proxy_state_calibrated_against_exact_spot_cost(self):
        """Proxy state cost must match exact variable spot cost within 0.5% relative error on 6E dataset."""
        proxy = ProxyEvaluator(self.problem, self.policy, self.state_6E)

        # 1. Proxy Cost
        proxy_cost = proxy.evaluate_state(self.state_6E)

        # 2. Individual route sum matching
        route_sum = sum(
            proxy.evaluate_route(r.service_date, r.stops, r.vehicle_type)
            for r in self.state_6E.spot_routes
        )
        self.assertAlmostEqual(proxy_cost, route_sum, places=4)

        # 3. Exact Spot Cost Breakdown
        calculator = CostCalculator(self.problem, self.policy)
        breakdown = calculator.calculate(self.plan_6E)

        spot_vehicle_cost = sum(
            line.total_cost_tl
            for line in breakdown.vehicle_cost_lines
            if line.ownership_type == "spot"
        )
        spot_part_ids = {
            p.part_id
            for r in self.state_6E.spot_routes
            for s in r.stops
            for p in s.pickups
        }
        spot_sla_cost = sum(
            line.penalty_tl
            for line in breakdown.sla_penalty_lines
            if line.demand_part_id in spot_part_ids
        )
        exact_spot_cost = spot_vehicle_cost + spot_sla_cost

        diff_tl = abs(proxy_cost - exact_spot_cost)
        rel_diff = diff_tl / exact_spot_cost

        print("\n--- Proxy Calibration Results ---")
        print(f"Proxy State Cost        : {proxy_cost:,.4f} TL")
        print(f"Exact Variable Spot Cost: {exact_spot_cost:,.4f} TL")
        print(f"TL Difference           : {diff_tl:,.4f} TL")
        print(f"Relative Difference     : {rel_diff * 100:.4f}%\n")

        # Must not spike to artificial ~36M
        self.assertLess(proxy_cost, 20_000_000.0)

        # Relative error must not exceed 0.5% (0.005)
        self.assertLessEqual(rel_diff, 0.005)

    def test_invalid_truck_handling_candidate_cannot_become_exact_incumbent(self):
        """Exact validator rejecting a candidate due to truck/handling limits prevents it from becoming exact incumbent."""
        engine = ALNSEngine(
            self.problem,
            self.policy,
            self.state_6E,
            self.plan_6E,
            15_956_118.3363,
            seed=42,
        )

        invalid_candidate = self.state_6E
        engine.materializer.materialize = MagicMock(return_value=MagicMock())
        engine.validator.validate = MagicMock(
            return_value=MagicMock(is_valid=False, errors=["Daily truck quota exceeded at OriginA"])
        )

        audit_result = engine._audit_state_exact(invalid_candidate)

        self.assertFalse(audit_result)
        self.assertIs(engine.best_ever_plan, self.plan_6E)
        self.assertEqual(engine.best_ever_cost, 15_956_118.3363)


if __name__ == "__main__":
    unittest.main()
