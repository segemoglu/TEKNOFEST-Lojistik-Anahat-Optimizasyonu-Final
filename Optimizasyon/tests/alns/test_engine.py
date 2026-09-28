"""Test the ALNS engine with basic operators."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy, CostCalculator
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import GreedySameOriginTwoStopBuilder, GreedySameOriginMultiStopBuilder
from src.alns.materializer import ALNSMaterializer
from src.alns.engine import ALNSEngine

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ALNSEngineTests(unittest.TestCase):
    """Test ALNS engine integration."""

    def test_engine_runs_without_crashing_and_preserves_incumbent(self) -> None:
        """Run a short 10-iteration loop; ensure valid output and no incumbent regression."""
        
        # 1. Load Problem
        loader = ExcelProblemLoader(project_root=REPOSITORY_ROOT)
        static = loader.load_static_problem_data()
        forecast_path = configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=REPOSITORY_ROOT,
        )
        demands = loader.load_demands_from_workbook(forecast_path, static_data=static)
        problem = ProblemData(static, demands)
        policy = EvaluationPolicy.from_manifest(loader.manifest)

        # 2. Build 6E Plan
        print("Building initial 6D plan...")
        base = GreedySameOriginTwoStopBuilder(problem, policy).build()
        print("Building initial 6E plan...")
        result = GreedySameOriginMultiStopBuilder(problem, policy).build_from_two_stop(base)
        original_plan = result.plan
        original_cost = result.total_cost_tl
        print(f"Original 6E Cost: {original_cost:,.4f} TL")

        # 3. Setup ALNS Engine
        materializer = ALNSMaterializer(problem, policy)
        initial_state = materializer.extract_state(original_plan)
        
        engine = ALNSEngine(
            problem, 
            policy, 
            initial_state,
            original_plan,
            original_cost, 
            seed=42
        )
        
        # Turn on logging to see Simulated Annealing accepts
        logging.basicConfig(level=logging.DEBUG)
        
        # 4. Run ALNS (short run)
        best_plan, best_cost = engine.run(
            max_iterations=10, 
            start_temperature=10000.0,
            end_temperature=1.0
        )
        
        print(f"Best Cost after 10 iterations: {best_cost:,.4f} TL")
        
        # 5. Assertions
        # It should not regress below the initial incumbent (thanks to tracking best_ever)
        self.assertLessEqual(best_cost, original_cost, "Engine lost the incumbent!")
        
        # The returned best_state must actually evaluate exactly to best_cost
        best_plan = engine.best_ever_plan
        calc = CostCalculator(problem, policy)
        calculated_cost = calc.calculate(best_plan).total_cost_tl
        self.assertAlmostEqual(
            calculated_cost, 
            best_cost, 
            places=2, 
            msg="Materialized cost of best_state does not match reported best_cost!"
        )
