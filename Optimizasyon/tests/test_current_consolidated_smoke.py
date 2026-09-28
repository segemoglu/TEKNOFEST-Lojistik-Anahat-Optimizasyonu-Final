"""Current-data comparison smoke test for two direct baseline strategies."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import (
    ConservativeDirectBaselineBuilder,
    GreedySameODConsolidatedBuilder,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CurrentConsolidatedSmokeTests(unittest.TestCase):
    """Compare both builders on one real forecast without fixed result counts."""

    def test_consolidated_plan_is_valid_and_strictly_smaller_and_cheaper(self) -> None:
        """Real same-O-D consolidation must beat the conservative fallback."""

        loader = ExcelProblemLoader(project_root=REPOSITORY_ROOT)
        static = loader.load_static_problem_data()
        forecast_path = configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=REPOSITORY_ROOT,
        )
        demands = loader.load_demands_from_workbook(
            forecast_path,
            static_data=static,
        )
        problem = ProblemData(static, demands)
        policy = EvaluationPolicy.from_manifest(loader.manifest)
        conservative = ConservativeDirectBaselineBuilder(problem, policy).build()
        consolidated = GreedySameODConsolidatedBuilder(problem, policy).build()

        self.assertTrue(conservative.validation_report.is_valid)
        self.assertTrue(consolidated.validation_report.is_valid)
        self.assertEqual(
            consolidated.plan.demand_parts,
            conservative.plan.demand_parts,
        )
        self.assertLess(
            consolidated.spot_vehicle_count,
            conservative.spot_vehicle_count,
        )
        self.assertLess(
            consolidated.total_cost_tl,
            conservative.total_cost_tl,
        )


if __name__ == "__main__":
    unittest.main()
