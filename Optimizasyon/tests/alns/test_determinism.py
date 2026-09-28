"""Unit tests for ALNS state canonical signature and cross-process determinism."""

from __future__ import annotations

import os
import sys
import subprocess
import unittest
from pathlib import Path

from src.domain import ProblemData, DemandPart
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.evaluator import ProxyEvaluator
from src.alns.destroy import random_removal, worst_removal, shaw_removal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DeterminismTests(unittest.TestCase):
    """Test suite for state signature canonicalization and cross-process determinism."""

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

    def setUp(self):
        self.demand = self.demands[0]
        self.part1 = DemandPart(part_id=f"{self.demand.demand_id}-1", demand_id=self.demand.demand_id, desi=100.0)
        self.part2 = DemandPart(part_id=f"{self.demand.demand_id}-2", demand_id=self.demand.demand_id, desi=200.0)

        stop1 = ALNSRouteStop(self.demand.origin, (), (self.part1,))
        stop2 = ALNSRouteStop(self.demand.destination, (self.part1,), ())
        self.route1 = ALNSSpotRoute(self.demand.demand_date, (stop1, stop2), "Kamyon")

        self.state1 = ALNSSolutionState(
            parts=(self.part1, self.part2),
            mandatory_trips=(),
            spot_routes=(self.route1,),
        )

    def test_same_state_same_signature_twice(self):
        """Calling signature() twice on the same state produces the identical output."""
        sig1 = self.state1.signature()
        sig2 = self.state1.signature()
        self.assertEqual(sig1, sig2)

    def test_different_parts_same_structure_different_signature(self):
        """States with identical route/stop structure but different parts produce different signatures."""
        stop1_alt = ALNSRouteStop(self.demand.origin, (), (self.part2,))
        stop2_alt = ALNSRouteStop(self.demand.destination, (self.part2,), ())
        route1_alt = ALNSSpotRoute(self.demand.demand_date, (stop1_alt, stop2_alt), "Kamyon")

        state_alt = ALNSSolutionState(
            parts=(self.part1, self.part2),
            mandatory_trips=(),
            spot_routes=(route1_alt,),
        )

        self.assertNotEqual(self.state1.signature(), state_alt.signature())

    def test_route_order_change_changes_signature(self):
        """Changing the order of spot routes produces a different signature."""
        other_demand = [d for d in self.demands if d.demand_id != self.demand.demand_id][0]
        other_part = DemandPart(part_id=f"{other_demand.demand_id}-1", demand_id=other_demand.demand_id, desi=50.0)
        
        stop_o1 = ALNSRouteStop(other_demand.origin, (), (other_part,))
        stop_o2 = ALNSRouteStop(other_demand.destination, (other_part,), ())
        route2 = ALNSSpotRoute(other_demand.demand_date, (stop_o1, stop_o2), "Kamyonet")

        state_order_a = ALNSSolutionState(
            parts=(self.part1, other_part),
            mandatory_trips=(),
            spot_routes=(self.route1, route2),
        )
        state_order_b = ALNSSolutionState(
            parts=(self.part1, other_part),
            mandatory_trips=(),
            spot_routes=(route2, self.route1),
        )

        self.assertNotEqual(state_order_a.signature(), state_order_b.signature())

    def test_subprocess_determinism_across_hash_seeds(self):
        """Subprocesses with PYTHONHASHSEED=1 and PYTHONHASHSEED=999 produce identical signature."""
        code = (
            "from pathlib import Path; "
            "from src.domain import ProblemData, DemandPart; "
            "from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop; "
            "from src.io import ExcelProblemLoader, configured_project_path; "
            "loader = ExcelProblemLoader(project_root=Path('.')); "
            "static = loader.load_static_problem_data(); "
            "forecast_path = configured_project_path(loader.manifest, 'development_forecast', project_root=Path('.')); "
            "demands = loader.load_demands_from_workbook(forecast_path, static_data=static); "
            "d = demands[0]; "
            "p = DemandPart(part_id=f'{d.demand_id}-1', demand_id=d.demand_id, desi=100.0); "
            "s1 = ALNSRouteStop(d.origin, (), (p,)); "
            "s2 = ALNSRouteStop(d.destination, (p,), ()); "
            "r = ALNSSpotRoute(d.demand_date, (s1, s2), 'Kamyon'); "
            "st = ALNSSolutionState(parts=(p,), mandatory_trips=(), spot_routes=(r,)); "
            "print(st.signature())"
        )

        env1 = os.environ.copy()
        env1["PYTHONHASHSEED"] = "1"
        res1 = subprocess.check_output([sys.executable, "-c", code], env=env1, cwd=REPOSITORY_ROOT, text=True).strip()

        env2 = os.environ.copy()
        env2["PYTHONHASHSEED"] = "999"
        res2 = subprocess.check_output([sys.executable, "-c", code], env=env2, cwd=REPOSITORY_ROOT, text=True).strip()

        self.assertEqual(res1, res2)

    def test_destroy_operators_deterministic_unassigned_order(self):
        """Destroy operators with fixed seed produce deterministic unassigned part ordering."""
        import random
        
        proxy = ProxyEvaluator(self.problem, self.policy, self.state1)
        
        # Test random_removal
        rng1 = random.Random(42)
        _, unassigned1_a = random_removal(self.state1, 1, rng1)
        rng2 = random.Random(42)
        _, unassigned1_b = random_removal(self.state1, 1, rng2)
        self.assertEqual(
            [(p[0].demand_id, p[0].part_id) for p in unassigned1_a],
            [(p[0].demand_id, p[0].part_id) for p in unassigned1_b],
        )

        # Test worst_removal
        rng3 = random.Random(42)
        _, unassigned2_a = worst_removal(self.state1, 1, rng3, self.problem, self.policy, proxy)
        rng4 = random.Random(42)
        _, unassigned2_b = worst_removal(self.state1, 1, rng4, self.problem, self.policy, proxy)
        self.assertEqual(
            [(p[0].demand_id, p[0].part_id) for p in unassigned2_a],
            [(p[0].demand_id, p[0].part_id) for p in unassigned2_b],
        )


if __name__ == "__main__":
    unittest.main()
