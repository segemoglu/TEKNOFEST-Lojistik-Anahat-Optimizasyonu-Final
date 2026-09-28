"""Unit tests for Regret-2 repair bug fix and controlled exception handling."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.domain import ProblemData, DemandPart
from src.domain.models import DemandInstance
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import GreedySameOriginTwoStopBuilder, GreedySameOriginMultiStopBuilder
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.evaluator import ProxyEvaluator
from src.alns.materializer import ALNSMaterializer
from src.alns.repair import greedy_insertion, regret_2_insertion
from src.alns.invariants import validate_alns_state
from src.alns.exceptions import RepairInfeasibleError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RegretRepairTests(unittest.TestCase):
    """Test suite for Regret-2 repair placement fix and RepairInfeasibleError handling."""

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
        self.part = DemandPart(
            part_id=f"{self.demand.demand_id}-1",
            demand_id=self.demand.demand_id,
            desi=100.0,
        )
        self.base_state = ALNSSolutionState(
            parts=(self.part,),
            mandatory_trips=(),
            spot_routes=(),
        )
        self.proxy = ProxyEvaluator(self.problem, self.policy, self.base_state)

    def test_empty_routes_creates_direct_route(self):
        """Regret-2 repair on empty spot routes creates a 2-stop direct route."""
        unassigned = [(self.part,)]
        repaired_state = regret_2_insertion(
            self.base_state, unassigned, self.problem, self.policy, self.proxy
        )
        self.assertEqual(len(repaired_state.spot_routes), 1)
        route = repaired_state.spot_routes[0]
        self.assertEqual(len(route.stops), 2)
        self.assertEqual(route.stops[0].center, self.demand.origin)
        self.assertEqual(route.stops[1].center, self.demand.destination)

    def test_no_matching_origin_destination_creates_direct_route(self):
        """Regret-2 creates a direct route when existing routes don't share origin/destination."""
        centers = (self.demand.origin, self.demand.destination)
        other_demand = [d for d in self.demands if d.origin not in centers and d.destination not in centers][0]
        other_part = DemandPart(part_id=f"{other_demand.demand_id}-1", demand_id=other_demand.demand_id, desi=100.0)
        
        stop1 = ALNSRouteStop(other_demand.origin, (), (other_part,))
        stop2 = ALNSRouteStop(other_demand.destination, (other_part,), ())
        existing_route = ALNSSpotRoute(other_demand.demand_date, (stop1, stop2), "Kamyon")
        
        state_with_existing = ALNSSolutionState(
            parts=(self.part, other_part),
            mandatory_trips=(),
            spot_routes=(existing_route,),
        )
        
        repaired_state = regret_2_insertion(
            state_with_existing, [(self.part,)], self.problem, self.policy, self.proxy
        )
        self.assertEqual(len(repaired_state.spot_routes), 2)
        self.assertTrue(validate_alns_state(repaired_state, self.problem, self.policy))

    def test_placed_part_has_complete_pickup_and_dropoff(self):
        """Part placed by regret_2_insertion has complete pickup and dropoff in the route."""
        unassigned = [(self.part,)]
        repaired_state = regret_2_insertion(
            self.base_state, unassigned, self.problem, self.policy, self.proxy
        )
        self.assertTrue(validate_alns_state(repaired_state, self.problem, self.policy))

    def test_infeasible_insertion_raises_controlled_exception(self):
        """Unassigned part with impossibly huge desi raises RepairInfeasibleError."""
        huge_part = DemandPart(
            part_id=f"{self.demand.demand_id}-huge",
            demand_id=self.demand.demand_id,
            desi=9999999.0, # Exceeds all vehicle capacities
        )
        huge_state = ALNSSolutionState(parts=(huge_part,), mandatory_trips=(), spot_routes=())
        proxy = ProxyEvaluator(self.problem, self.policy, huge_state)
        
        with self.assertRaises(RepairInfeasibleError):
            regret_2_insertion(huge_state, [(huge_part,)], self.problem, self.policy, proxy)

    def test_engine_catches_controlled_exception_without_crashing(self):
        """ALNSEngine catches RepairInfeasibleError gracefully and preserves incumbent."""
        from src.alns.engine import ALNSEngine
        
        base = GreedySameOriginTwoStopBuilder(self.problem, self.policy).build()
        result = GreedySameOriginMultiStopBuilder(self.problem, self.policy).build_from_two_stop(base)
        materializer = ALNSMaterializer(self.problem, self.policy)
        initial_state = materializer.extract_state(result.plan)
        
        engine = ALNSEngine(
            self.problem, self.policy, initial_state, result.plan, result.total_cost_tl, seed=42
        )
        
        # Mock repair_op to raise RepairInfeasibleError
        def failing_repair(*args, **kwargs):
            raise RepairInfeasibleError("Mock infeasible insertion")
            
        engine.repair_ops = [failing_repair]
        engine.repair_weights = [1.0]
        
        # Must run 5 iterations without crashing and return initial plan
        final_plan, final_cost = engine.run(max_iterations=5)
        self.assertEqual(final_cost, result.total_cost_tl)

    def test_repair_output_passes_state_invariants(self):
        """Greedy and Regret-2 repair outputs pass state invariant gate."""
        unassigned = [(self.part,)]
        greedy_state = greedy_insertion(self.base_state, unassigned, self.problem, self.policy, self.proxy)
        self.assertTrue(validate_alns_state(greedy_state, self.problem, self.policy))
        
        regret_state = regret_2_insertion(self.base_state, unassigned, self.problem, self.policy, self.proxy)
        self.assertTrue(validate_alns_state(regret_state, self.problem, self.policy))


if __name__ == "__main__":
    unittest.main()
