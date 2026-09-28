"""Round-trip test for ALNS state materialization."""

from __future__ import annotations

import unittest
from pathlib import Path
import math

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy, CostCalculator, TransportPlanValidator
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import GreedySameOriginTwoStopBuilder, GreedySameOriginMultiStopBuilder
from src.alns.state import ALNSSolutionState
from src.alns.materializer import ALNSMaterializer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ALNSMaterializerTests(unittest.TestCase):
    """Test ALNS materialization accuracy."""

    def test_materializer_round_trip_retains_identical_cost_and_routing(self) -> None:
        """Extract state from 6E and materialize it back; cost and stops must exactly match."""

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

        # 2. Build 6E Plan (Incumbent)
        print("Building initial 6D plan...")
        base = GreedySameOriginTwoStopBuilder(problem, policy).build()
        print("Building initial 6E plan...")
        result = GreedySameOriginMultiStopBuilder(problem, policy).build_from_two_stop(base)
        original_plan = result.plan
        original_cost = result.total_cost_tl
        print(f"Original 6E Cost: {original_cost:,.4f} TL")

        # 3. Extract ALNS State
        materializer = ALNSMaterializer(problem, policy)
        state = materializer.extract_state(original_plan)
        print(f"Extracted ALNS State: {len(state.spot_routes)} spot routes")

        # Test Deepcopy speed/functionality
        cloned_state = state.clone()
        self.assertEqual(len(cloned_state.spot_routes), len(state.spot_routes))
        self.assertEqual(state.signature(), cloned_state.signature())

        # 4. Materialize State back to TransportPlan
        print("Materializing ALNS State back to TransportPlan...")
        rematerialized_plan = materializer.materialize(state)

        # 5. Evaluate the new plan
        calc = CostCalculator(problem, policy)
        new_cost_report = calc.calculate(rematerialized_plan)
        new_cost = new_cost_report.total_cost_tl
        print(f"Rematerialized Cost: {new_cost:,.4f} TL")

        # 6. Validate the new plan is legal
        validator = TransportPlanValidator(problem, policy)
        report = validator.validate(rematerialized_plan)
        self.assertTrue(report.is_valid, f"Rematerialized plan is invalid: {report.errors}")

        # 7. Assert Costs Match within tiny tolerance
        self.assertAlmostEqual(
            original_cost,
            new_cost,
            places=2,
            msg="Cost changed during round-trip materialization!",
        )

        # 8. Assert node visits order and assignments are structurally identical
        def extract_visits(plan):
            visits = []
            for trip in plan.trips:
                if trip.ownership_type == "spot":
                    # For a spot trip, extract ordered destinations and total desi dropped off
                    stops = []
                    for leg in trip.legs:
                        desi_dropped = math.fsum(a.desi for a in leg.assignments if a.is_final_delivery)
                        stops.append((leg.destination, round(desi_dropped, 2)))
                    # Sort trips by a tuple representation to ensure order independence in matching
                    visits.append((trip.vehicle_type, trip.legs[0].departure_at.date(), trip.legs[0].origin, tuple(stops)))
            return sorted(visits)

        original_visits = extract_visits(original_plan)
        new_visits = extract_visits(rematerialized_plan)

        self.assertEqual(len(original_visits), len(new_visits))
        for orig, new in zip(original_visits, new_visits):
            self.assertEqual(orig, new, "Node visits or vehicle types changed!")
