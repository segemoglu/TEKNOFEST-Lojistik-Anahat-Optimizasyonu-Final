"""Unit tests for ALNS repair vehicle selection logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.domain import ProblemData, DemandPart
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.evaluator import ProxyEvaluator
from src.alns.repair import (
    greedy_insertion,
    regret_2_insertion,
    select_best_vehicle_for_route,
)
from src.alns.exceptions import RepairInfeasibleError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class VehicleSelectionTests(unittest.TestCase):
    """Test suite for vehicle selection helper and cost-based vehicle choice."""

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

    def test_cheaper_larger_vehicle_selected_over_expensive_small_vehicle(self):
        """Vehicle selection picks the lowest cost vehicle, even if a smaller capacity vehicle is available."""
        stop1 = ALNSRouteStop(self.demand.origin, (), (self.part,))
        stop2 = ALNSRouteStop(self.demand.destination, (self.part,), ())
        stops = (stop1, stop2)

        # Mock vehicle costs such that a larger vehicle is artificially cheaper
        original_vehicles = self.proxy.vehicles
        try:
            # Temporarily modify spot_hourly_cost_tl and spot_cost_per_km_tl
            kamyonet = original_vehicles["Kamyonet"]
            kamyon = original_vehicles["Kamyon"]
            
            # Make Kamyon cheaper per km than Kamyonet in proxy
            object.__setattr__(kamyonet, "spot_cost_per_km_tl", 1000.0)
            object.__setattr__(kamyon, "spot_cost_per_km_tl", 1.0)
            
            cost, best_v_name = select_best_vehicle_for_route(
                stops, self.demand.demand_date, self.proxy, self.policy
            )
            self.assertEqual(best_v_name, "Kamyon")
        finally:
            # Restore original costs
            object.__setattr__(kamyonet, "spot_cost_per_km_tl", 22.0)
            object.__setattr__(kamyon, "spot_cost_per_km_tl", 28.0)

    def _huge_part(self) -> DemandPart:
        """Return a load only a Tır can carry (over the 12.000 desi Kamyon)."""

        return DemandPart(
            part_id=f"{self.demand.demand_id}-huge",
            demand_id=self.demand.demand_id,
            desi=15000.0,
        )

    def test_cheapest_vehicle_rejected_if_capacity_insufficient(self):
        """A cheap vehicle is rejected if its capacity is smaller than the route max load."""
        # Both centres have spare daily Tır quota, so capacity alone decides.
        stop1 = ALNSRouteStop("Kocaeli", (), (self._huge_part(),))
        stop2 = ALNSRouteStop("Mersin", (self._huge_part(),), ())
        stops = (stop1, stop2)

        cost, best_v_name = select_best_vehicle_for_route(
            stops, self.demand.demand_date, self.proxy, self.policy
        )
        self.assertEqual(best_v_name, "Tır")

    def test_truck_refused_where_daily_quota_is_zero(self):
        """A Tır-only load is infeasible into a centre with no Tır quota.

        Bilecik has a daily quota of zero, so no vehicle can serve this route
        and the helper must say so instead of proposing a Tır the exact audit
        would later reject.
        """

        stop1 = ALNSRouteStop("Kocaeli", (), (self._huge_part(),))
        stop2 = ALNSRouteStop("Bilecik", (self._huge_part(),), ())

        cost, best_v_name = select_best_vehicle_for_route(
            (stop1, stop2), self.demand.demand_date, self.proxy, self.policy
        )

        self.assertEqual(cost, float("inf"))
        self.assertIsNone(best_v_name)

    def test_truck_allowed_only_where_quota_permits(self):
        """The quota screen accepts centres with room and refuses those without."""

        self.assertTrue(
            self.proxy.can_use_truck(self.demand.demand_date, {"Kocaeli", "Mersin"})
        )
        self.assertFalse(
            self.proxy.can_use_truck(self.demand.demand_date, {"Kocaeli", "Bilecik"})
        )

    def test_deterministic_tie_breaking_by_vehicle_name(self):
        """When two vehicles yield identical route cost, alphabetical vehicle name breaks ties."""
        stop1 = ALNSRouteStop(self.demand.origin, (), (self.part,))
        stop2 = ALNSRouteStop(self.demand.destination, (self.part,), ())
        stops = (stop1, stop2)

        kamyonet = self.proxy.vehicles["Kamyonet"]
        kamyon = self.proxy.vehicles["Kamyon"]
        try:
            # Make both vehicles have identical cost
            object.__setattr__(kamyonet, "spot_hourly_cost_tl", 100.0)
            object.__setattr__(kamyonet, "spot_cost_per_km_tl", 10.0)
            object.__setattr__(kamyon, "spot_hourly_cost_tl", 100.0)
            object.__setattr__(kamyon, "spot_cost_per_km_tl", 10.0)

            cost, best_v_name = select_best_vehicle_for_route(
                stops, self.demand.demand_date, self.proxy, self.policy
            )
            # Alphabetically 'Kamyon' < 'Kamyonet'
            self.assertEqual(best_v_name, "Kamyon")
        finally:
            object.__setattr__(kamyonet, "spot_hourly_cost_tl", 220.0)
            object.__setattr__(kamyonet, "spot_cost_per_km_tl", 22.0)
            object.__setattr__(kamyon, "spot_hourly_cost_tl", 280.0)
            object.__setattr__(kamyon, "spot_cost_per_km_tl", 28.0)

    def test_direct_route_uses_vehicle_selection_helper(self):
        """Direct route insertion uses select_best_vehicle_for_route."""
        repaired_state = greedy_insertion(
            self.base_state, [(self.part,)], self.problem, self.policy, self.proxy
        )
        self.assertEqual(len(repaired_state.spot_routes), 1)
        # Kamyonet is cheapest for 100 desi direct route under standard rates
        self.assertIn(repaired_state.spot_routes[0].vehicle_type, self.proxy.vehicles)

    def test_final_vehicle_sweep_uses_vehicle_selection_helper(self):
        """Vehicle sweep at end of repair updates vehicle type using select_best_vehicle_for_route."""
        stop1 = ALNSRouteStop(self.demand.origin, (), (self.part,))
        stop2 = ALNSRouteStop(self.demand.destination, (self.part,), ())
        # Start route with Tır (oversized/expensive for 100 desi)
        oversized_route = ALNSSpotRoute(self.demand.demand_date, (stop1, stop2), "Tır")
        state_with_oversized = ALNSSolutionState(
            parts=(self.part,),
            mandatory_trips=(),
            spot_routes=(oversized_route,),
        )
        
        # Calling greedy_insertion with empty unassigned triggers vehicle sweep
        repaired_state = greedy_insertion(
            state_with_oversized, [], self.problem, self.policy, self.proxy
        )
        # Swept to cheapest suitable vehicle (Kamyonet)
        self.assertEqual(repaired_state.spot_routes[0].vehicle_type, "Kamyonet")

    def test_no_suitable_vehicle_returns_none_without_silent_loss(self):
        """When no vehicle capacity fits the route load, helper returns (inf, None)."""
        impossible_part = DemandPart(
            part_id=f"{self.demand.demand_id}-impossible",
            demand_id=self.demand.demand_id,
            desi=9999999.0,
        )
        stop1 = ALNSRouteStop(self.demand.origin, (), (impossible_part,))
        stop2 = ALNSRouteStop(self.demand.destination, (impossible_part,), ())
        
        cost, best_v = select_best_vehicle_for_route(
            (stop1, stop2), self.demand.demand_date, self.proxy, self.policy
        )
        self.assertTrue(cost == float('inf'))
        self.assertIsNone(best_v)


if __name__ == "__main__":
    unittest.main()
