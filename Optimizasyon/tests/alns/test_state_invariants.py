"""Unit tests for ALNS state invariant gate."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from src.domain import ProblemData, DemandPart
from src.domain.models import DemandInstance
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.planning import GreedySameOriginTwoStopBuilder, GreedySameOriginMultiStopBuilder
from src.alns.state import ALNSSolutionState, ALNSSpotRoute, ALNSRouteStop
from src.alns.materializer import ALNSMaterializer
from src.alns.invariants import validate_alns_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class StateInvariantsTests(unittest.TestCase):
    """Test suite for ALNS State Invariant Gate validation rules."""

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
        cls.demand_map = {d.demand_id: d for d in cls.demands}

    def _create_mock_part(self, demand_id: str, part_id: int, desi: float = 100.0) -> DemandPart:
        return DemandPart(
            part_id=f"{demand_id}-{part_id}",
            demand_id=demand_id,
            desi=desi,
        )

    def test_valid_small_state_accepted(self):
        """A valid small state with complete pickup and dropoff is accepted."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertTrue(validate_alns_state(state, self.problem, self.policy))

    def test_missing_part_rejected(self):
        """A state where a required spot part is missing from spot routes is rejected."""
        demand = self.demands[0]
        part1 = self._create_mock_part(demand.demand_id, 1, 50.0)
        part2 = self._create_mock_part(demand.demand_id, 2, 50.0)
        
        # Only part1 is routed, part2 is missing
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part1,))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part1,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part1, part2), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_duplicate_pickup_rejected(self):
        """A part picked up twice across routes or stops is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part, part))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_duplicate_dropoff_rejected(self):
        """A part dropped off twice across routes or stops is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part, part), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_wrong_origin_pickup_rejected(self):
        """A part picked up at a center that is not its demand origin is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        wrong_center = "İzmir" if demand.origin != "İzmir" else "Ankara"
        
        stop1 = ALNSRouteStop(center=wrong_center, dropoffs=(), pickups=(part,))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_wrong_destination_dropoff_rejected(self):
        """A part dropped off at a center that is not its demand destination is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        wrong_center = "İzmir" if demand.destination != "İzmir" else "Ankara"
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        stop2 = ALNSRouteStop(center=wrong_center, dropoffs=(part,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_dropoff_before_pickup_rejected(self):
        """A route where dropoff occurs before pickup (onboard load goes negative) is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        
        # Dropoff at stop1, pickup at stop2 -> Negative onboard load at stop1
        stop1 = ALNSRouteStop(center=demand.destination, dropoffs=(part,), pickups=())
        stop2 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_remaining_load_at_route_end_rejected(self):
        """A route ending with onboard load remaining (missing dropoff) is rejected."""
        demand = self.demands[0]
        part = self._create_mock_part(demand.demand_id, 1, 100.0)
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        # Route ends at stop1 without dropoff
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1,), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_capacity_overload_rejected(self):
        """A route exceeding the vehicle desi capacity beyond tolerance is rejected."""
        demand = self.demands[0]
        # Kamyon capacity is 12000 desi. Creating a part with 50000 desi.
        part = self._create_mock_part(demand.demand_id, 1, 50000.0)
        
        stop1 = ALNSRouteStop(center=demand.origin, dropoffs=(), pickups=(part,))
        stop2 = ALNSRouteStop(center=demand.destination, dropoffs=(part,), pickups=())
        route = ALNSSpotRoute(service_date=demand.demand_date, stops=(stop1, stop2), vehicle_type="Kamyon")
        
        state = ALNSSolutionState(parts=(part,), mandatory_trips=(), spot_routes=(route,))
        self.assertFalse(validate_alns_state(state, self.problem, self.policy))

    def test_valid_initial_6e_state_accepted(self):
        """The initial 6E plan state extracted via materializer is valid and accepted."""
        base = GreedySameOriginTwoStopBuilder(self.problem, self.policy).build()
        result = GreedySameOriginMultiStopBuilder(self.problem, self.policy).build_from_two_stop(base)
        
        materializer = ALNSMaterializer(self.problem, self.policy)
        initial_state = materializer.extract_state(result.plan)
        
        self.assertTrue(validate_alns_state(initial_state, self.problem, self.policy))


if __name__ == "__main__":
    unittest.main()
