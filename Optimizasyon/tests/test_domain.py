"""Unit checks for immutable static, demand and combined domain models."""

from __future__ import annotations

from datetime import date, time
import unittest

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandInstance,
    HandlingCapacity,
    MandatoryRentalRoute,
    ProblemData,
    StaticProblemData,
    TravelArc,
    VehicleType,
)


def make_static_data() -> StaticProblemData:
    """Build minimal immutable static data for domain-model tests."""

    return StaticProblemData(
        centers=[Center("A"), Center("B")],
        travel_arcs=[
            TravelArc("A", "B", 10.0, {"Tır": 1.0}, 1),
            TravelArc("B", "A", 10.0, {"Tır": 1.0}, 1),
        ],
        vehicle_types=[VehicleType("Tır", 100.0, 10.0, 2.0, 20.0, 3.0)],
        mandatory_rental_routes=[MandatoryRentalRoute("A", "B", "Tır", 1)],
        handling_capacities=[HandlingCapacity("A", 1_000.0), HandlingCapacity("B", 1_000.0)],
        daily_truck_capacities=[DailyTruckCapacity("A", 2), DailyTruckCapacity("B", 2)],
    )


class DomainModelTests(unittest.TestCase):
    """Verify source problem dataclasses preserve their invariants."""

    def test_static_collections_and_nested_travel_times_are_immutable(self) -> None:
        """Collection inputs and nested travel mappings must be read-only."""

        static = make_static_data()

        self.assertIsInstance(static.centers, tuple)
        self.assertEqual(static.center_names, frozenset({"A", "B"}))
        with self.assertRaises(TypeError):
            static.travel_arcs[0].travel_hours_by_vehicle["Tır"] = 2.0  # type: ignore[index]

    def test_problem_data_keeps_demand_separate_from_static_data(self) -> None:
        """ProblemData must combine rather than merge static and demand data."""

        demand = DemandInstance(
            demand_id="D1",
            demand_date=date(2026, 6, 29),
            time_slot=time(9),
            origin="A",
            destination="B",
            desi=150.0,
            split_required=True,
        )

        problem = ProblemData(static=make_static_data(), demands=[demand])

        self.assertIsInstance(problem.demands, tuple)
        self.assertTrue(problem.demands[0].split_required)
        self.assertEqual(problem.static.maximum_vehicle_capacity_desi, 100.0)


if __name__ == "__main__":
    unittest.main()
