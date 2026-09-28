"""Immutable data models for the linehaul problem input.

The models in this module deliberately contain no routing, forecasting or
optimization behaviour.  They form the boundary between source-specific I/O
and later planning layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Center:
    """A transfer centre known by its canonical source-data name."""

    name: str


@dataclass(frozen=True, slots=True)
class TravelArc:
    """Directed distance, travel times and SLA target for an O-D pair."""

    origin: str
    destination: str
    distance_km: float
    travel_hours_by_vehicle: Mapping[str, float] = field(hash=False)
    target_delivery_days: int

    def __post_init__(self) -> None:
        """Protect the nested travel-time mapping from later mutation."""

        object.__setattr__(
            self,
            "travel_hours_by_vehicle",
            MappingProxyType(dict(self.travel_hours_by_vehicle)),
        )

    def travel_hours_for(self, vehicle_type: str) -> float:
        """Return the source travel time for ``vehicle_type``."""

        return self.travel_hours_by_vehicle[vehicle_type]


@dataclass(frozen=True, slots=True)
class VehicleType:
    """Capacity and cost parameters for one physical vehicle class."""

    name: str
    capacity_desi: float
    rented_hourly_cost_tl: float
    rented_cost_per_km_tl: float
    spot_hourly_cost_tl: float
    spot_cost_per_km_tl: float


@dataclass(frozen=True, slots=True)
class MandatoryRentalRoute:
    """Daily mandatory rented-vehicle count on one fixed directed route."""

    origin: str
    destination: str
    vehicle_type: str
    vehicle_count: int


@dataclass(frozen=True, slots=True)
class HandlingCapacity:
    """Calendar-day total handling capacity at one transfer centre."""

    center: str
    max_daily_desi: float


@dataclass(frozen=True, slots=True)
class DailyTruckCapacity:
    """Calendar-day total Tır visit/processing quota at one centre."""

    center: str
    max_daily_truck_visits: int


@dataclass(frozen=True, slots=True)
class StaticProblemData:
    """Problem inputs that stay unchanged in the surprise demand dataset."""

    centers: tuple[Center, ...]
    travel_arcs: tuple[TravelArc, ...]
    vehicle_types: tuple[VehicleType, ...]
    mandatory_rental_routes: tuple[MandatoryRentalRoute, ...]
    handling_capacities: tuple[HandlingCapacity, ...]
    daily_truck_capacities: tuple[DailyTruckCapacity, ...]

    def __post_init__(self) -> None:
        """Normalize collection inputs to tuples for deep immutability."""

        for field_name in (
            "centers",
            "travel_arcs",
            "vehicle_types",
            "mandatory_rental_routes",
            "handling_capacities",
            "daily_truck_capacities",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @property
    def center_names(self) -> frozenset[str]:
        """Return all canonical centre names as an immutable set."""

        return frozenset(center.name for center in self.centers)

    @property
    def maximum_vehicle_capacity_desi(self) -> float:
        """Return the largest vehicle capacity, or zero for an empty catalog."""

        return max(
            (vehicle.capacity_desi for vehicle in self.vehicle_types),
            default=0.0,
        )


@dataclass(frozen=True, slots=True)
class DemandInstance:
    """One dated demand record, independent from the static master data."""

    demand_id: str
    demand_date: date
    time_slot: time
    origin: str
    destination: str
    desi: float
    split_required: bool = False


@dataclass(frozen=True, slots=True)
class ProblemData:
    """Validated static data paired with one demand collection."""

    static: StaticProblemData
    demands: tuple[DemandInstance, ...]

    def __post_init__(self) -> None:
        """Normalize demand inputs to an immutable tuple."""

        object.__setattr__(self, "demands", tuple(self.demands))


__all__ = [
    "Center",
    "DailyTruckCapacity",
    "DemandInstance",
    "HandlingCapacity",
    "MandatoryRentalRoute",
    "ProblemData",
    "StaticProblemData",
    "TravelArc",
    "VehicleType",
]
