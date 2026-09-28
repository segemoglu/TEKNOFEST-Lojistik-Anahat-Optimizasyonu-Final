"""Lightweight state representation for ALNS.

This module provides the ALNSSolutionState class and related structures, which
allow the ALNS algorithm to manipulate routes efficiently without the overhead
of full TransportPlan evaluation and object tree recreation during every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Tuple

from src.domain import DemandPart, VehicleTrip


@dataclass(frozen=True, slots=True)
class ALNSRouteStop:
    """One stop in a spot route, holding its dropoffs and pickups."""

    center: str
    dropoffs: tuple[DemandPart, ...]
    pickups: tuple[DemandPart, ...]
    dropoff_desi: float = -1.0
    pickup_desi: float = -1.0
    stop_key: tuple = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.dropoff_desi < 0:
            object.__setattr__(self, 'dropoff_desi', sum(part.desi for part in self.dropoffs))
        if self.pickup_desi < 0:
            object.__setattr__(self, 'pickup_desi', sum(part.desi for part in self.pickups))
        dropoff_ids = tuple(p.part_id for p in self.dropoffs)
        pickup_ids = tuple(p.part_id for p in self.pickups)
        object.__setattr__(
            self,
            'stop_key',
            (self.center, dropoff_ids, pickup_ids, self.dropoff_desi, self.pickup_desi)
        )


@dataclass(frozen=True, slots=True)
class ALNSSpotRoute:
    """A sequence of stops performed by one spot vehicle."""

    service_date: date
    stops: tuple[ALNSRouteStop, ...]
    vehicle_type: str

    @property
    def origin(self) -> str:
        return self.stops[0].center


@dataclass(frozen=True, slots=True)
class ALNSSolutionState:
    """A lightweight, ALNS-mutable representation of a complete solution.

    It holds references to immutable DemandParts, unchanged mandatory trips,
    and a mutable list of spot routes.
    """

    parts: tuple[DemandPart, ...]
    mandatory_trips: tuple[VehicleTrip, ...]
    spot_routes: tuple[ALNSSpotRoute, ...]

    def clone(self) -> ALNSSolutionState:
        """Return a shallow copy of the state.

        Since all nested collections and objects are immutable, a shallow copy
        is effectively a fast deep copy.
        """
        return ALNSSolutionState(
            self.parts,
            self.mandatory_trips,
            self.spot_routes,
        )

    def canonical_tuple(self) -> tuple:
        """Return a hash-independent, cross-process deterministic representation of the solution state.
        
        Includes route order, service date, vehicle type, stop order, center,
        and sorted pickup/dropoff part IDs.
        """
        route_entries = []
        for route in self.spot_routes:
            stop_entries = []
            for stop in route.stops:
                pickups = tuple(sorted(p.part_id for p in stop.pickups))
                dropoffs = tuple(sorted(p.part_id for p in stop.dropoffs))
                stop_entries.append((stop.center, pickups, dropoffs))
            route_entries.append((
                route.service_date.isoformat(),
                route.vehicle_type,
                tuple(stop_entries),
            ))
        return tuple(route_entries)

    def signature(self) -> str:
        """Return a deterministic SHA-256 hex digest signature for state comparison across processes."""
        import hashlib
        canonical_bytes = str(self.canonical_tuple()).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

__all__ = [
    "ALNSRouteStop",
    "ALNSSpotRoute",
    "ALNSSolutionState",
]
