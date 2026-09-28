"""Spatial-temporal index for fast ALNS spot route candidate lookup.

Maps (service_date, center) to spot route indices to eliminate full scans
over unrelated routes during repair operations.
"""

from __future__ import annotations

from collections import defaultdict
import datetime

from src.alns.state import ALNSSpotRoute


class RouteIndex:
    """Deterministic spatial-temporal index for ALNS spot routes."""

    def __init__(self, spot_routes: list[ALNSSpotRoute] | tuple[ALNSSpotRoute, ...]):
        self.index: dict[tuple[datetime.date, str], set[int]] = defaultdict(set)
        self.route_centers: list[set[str]] = []
        self.route_dates: list[datetime.date] = []

        for i, route in enumerate(spot_routes):
            self._add_route(i, route)

    def _add_route(self, route_idx: int, route: ALNSSpotRoute) -> None:
        centers = {stop.center for stop in route.stops}
        self.route_centers.append(centers)
        self.route_dates.append(route.service_date)
        for center in centers:
            self.index[(route.service_date, center)].add(route_idx)

    def update_route(self, route_idx: int, new_route: ALNSSpotRoute) -> None:
        """Update an existing route in the index (purge old center keys, add new ones)."""
        old_date = self.route_dates[route_idx]
        old_centers = self.route_centers[route_idx]

        for center in old_centers:
            key = (old_date, center)
            if key in self.index:
                self.index[key].discard(route_idx)
                if not self.index[key]:
                    del self.index[key]

        new_centers = {stop.center for stop in new_route.stops}
        self.route_centers[route_idx] = new_centers
        self.route_dates[route_idx] = new_route.service_date
        for center in new_centers:
            self.index[(new_route.service_date, center)].add(route_idx)

    def append_route(self, new_route: ALNSSpotRoute) -> int:
        """Append a new route to the index and return its assigned route index."""
        route_idx = len(self.route_centers)
        self._add_route(route_idx, new_route)
        return route_idx

    def get_candidate_indices(
        self,
        service_date: datetime.date,
        origin: str,
        destination: str,
        extra_days: int = 0,
    ) -> list[int]:
        """Return sorted route indices touching the origin or the destination.

        Args:
            service_date: The day the cargo becomes ready.
            origin: Centre the cargo must be picked up at.
            destination: Centre it must be dropped at.
            extra_days: How many later service days may also be offered.  Cargo
                can always wait at its origin, so a route running one or two days
                later is a legal home for it; the proxy prices both the waiting
                and any SLA penalty, and the exact audit prices the extra
                mandatory rental day.  Zero keeps the original same-day-only
                behaviour.

        Returns:
            Deterministically ordered route indices.
        """

        combined: set[int] = set()
        for offset in range(max(extra_days, 0) + 1):
            day = service_date + datetime.timedelta(days=offset)
            combined |= self.index.get((day, origin), set())
            combined |= self.index.get((day, destination), set())
        return sorted(combined)


__all__ = ["RouteIndex"]
