"""Reserve daily Tır visit quota while greedy strategies build spot routes.

The competition caps how many Tır visits each transfer centre can process per
calendar day, counting arrivals and departures in one shared pool.  Mandatory
rentals consume part of that pool on a fixed schedule, so a planner that wants
to use a spot Tır must know how much is left before it commits to one.

This ledger mirrors :class:`~src.planning.handling_ledger.HandlingCapacityLedger`
deliberately: same reserve / :meth:`fork` / :meth:`commit_from` shape, so the
two-stop and multi-stop builders can evaluate trial routes without leaking
phantom quota when a trial is rejected.

It models the planner's own consumption only.  ``TransportPlanValidator`` remains
the authority on the finished plan.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date

from src.domain import DailyTruckCapacity, MandatoryRentalRoute


class TruckQuotaError(ValueError):
    """Explain a Tır reservation that can never fit the static daily quota.

    Raised for unknown centres and for negative visit counts.  Ordinary "no
    room left today" answers are returned as ``False`` by :meth:`can_reserve`
    instead, because they are a normal planning outcome rather than a defect.
    """


class DailyTruckVisitLedger:
    """Track remaining daily Tır visits per transfer centre.

    Args:
        capacities: Static per-centre daily Tır visit quotas.
        mandatory_routes: Fixed daily rental routes; the Tır ones are charged
            against every service day before any spot reservation is offered.
        truck_vehicle_type: The vehicle type name the quota applies to.

    Connected modules and rules:
        ``GreedySameODConsolidatedBuilder`` and the two/multi-stop builders call
        :meth:`can_reserve` before packing a bin large enough to require a Tır,
        then :meth:`reserve` once the choice is committed.  One stop is one
        visit, matching the validator rule that a continuous unload-and-reload
        at a centre counts once while leaving and returning counts twice.

        Mandatory usage is identical on every service day because the rental
        fleet is fixed, so it is stored once as a per-centre baseline instead of
        being expanded per date.
    """

    def __init__(
        self,
        capacities: tuple[DailyTruckCapacity, ...],
        mandatory_routes: tuple[MandatoryRentalRoute, ...],
        truck_vehicle_type: str,
        extra_arrival_reserve: Mapping[str, int] | None = None,
    ) -> None:
        """Index static quotas and pre-charge the fixed mandatory Tır schedule.

        ``extra_arrival_reserve`` holds back additional visits at centres where
        a mandatory Tır may arrive the day after it departs.  Departure time
        depends on cargo readiness, so such a route can arrive on day N on one
        date and day N+1 on another, putting two of its arrivals on the same
        calendar day.  Reserving that worst case keeps spot Tır planning honest
        instead of discovering the clash in the validator.
        """

        self._capacity = {
            item.center: item.max_daily_truck_visits for item in capacities
        }
        self._mandatory_baseline: Counter[str] = Counter()
        for route in mandatory_routes:
            if route.vehicle_type != truck_vehicle_type:
                continue
            self._mandatory_baseline[route.origin] += route.vehicle_count
            self._mandatory_baseline[route.destination] += route.vehicle_count
        for center, count in (extra_arrival_reserve or {}).items():
            self._mandatory_baseline[center] += count
        self._spot_used: dict[tuple[str, date], int] = {}

    def remaining(self, center: str, service_date: date) -> int:
        """Return spot Tır visits still available at one centre on one day.

        Raises:
            TruckQuotaError: If the centre has no static quota record.
        """

        if center not in self._capacity:
            raise TruckQuotaError(f"Tır kotası bulunmayan merkez: {center!r}.")
        used = self._mandatory_baseline[center] + self._spot_used.get(
            (center, service_date), 0
        )
        return self._capacity[center] - used

    def can_reserve(
        self,
        centers: tuple[str, ...],
        service_date: date,
        visits_per_center: int = 1,
    ) -> bool:
        """Report whether every listed stop can take the requested visits.

        Args:
            centers: One entry per stop the Tır would make.  A centre appearing
                twice is charged twice, matching the leave-and-return rule.
            service_date: Calendar day the visits fall on.
            visits_per_center: How many separate Tır trips are being planned.

        Returns:
            ``True`` only when no centre would exceed its daily quota.

        Raises:
            TruckQuotaError: For a negative count or an unknown centre.
        """

        if visits_per_center < 0:
            raise TruckQuotaError("Tır ziyaret sayısı negatif olamaz.")
        if visits_per_center == 0 or not centers:
            return True
        needed: Counter[str] = Counter()
        for center in centers:
            needed[center] += visits_per_center
        return all(
            self.remaining(center, service_date) >= count
            for center, count in needed.items()
        )

    def reserve(
        self,
        centers: tuple[str, ...],
        service_date: date,
        visits_per_center: int = 1,
    ) -> None:
        """Consume quota for a committed choice, or explain why it cannot fit.

        Raises:
            TruckQuotaError: If the reservation does not fit.  Callers are
                expected to ask :meth:`can_reserve` first; this check exists so
                a missing guard fails loudly instead of silently overbooking.
        """

        if not self.can_reserve(centers, service_date, visits_per_center):
            raise TruckQuotaError(
                f"{service_date} tarihinde {centers!r} için "
                f"{visits_per_center} Tır ziyareti kotaya sığmıyor."
            )
        for center in centers:
            key = (center, service_date)
            self._spot_used[key] = self._spot_used.get(key, 0) + visits_per_center

    def fork(self) -> "DailyTruckVisitLedger":
        """Return an independent copy for trial route evaluation."""

        clone = DailyTruckVisitLedger((), (), "")
        clone._capacity = dict(self._capacity)
        clone._mandatory_baseline = Counter(self._mandatory_baseline)
        clone._spot_used = dict(self._spot_used)
        return clone

    def commit_from(self, trial: "DailyTruckVisitLedger") -> None:
        """Adopt a successful trial's reservations atomically.

        Raises:
            TruckQuotaError: If the trial was built from different static data,
                which would silently change what "remaining" means.
        """

        if (
            self._capacity != trial._capacity
            or self._mandatory_baseline != trial._mandatory_baseline
        ):
            raise TruckQuotaError(
                "Farklı kapasite veya zorunlu taban taşıyan ledger commit edilemez."
            )
        self._spot_used = dict(trial._spot_used)

    def spot_visits_by_center_day(self) -> dict[tuple[str, date], int]:
        """Return a deterministic snapshot of committed spot Tır visits."""

        return dict(sorted(self._spot_used.items()))


__all__ = ["DailyTruckVisitLedger", "TruckQuotaError"]
