"""Build a strategy-local same-day/same-O-D consolidated spot baseline.

``GreedySameODConsolidatedBuilder`` reuses the conservative builder's
partitioning, handling ledger, mandatory-rental schedule, cost calculation and
independent validation gate.  It changes only spot-trip construction by using
deterministic First-Fit Decreasing bins.  The choices in this module are not
global competition rules and are not optimizer, route-pool or exporter logic.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import math

from src.domain import (
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    TravelArc,
    VehicleLeg,
    VehicleTrip,
    VehicleType,
)

from .baseline import (
    MINUTES_PER_HOUR,
    BaselineBuildError,
    ConservativeDirectBaselineBuilder,
)
from .handling_ledger import HandlingCapacityLedger


GroupKey = tuple[date, str, str]


class GreedySameODConsolidatedBuilder(ConservativeDirectBaselineBuilder):
    """Consolidate same-date/same-O-D parts with deterministic FFD packing.

    Args:
        problem: Standard static data and immutable demand collection.
        policy: Evaluation policy supplying desi tolerance, handling duration,
            cost rules and validator settings.

    Returns and connected layers:
        Inherited :meth:`build` returns ``BaselineBuildResult``.  Partitioning,
        the shared ``HandlingCapacityLedger``, empty mandatory rentals,
        ``CostCalculator`` and ``TransportPlanValidator`` remain owned by the
        conservative planning/evaluation layers.  This subclass overrides only
        creation of direct spot trips.

    Strategy rules and failures:
        Existing atomic ``DemandPart`` records are grouped by demand date,
        origin and destination.  FFD may place 09:00 and 17:00 parts together;
        every bin waits for the whole group's latest ready time.  Each bin uses
        one new non-Tır vehicle, one leg and grouped load/unload operations.
        Unknown data, unavailable vehicles, handling infeasibility, costing or
        validator defects surface as inherited ``BaselineBuildError`` records.
        These choices are strategy-local, not global business rules.
    """

    def _build_spot_trips(
        self,
        parts: tuple[DemandPart, ...],
        ledger: HandlingCapacityLedger,
    ) -> tuple[VehicleTrip, ...]:
        """Create direct consolidated trips in deterministic group/bin order."""

        groups = self._group_parts(parts)
        truck_allowance = self._reserve_truck_allowance(groups)
        trips: list[VehicleTrip] = []
        next_index = 1
        for group_key in sorted(groups):
            group_parts = tuple(groups[group_key])
            bins = self._pack_group(
                group_parts,
                truck_allowance.get(group_key, 0),
            )
            for bin_parts in bins:
                trips.append(
                    self._build_bin_trip(
                        next_index,
                        group_key,
                        self._parts_ready_at(bin_parts),
                        bin_parts,
                        ledger,
                    )
                )
                next_index += 1
        return tuple(trips)

    def _reserve_truck_allowance(
        self,
        groups: dict[GroupKey, list[DemandPart]],
    ) -> dict[GroupKey, int]:
        """Book daily Tır quota and return how many Tır bins each group may use.

        Args:
            groups: Atoms grouped by demand date, origin and destination.

        Returns:
            Group key to the number of Tır-sized bins reserved for it.  Groups
            absent from the mapping keep the largest non-Tır capacity, so every
            one of their bins can be served by a Kamyon.

        Why the order matters:
            Daily Tır quota is scarce and shared.  After the fixed mandatory
            fleet, İstanbul has two spot visits left, Yalova one, and Balıkesir
            and Tekirdağ none.  Iterating alphabetically would hand that
            allowance to whichever O-D pair sorts first rather than to the
            traffic that can fill a Tır, so groups are ranked by descending
            total desi with the group key breaking ties.  Trips are still built
            in plain sorted order afterwards, so vehicle identifiers do not
            depend on this ranking.

        Partial allowance:
            A group needing three Tır bins on a day with room for one still gets
            that one.  Falling back to non-Tır bins for the whole group would
            waste quota that is already scarce.
        """

        non_truck_capacity = self._maximum_non_truck_capacity()
        truck_capacity = self._truck_capacity_desi()
        allowance: dict[GroupKey, int] = {}
        if truck_capacity <= non_truck_capacity:
            return allowance

        ranked = sorted(
            groups,
            key=lambda key: (
                -math.fsum(part.desi for part in groups[key]),
                key,
            ),
        )
        for group_key in ranked:
            service_date, origin, destination = group_key
            needed = sum(
                1
                for bin_parts in self._first_fit_decreasing(
                    tuple(groups[group_key]),
                    truck_capacity,
                )
                if math.fsum(part.desi for part in bin_parts)
                > non_truck_capacity + self.policy.desi_tolerance
            )
            if needed == 0:
                continue
            stops = (origin, destination)
            affordable = self._affordable_truck_bins(stops, service_date, needed)
            if affordable == 0:
                continue
            self._truck_ledger.reserve(stops, service_date, affordable)
            allowance[group_key] = affordable
        return allowance

    def _affordable_truck_bins(
        self,
        stops: tuple[str, ...],
        service_date: date,
        needed: int,
    ) -> int:
        """Return the largest reservable Tır bin count, possibly zero."""

        for count in range(needed, 0, -1):
            if self._truck_ledger.can_reserve(stops, service_date, count):
                return count
        return 0

    def _pack_group(
        self,
        parts: tuple[DemandPart, ...],
        truck_allowance: int,
    ) -> tuple[tuple[DemandPart, ...], ...]:
        """Pack one group, spending at most ``truck_allowance`` Tır bins.

        Packing runs once at Tır capacity.  Oversized bins beyond the reserved
        allowance are re-packed at non-Tır capacity, so a bin never demands a
        vehicle whose daily quota was not booked.  Because FFD fills earlier
        bins first, the retained Tır bins are the fullest ones.
        """

        non_truck_capacity = self._maximum_non_truck_capacity()
        if truck_allowance <= 0:
            return self._first_fit_decreasing(parts, non_truck_capacity)

        packed: list[tuple[DemandPart, ...]] = []
        spent = 0
        for bin_parts in self._first_fit_decreasing(parts, self._truck_capacity_desi()):
            needs_truck = (
                math.fsum(part.desi for part in bin_parts)
                > non_truck_capacity + self.policy.desi_tolerance
            )
            if not needs_truck:
                packed.append(bin_parts)
            elif spent < truck_allowance:
                packed.append(bin_parts)
                spent += 1
            else:
                packed.extend(
                    self._first_fit_decreasing(bin_parts, non_truck_capacity)
                )
        return tuple(packed)

    def _group_parts(
        self,
        parts: tuple[DemandPart, ...],
    ) -> dict[GroupKey, list[DemandPart]]:
        """Group unchanged atoms by demand date, origin and destination."""

        grouped: dict[GroupKey, list[DemandPart]] = defaultdict(list)
        for part in parts:
            demand = self._demands[part.demand_id]
            key = (demand.demand_date, demand.origin, demand.destination)
            grouped[key].append(part)
        return dict(grouped)

    def _parts_ready_at(self, parts: tuple[DemandPart, ...]) -> datetime:
        """Return the latest ready timestamp across the given atoms.

        Callers pass one **bin**, not the whole O-D/day group.  Grouping is by
        date and O-D only, so a group normally mixes the 09:00 and 17:00 demand
        slots; taking the group maximum held every bin back to 17:00, including
        bins made purely of 09:00 cargo.  Measured on the development forecast
        that delayed 649.096 desi — 9,9% of all volume — for no gain.  Bins that
        really do contain 17:00 cargo still wait, because the maximum inside
        such a bin is 17:00 anyway.
        """

        return max(
            datetime.combine(
                self._demands[part.demand_id].demand_date,
                self._demands[part.demand_id].time_slot,
            )
            for part in parts
        )

    def _first_fit_decreasing(
        self,
        parts: tuple[DemandPart, ...],
        maximum_capacity_desi: float,
    ) -> tuple[tuple[DemandPart, ...], ...]:
        """Pack atoms by descending desi into the first fitting vehicle bin.

        Sorting ties by ``part_id`` and retaining bin creation order makes FFD
        repeatable.  Capacity comparisons use ``policy.desi_tolerance`` desi;
        atom desi values and identities are never changed or rounded.
        """

        ordered = sorted(parts, key=lambda part: (-part.desi, part.part_id))
        bins: list[list[DemandPart]] = []
        for part in ordered:
            target = next(
                (
                    vehicle_bin
                    for vehicle_bin in bins
                    if math.fsum(item.desi for item in vehicle_bin) + part.desi
                    <= maximum_capacity_desi + self.policy.desi_tolerance
                ),
                None,
            )
            if target is None:
                bins.append([part])
            else:
                target.append(part)
        return tuple(tuple(vehicle_bin) for vehicle_bin in bins)

    def _build_bin_trip(
        self,
        index: int,
        group_key: GroupKey,
        ready_at: datetime,
        parts: tuple[DemandPart, ...],
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip:
        """Schedule one FFD bin as one direct vehicle with grouped handling."""

        _, origin, destination = group_key
        total_desi = math.fsum(part.desi for part in parts)
        arc = self._arc_for(origin, destination)
        vehicle_type = self._select_bin_vehicle(total_desi, arc)
        load_start, load_end = ledger.schedule(origin, ready_at, total_desi)
        arrival_at = load_end + timedelta(
            hours=arc.travel_hours_for(vehicle_type.name)
        )
        unload_start, unload_end = ledger.schedule(
            destination,
            arrival_at,
            total_desi,
        )
        return self._grouped_trip_models(
            index,
            origin,
            destination,
            parts,
            total_desi,
            vehicle_type,
            load_start,
            load_end,
            arrival_at,
            unload_start,
            unload_end,
        )

    def _select_bin_vehicle(
        self,
        total_desi: float,
        arc: TravelArc,
    ) -> VehicleType:
        """Choose the cheapest feasible type for one bin's total desi.

        Tır enters the candidate set only when the bin exceeds every non-Tır
        capacity, which is exactly when ``_reserve_truck_groups`` has already
        booked a daily visit for it.  For any smaller bin a Kamyon, Hafif Kamyon
        or Kamyonet is always cheaper than a Tır, so excluding it there costs
        nothing and keeps the ledger and the vehicle choice in step.
        """

        non_truck_capacity = self._maximum_non_truck_capacity()
        candidates = [
            vehicle
            for vehicle in self._vehicle_types.values()
            if vehicle.capacity_desi + self.policy.desi_tolerance >= total_desi
            and (
                vehicle.name != self.policy.truck_vehicle_type
                or total_desi > non_truck_capacity + self.policy.desi_tolerance
            )
        ]
        if not candidates:
            raise BaselineBuildError(
                f"{total_desi} desilik konsolide kutu için uygun araç yok.",
                error_codes=("non_truck_vehicle_unavailable",),
            )
        return min(
            candidates,
            key=lambda vehicle: (
                self._approximate_bin_cost(total_desi, vehicle, arc),
                vehicle.capacity_desi,
                vehicle.name,
            ),
        )

    def _approximate_bin_cost(
        self,
        total_desi: float,
        vehicle: VehicleType,
        arc: TravelArc,
    ) -> float:
        """Estimate direct bin cost in TL with minute-based hourly ceiling."""

        handling_minutes = (
            2.0 * total_desi * self.policy.handling_minutes_per_desi
        )
        usage_minutes = (
            handling_minutes
            + arc.travel_hours_for(vehicle.name) * MINUTES_PER_HOUR
        )
        return (
            math.ceil(usage_minutes / MINUTES_PER_HOUR)
            * vehicle.spot_hourly_cost_tl
            + arc.distance_km * vehicle.spot_cost_per_km_tl
        )

    @staticmethod
    def _grouped_trip_models(
        index: int,
        origin: str,
        destination: str,
        parts: tuple[DemandPart, ...],
        total_desi: float,
        vehicle_type: VehicleType,
        load_start: datetime,
        load_end: datetime,
        arrival_at: datetime,
        unload_start: datetime,
        unload_end: datetime,
    ) -> VehicleTrip:
        """Create one leg and two grouped operations without altering atoms."""

        stem = f"GSOD-{index:05d}"
        part_ids = tuple(part.part_id for part in parts)
        assignments = tuple(
            DemandAssignment(part.part_id, part.desi, is_final_delivery=True)
            for part in parts
        )
        load = HandlingOperation(
            f"{stem}-LOAD", stem, origin, "load", load_start, load_end,
            total_desi, part_ids,
        )
        unload = HandlingOperation(
            f"{stem}-UNLOAD", stem, destination, "unload", unload_start,
            unload_end, total_desi, part_ids,
        )
        leg = VehicleLeg(
            f"{stem}-LEG", origin, destination, load_end, arrival_at,
            assignments, load.operation_id, unload.operation_id,
        )
        return VehicleTrip(
            f"{stem}-TRIP", stem, vehicle_type.name, "spot", (leg,),
            (load, unload),
        )


__all__ = ["GreedySameODConsolidatedBuilder"]
