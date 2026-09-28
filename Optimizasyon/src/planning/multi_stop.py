"""Extend economical 6D pairs into long same-origin spot milk-run routes.

``GreedySameOriginMultiStopBuilder`` keeps the 6B logical spot bins and uses
6D's positive two-stop candidates as route seeds.  A deterministic
cheapest-insertion pass tries one unused destination bin at every route
position, up to ``MAX_DELIVERY_STOPS`` stops.  Every insertion is scheduled on a
forked handling ledger and accepted only when real vehicle, SLA and
mandatory-horizon TL improves.  This is a bounded greedy route heuristic, not
ALNS or global set-partitioning optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from time import perf_counter

from src.domain import (
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    TransportPlan,
    VehicleLeg,
    VehicleTrip,
    VehicleType,
)
from src.evaluation import CostCalculator

from .baseline import BaselineBuildError, BaselineBuildResult
from .handling_ledger import HandlingCapacityError, HandlingCapacityLedger
from .two_stop import (
    GreedySameOriginTwoStopBuilder,
    _PairEvaluation,
    _SpotBin,
)


# A route grows only while each added stop strictly lowers the real
# CostCalculator total, so this is a runtime guard rather than a quality choice.
# It was 4, which measurably capped the plan: 155 spot vehicles ended on exactly
# four legs costing 1,64 TL/desi, worse than both the three-leg (1,13) and
# six-leg (0,98) buckets, because growth stopped mid-way.  The average daily
# origin-destination flow is 1.305 desi against a 12.000 desi Kamyon, and dock
# quota applies to Tır only (Q&A 10), so long non-Tır milk-runs break no rule.
#
# Raising this from 4 needed S-28 and S-29 first.  Long routes span several
# calendar days, and two layers had silently assumed they never did: the
# materializer left the mandatory schedule at the horizon 6E fixed, and the proxy
# charged every stop to `service_date`.  With the cap alone, 712 of 747 audited
# candidates were rejected as `mandatory_vehicle_missing` and 698 as
# `daily_truck_quota_exceeded`, and the finished plan got 1.080.154 TL dearer even
# though the seed was 951.303 TL cheaper.  With both fixed, all four seeds
# improved by 628.505-1.221.003 TL.
MAX_DELIVERY_STOPS = 20
MIN_MULTI_STOP_DELIVERIES = 3


@dataclass(frozen=True, slots=True)
class _RouteEvaluation:
    """Hold one fully scheduled/costed route and its isolated ledger state."""

    ordered_bins: tuple[_SpotBin, ...]
    trip: VehicleTrip
    trial_ledger: HandlingCapacityLedger
    saving_tl: float
    combined_total_tl: float
    distance_km: float
    finish_at: datetime

    @property
    def key(self) -> tuple[int, ...]:
        """Return stable source-bin IDs independently of delivery order."""

        return tuple(sorted(item.index for item in self.ordered_bins))


@dataclass(frozen=True, slots=True)
class _MultiStopAttempt:
    """Bundle one complete plan and selected bounded multi-stop routes."""

    result: BaselineBuildResult
    routes: tuple[_RouteEvaluation, ...]


class GreedySameOriginMultiStopBuilder(GreedySameOriginTwoStopBuilder):
    """Grow 6D route seeds by real-cost cheapest insertion into milk-runs.

    Args:
        problem: Static directed network and exact immutable forecast demand.
        policy: Shared capacity, handling, cost, SLA, mandatory-horizon and
            validation rules used by every previous baseline.

    Returns and connected layers:
        Inherited :meth:`build` returns a validator-approved
        ``BaselineBuildResult`` only when at least one route reaches
        ``MIN_MULTI_STOP_DELIVERIES`` stops and makes the complete plan strictly
        cheaper than a fresh 6D reference.
        The implementation reuses 6B partitioning and mandatory filling, 6D
        pair seeds, transactional ``HandlingCapacityLedger`` trials and the
        authoritative ``CostCalculator``/validator finalization path.

    Strategy boundary and failures:
        Bins must share service date and origin, use distinct destinations,
        preserve whole ``DemandPart`` records and fit one non-Tır spot vehicle.
        Each added stop is tried only at insertion positions around the current
        route order.  Missing arcs, same-day load failure or non-positive
        incremental TL rejects that insertion.  If the full plan does not beat
        6D, ``BaselineBuildError`` leaves 6D as the safe caller fallback.
    """

    def _build_result(self, started_at: float) -> BaselineBuildResult:
        """Build one 6E attempt and require strict complete-plan gain over 6D."""

        reference = GreedySameOriginTwoStopBuilder(
            self.problem,
            self.policy,
        ).build()
        return self._build_against_reference(reference, started_at)

    def build_from_two_stop(
        self,
        reference: BaselineBuildResult,
    ) -> BaselineBuildResult:
        """Build 6E from a caller-supplied valid 6D result without rebuilding it.

        Args:
            reference: Validator-approved two-stop result for the same problem
                and policy.  Comparison/export CLIs use this to report 6D and
                6E without paying the 6D runtime twice.

        Returns:
            Strictly cheaper validator-approved multi-stop result.

        Raises:
            BaselineBuildError: If the reference is invalid, lacks a two-stop
                spot route, its atomic parts differ from this problem, or the
                completed 6E attempt fails to beat it.  Handling/matrix input
                failures are converted to structured baseline errors.
        """

        started_at = perf_counter()
        try:
            self._require_two_stop_reference(reference)
            return self._build_against_reference(reference, started_at)
        except BaselineBuildError:
            raise
        except HandlingCapacityError as exc:
            raise BaselineBuildError(
                str(exc),
                error_codes=("handling_capacity_unavailable",),
            ) from exc
        except (KeyError, ValueError) as exc:
            raise BaselineBuildError(
                f"Çok-duraklı plan kurulamadı: {exc}",
                error_codes=("multi_stop_input_invalid",),
            ) from exc

    def _build_against_reference(
        self,
        reference: BaselineBuildResult,
        started_at: float,
    ) -> BaselineBuildResult:
        """Require a multi-stop route and strict full-plan saving versus 6D."""

        attempt = self._build_multistop_attempt(reference)
        has_multi_stop = any(
            len(route.trip.legs) >= MIN_MULTI_STOP_DELIVERIES
            for route in attempt.routes
        )
        is_cheaper = (
            attempt.result.total_cost_tl
            < reference.total_cost_tl - self.policy.cost_tolerance_tl
        )
        if has_multi_stop and is_cheaper:
            return self._with_elapsed_time(attempt.result, started_at)
        raise BaselineBuildError(
            "Çok-duraklı strateji 6D'ye karşı kesin toplam tasarruf üretmedi.",
            error_codes=("multi_stop_no_strict_saving",),
        )

    def _require_two_stop_reference(
        self,
        reference: BaselineBuildResult,
    ) -> None:
        """Check validity, route shape and exact atomic parts of supplied 6D."""

        if not reference.validation_report.is_valid:
            raise BaselineBuildError(
                "6E için verilen 6D referansı validator'dan geçmiyor.",
                error_codes=("multi_stop_reference_invalid",),
            )
        has_two_stop = any(
            trip.ownership_type == "spot" and len(trip.legs) == 2
            for trip in reference.plan.trips
        )
        if not has_two_stop:
            raise BaselineBuildError(
                "6E referansında iki-duraklı spot rota bulunmuyor.",
                error_codes=("multi_stop_reference_missing_two_stop",),
            )
        expected_parts = self._partition_parts()
        if reference.plan.demand_parts != expected_parts:
            raise BaselineBuildError(
                "6E referansının atomik DemandPart kayıtları problemle uyuşmuyor.",
                error_codes=("multi_stop_reference_parts_mismatch",),
            )

    def _build_multistop_attempt(
        self,
        reference: BaselineBuildResult,
    ) -> _MultiStopAttempt:
        """Build mandatory, bounded routes and remaining directs on fresh state."""

        parts = self._partition_parts()
        ledger = self._empty_ledger()
        # This attempt rebuilds from scratch, so the Tır ledger must reset too;
        # otherwise a second attempt on the same builder would see quota already
        # consumed by the first one and silently pack smaller bins.  6D resets
        # for the same reason in ``_build_attempt``.
        self._truck_ledger = self._new_truck_ledger()
        mandatory, assigned_ids = self._build_forecast_mandatory(parts, ledger)
        remaining = tuple(
            part for part in parts if part.part_id not in assigned_ids
        )
        bins = self._logical_spot_bins(remaining)
        calculator = CostCalculator(self.problem, self.policy)
        routes = self._select_and_grow_routes(
            bins,
            ledger,
            calculator,
            reference,
        )
        used = {index for route in routes for index in route.key}
        direct = tuple(
            self._schedule_direct_bin(item, ledger)
            for item in bins
            if item.index not in used
        )
        spot = tuple(route.trip for route in routes) + direct
        result = self._finalize_plan(parts, mandatory, spot, ledger)
        return _MultiStopAttempt(result, routes)

    def _select_and_grow_routes(
        self,
        bins: tuple[_SpotBin, ...],
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> tuple[_RouteEvaluation, ...]:
        """Rank 6D seeds, grow each before commit, and keep bins disjoint."""

        scored: list[_PairEvaluation] = []
        for candidate in self._candidate_pairs(bins):
            evaluation = self._evaluate_pair(
                candidate,
                ledger,
                calculator,
                reference,
            )
            if (
                evaluation is not None
                and evaluation.saving_tl > self.policy.cost_tolerance_tl
            ):
                scored.append(evaluation)
        used: set[int] = set()
        selected: list[_RouteEvaluation] = []
        for preliminary in sorted(scored, key=self._evaluation_sort_key):
            if any(index in used for index in preliminary.candidate.key):
                continue
            actual = self._evaluate_pair(
                preliminary.candidate,
                ledger,
                calculator,
                reference,
            )
            if actual is None or actual.saving_tl <= self.policy.cost_tolerance_tl:
                continue
            seed = self._route_from_pair(actual)
            grown = self._grow_route(
                seed,
                bins,
                used,
                ledger,
                calculator,
                reference,
            )
            ledger.commit_from(grown.trial_ledger)
            selected.append(grown)
            used.update(grown.key)
        return tuple(selected)

    @staticmethod
    def _route_from_pair(pair: _PairEvaluation) -> _RouteEvaluation:
        """Convert a scheduled 6D pair into an ordered insertion seed."""

        by_destination = {
            pair.candidate.first_bin.destination: pair.candidate.first_bin,
            pair.candidate.second_bin.destination: pair.candidate.second_bin,
        }
        ordered = tuple(
            by_destination[leg.destination] for leg in pair.trip.legs
        )
        return _RouteEvaluation(
            ordered,
            pair.trip,
            pair.trial_ledger,
            pair.saving_tl,
            pair.combined_total_tl,
            pair.distance_km,
            pair.finish_at,
        )

    def _grow_route(
        self,
        seed: _RouteEvaluation,
        bins: tuple[_SpotBin, ...],
        already_used: set[int],
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> _RouteEvaluation:
        """Repeatedly accept the best strictly positive third/fourth insertion."""

        current = seed
        while len(current.ordered_bins) < MAX_DELIVERY_STOPS:
            options: list[_RouteEvaluation] = []
            for extra in bins:
                if not self._compatible_extra(current, extra, already_used):
                    continue
                evaluation = self._evaluate_insertion(
                    current,
                    extra,
                    ledger,
                    calculator,
                    reference,
                )
                if (
                    evaluation is not None
                    and evaluation.saving_tl
                    > current.saving_tl + self.policy.cost_tolerance_tl
                ):
                    options.append(evaluation)
            if not options:
                break
            current = min(
                options,
                key=lambda item: self._insertion_sort_key(item, current),
            )
        return current

    def _compatible_extra(
        self,
        current: _RouteEvaluation,
        extra: _SpotBin,
        already_used: set[int],
    ) -> bool:
        """Check exclusivity, group, destination and any non-Tır capacity."""

        current_indices = set(current.key)
        first = current.ordered_bins[0]
        if extra.index in already_used or extra.index in current_indices:
            return False
        if extra.service_date != first.service_date or extra.origin != first.origin:
            return False
        if extra.destination in {item.destination for item in current.ordered_bins}:
            return False
        return bool(self._eligible_route_vehicles(current.ordered_bins + (extra,)))

    def _evaluate_insertion(
        self,
        current: _RouteEvaluation,
        extra: _SpotBin,
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> _RouteEvaluation | None:
        """Compare current-route-plus-direct with every cheapest insertion slot."""

        candidate_bins = current.ordered_bins + (extra,)
        direct_ledger = ledger.fork()
        try:
            direct_trips = tuple(
                self._schedule_direct_bin(item, direct_ledger)
                for item in sorted(candidate_bins, key=lambda value: value.index)
            )
        except HandlingCapacityError:
            return None
        parts = self._route_parts(candidate_bins)
        direct_cost = calculator.calculate(TransportPlan(parts, direct_trips))
        direct_finish = self._spot_completion_at(direct_trips)
        direct_total = direct_cost.total_cost_tl + self._horizon_surcharge(
            direct_finish,
            reference,
        )
        options = self._insertion_options(
            current,
            extra,
            ledger,
            calculator,
            reference,
        )
        if not options:
            return None
        best = min(options, key=self._route_option_key)
        return _RouteEvaluation(
            best.ordered_bins,
            best.trip,
            best.trial_ledger,
            direct_total - best.combined_total_tl,
            best.combined_total_tl,
            best.distance_km,
            best.finish_at,
        )

    def _insertion_options(
        self,
        current: _RouteEvaluation,
        extra: _SpotBin,
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> tuple[_RouteEvaluation, ...]:
        """Try every insertion position and every capacity-feasible vehicle."""

        options: list[_RouteEvaluation] = []
        base = current.ordered_bins
        all_bins = base + (extra,)
        for position in range(len(base) + 1):
            order = base[:position] + (extra,) + base[position:]
            for vehicle in self._eligible_route_vehicles(all_bins):
                trial = ledger.fork()
                trip = self._schedule_route(order, vehicle, trial)
                if trip is None:
                    continue
                cost = calculator.calculate(
                    TransportPlan(self._route_parts(order), (trip,))
                )
                finish = self._spot_completion_at((trip,))
                total = cost.total_cost_tl + self._horizon_surcharge(
                    finish,
                    reference,
                )
                options.append(
                    _RouteEvaluation(
                        order,
                        trip,
                        trial,
                        0.0,
                        total,
                        cost.vehicle_cost_lines[0].distance_km,
                        finish or trip.legs[-1].arrival_at,
                    )
                )
        return tuple(options)

    def _eligible_route_vehicles(
        self,
        bins: tuple[_SpotBin, ...],
    ) -> tuple[VehicleType, ...]:
        """Return non-Tır types that hold the complete initial route load."""

        total_desi = math.fsum(item.total_desi for item in bins)
        return tuple(
            sorted(
                (
                    vehicle
                    for vehicle in self._vehicle_types.values()
                    if vehicle.name != self.policy.truck_vehicle_type
                    and total_desi
                    <= vehicle.capacity_desi + self.policy.desi_tolerance
                ),
                key=lambda vehicle: (vehicle.capacity_desi, vehicle.name),
            )
        )

    def _schedule_route(
        self,
        ordered_bins: tuple[_SpotBin, ...],
        vehicle: VehicleType,
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip | None:
        """Schedule one same-day load and ordered destination-specific unloads."""

        first = ordered_bins[0]
        total_desi = math.fsum(item.total_desi for item in ordered_bins)
        ready_at = max(item.ready_at for item in ordered_bins)
        # Same-day loading is a strategy-local restriction, not a competition
        # rule, and the direct bins this route is compared against never had it.
        # See two_stop._schedule_two_stop for the same reasoning.
        try:
            load_interval = ledger.schedule(first.origin, ready_at, total_desi)
        except HandlingCapacityError:
            return None
        arrivals: list[datetime] = []
        unloads: list[tuple[datetime, datetime]] = []
        departure_at = load_interval[1]
        current_center = first.origin
        try:
            for stop in ordered_bins:
                arc = self._arcs[(current_center, stop.destination)]
                arrival_at = departure_at + timedelta(
                    hours=arc.travel_hours_for(vehicle.name)
                )
                unload = ledger.schedule(
                    stop.destination,
                    arrival_at,
                    stop.total_desi,
                )
                arrivals.append(arrival_at)
                unloads.append(unload)
                departure_at = unload[1]
                current_center = stop.destination
        except (KeyError, HandlingCapacityError):
            return None
        return self._route_models(
            ordered_bins,
            vehicle,
            load_interval,
            tuple(arrivals),
            tuple(unloads),
        )

    @staticmethod
    def _route_models(
        ordered_bins: tuple[_SpotBin, ...],
        vehicle: VehicleType,
        load_interval: tuple[datetime, datetime],
        arrivals: tuple[datetime, ...],
        unloads: tuple[tuple[datetime, datetime], ...],
    ) -> VehicleTrip:
        """Create one load, N shrinking-load legs and N exact unloads."""

        key = tuple(sorted(item.index for item in ordered_bins))
        stem = "MSOD-" + "-".join(f"{index:05d}" for index in key)
        all_parts = tuple(
            part for stop in ordered_bins for part in stop.parts
        )
        load = HandlingOperation(
            f"{stem}-LOAD",
            stem,
            ordered_bins[0].origin,
            "load",
            load_interval[0],
            load_interval[1],
            math.fsum(part.desi for part in all_parts),
            tuple(part.part_id for part in all_parts),
        )
        unload_operations = tuple(
            HandlingOperation(
                f"{stem}-UNLOAD-{index + 1}",
                stem,
                stop.destination,
                "unload",
                interval[0],
                interval[1],
                stop.total_desi,
                tuple(part.part_id for part in stop.parts),
            )
            for index, (stop, interval) in enumerate(
                zip(ordered_bins, unloads, strict=True)
            )
        )
        legs = GreedySameOriginMultiStopBuilder._route_legs(
            stem,
            ordered_bins,
            load,
            unload_operations,
            arrivals,
        )
        return VehicleTrip(
            f"{stem}-TRIP",
            stem,
            vehicle.name,
            "spot",
            legs,
            (load,) + unload_operations,
        )

    @staticmethod
    def _route_legs(
        stem: str,
        ordered_bins: tuple[_SpotBin, ...],
        load: HandlingOperation,
        unloads: tuple[HandlingOperation, ...],
        arrivals: tuple[datetime, ...],
    ) -> tuple[VehicleLeg, ...]:
        """Build leg assignments whose carried load shrinks after every stop."""

        legs: list[VehicleLeg] = []
        for index, stop in enumerate(ordered_bins):
            assignments = tuple(
                DemandAssignment(
                    part.part_id,
                    part.desi,
                    loaded_at_origin=index == 0,
                    unloaded_at_destination=remaining_index == index,
                    is_final_delivery=remaining_index == index,
                )
                for remaining_index in range(index, len(ordered_bins))
                for part in ordered_bins[remaining_index].parts
            )
            origin = (
                ordered_bins[0].origin
                if index == 0
                else ordered_bins[index - 1].destination
            )
            departure = load.end_at if index == 0 else unloads[index - 1].end_at
            legs.append(
                VehicleLeg(
                    f"{stem}-LEG-{index + 1}",
                    origin,
                    stop.destination,
                    departure,
                    arrivals[index],
                    assignments,
                    load.operation_id if index == 0 else None,
                    unloads[index].operation_id,
                )
            )
        return tuple(legs)

    @staticmethod
    def _route_parts(bins: tuple[_SpotBin, ...]) -> tuple[DemandPart, ...]:
        """Return unique unchanged atoms in stable source-bin order."""

        return tuple(
            part
            for item in sorted(bins, key=lambda value: value.index)
            for part in item.parts
        )

    @staticmethod
    def _route_option_key(item: _RouteEvaluation) -> tuple:
        """Choose lowest TL, km, completion, route, type and source IDs."""

        return (
            item.combined_total_tl,
            item.distance_km,
            item.finish_at,
            tuple(stop.destination for stop in item.ordered_bins),
            item.trip.vehicle_type,
            item.key,
        )

    @staticmethod
    def _insertion_sort_key(
        item: _RouteEvaluation,
        current: _RouteEvaluation,
    ) -> tuple:
        """Prefer largest incremental TL gain then deterministic route metrics."""

        return (
            -(item.saving_tl - current.saving_tl),
            item.combined_total_tl,
            item.distance_km,
            item.finish_at,
            tuple(stop.destination for stop in item.ordered_bins),
            item.trip.vehicle_type,
            item.key,
        )


__all__ = ["GreedySameOriginMultiStopBuilder"]
