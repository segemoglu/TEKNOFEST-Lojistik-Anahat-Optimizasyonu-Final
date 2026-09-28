"""Replaceable demand-forecast providers for development-time planning.

This module converts source-independent historical ``DemandInstance`` records
into another immutable ``DemandInstance`` tuple.  It contains no optimizer,
route generation, serialized model loading or Excel I/O.  Downstream planning
code should consume only the returned tuple, not a concrete provider class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, time, timedelta
import math
from types import MappingProxyType
from typing import Any, Literal, Protocol

from src.domain import DemandInstance, StaticProblemData


ForecastMethod = Literal["primary", "od_time", "od_overall", "zero"]
MINIMUM_DEMAND_ID_DIGITS = 5
DEMAND_ID_PREFIX = "D"


@dataclass(frozen=True, slots=True)
class ForecastWindow:
    """Define inclusive forecast dates and ordered daily completion slots.

    ``HistoricalMeanForecastProvider`` consumes this immutable value.  Slot
    order is preserved in deterministic output ordering; duplicate slots or a
    reversed/empty window raise ``ValueError`` before forecasting begins.
    """

    start_date: date
    end_date: date
    time_slots: tuple[time, ...]

    def __post_init__(self) -> None:
        """Freeze slots and reject an unusable forecast range."""

        object.__setattr__(self, "time_slots", tuple(self.time_slots))
        if self.end_date < self.start_date:
            raise ValueError("Forecast bitiş tarihi başlangıçtan erken olamaz.")
        if not self.time_slots:
            raise ValueError("Forecast en az bir günlük saat içermelidir.")
        if any(not isinstance(slot, time) for slot in self.time_slots):
            raise ValueError("Forecast saatleri datetime.time olmalıdır.")
        if len(self.time_slots) != len(set(self.time_slots)):
            raise ValueError("Forecast saatleri tekrar edemez.")


@dataclass(frozen=True, slots=True)
class ForecastStatistics:
    """Summarize one generated forecast and its fallback method usage.

    The development CLI prints these record, route, desi and split counts.
    ``method_counts`` is immutable and uses desi-independent record counts;
    malformed negative counts are rejected with ``ValueError``.
    """

    record_count: int
    active_route_count: int
    total_forecast_desi: float
    split_required_count: int
    method_counts: Mapping[ForecastMethod, int]

    def __post_init__(self) -> None:
        """Protect method counts and reject negative summary values."""

        counts = dict(self.method_counts)
        if any(value < 0 for value in counts.values()):
            raise ValueError("Forecast yöntem sayaçları negatif olamaz.")
        if self.record_count < 0 or self.active_route_count < 0:
            raise ValueError("Forecast kayıt ve aktif hat sayısı negatif olamaz.")
        if self.split_required_count < 0 or self.total_forecast_desi < 0:
            raise ValueError("Forecast desi ve split sayısı negatif olamaz.")
        object.__setattr__(self, "method_counts", MappingProxyType(counts))


@dataclass(frozen=True, slots=True)
class ForecastBatch:
    """Pair standard demand records with development-only diagnostics.

    CLI and tests use the statistics, while downstream planning layers should
    pass only ``demands`` into ``ProblemData`` or another consumer.  Inputs are
    normalized to an immutable tuple; generation errors occur in the provider.
    """

    demands: tuple[DemandInstance, ...]
    statistics: ForecastStatistics

    def __post_init__(self) -> None:
        """Freeze demand collections supplied by provider implementations."""

        object.__setattr__(self, "demands", tuple(self.demands))


class ForecastProvider(Protocol):
    """Contract for replaceable sources of standard forecast demands.

    Implementations receive their own data/config dependencies at construction
    and return deterministic ``DemandInstance`` records from :meth:`generate`.
    Future competition forecasts can implement this protocol without changing
    optimizer-facing code.  Provider-specific I/O or validation failures may
    be raised explicitly by the concrete implementation.
    """

    def generate(self) -> tuple[DemandInstance, ...]:
        """Return one immutable, source-independent forecast demand collection."""

        ...


@dataclass(frozen=True, slots=True)
class _HistoricalMeanIndex:
    """Store valid historical desi values at the three lookup granularities."""

    primary: Mapping[tuple[str, str, int, time], tuple[float, ...]]
    od_time: Mapping[tuple[str, str, time], tuple[float, ...]]
    od_overall: Mapping[tuple[str, str], tuple[float, ...]]

    def resolve(
        self,
        origin: str,
        destination: str,
        weekday: int,
        slot: time,
    ) -> tuple[float, ForecastMethod]:
        """Return the first available arithmetic mean in the fixed fallback order."""

        candidates: tuple[tuple[tuple[float, ...] | None, ForecastMethod], ...] = (
            (self.primary.get((origin, destination, weekday, slot)), "primary"),
            (self.od_time.get((origin, destination, slot)), "od_time"),
            (self.od_overall.get((origin, destination)), "od_overall"),
        )
        for values, method in candidates:
            if values:
                # Dividing before fsum avoids overflow while preserving the
                # exact arithmetic-mean rule for non-negative desi values.
                return math.fsum(value / len(values) for value in values), method
        return 0.0, "zero"


@dataclass(frozen=True, slots=True)
class HistoricalMeanForecastProvider:
    """Generate a deterministic development forecast from historical means.

    Args:
        historical_demands: Potential history; only records strictly before
            ``window.start_date`` are eligible, preventing forecast leakage.
        static_data: Vehicle master data used only for the maximum desi capacity
            and resulting ``split_required`` marker.
        window: Inclusive dates and ordered completion times to generate.

    Returns and connections:
        :meth:`generate` satisfies ``ForecastProvider`` and returns standard
        domain records.  The CLI calls :meth:`generate_with_statistics` for
        fallback counts.  No demand parts or vehicle-capacity splits are made.

    Important rules and errors:
        Active O-D lines come from pre-window records, not the static network.
        Invalid negative/non-finite historical desi is excluded from means so
        output remains finite and non-negative.  Invalid windows are rejected
        by ``ForecastWindow``; an empty eligible history yields no records.
    """

    historical_demands: tuple[DemandInstance, ...]
    static_data: StaticProblemData
    window: ForecastWindow

    def __post_init__(self) -> None:
        """Freeze historical input so repeated runs see the same records."""

        object.__setattr__(
            self,
            "historical_demands",
            tuple(self.historical_demands),
        )

    def generate(self) -> tuple[DemandInstance, ...]:
        """Return deterministic ``DemandInstance`` records without diagnostics."""

        return self.generate_with_statistics().demands

    def generate_with_statistics(self) -> ForecastBatch:
        """Generate forecast records plus route, desi, split and method counts."""

        history = tuple(
            demand
            for demand in self.historical_demands
            if demand.demand_date < self.window.start_date
        )
        active_routes = sorted(
            {(demand.origin, demand.destination) for demand in history}
        )
        means = _build_mean_index(history)
        forecast_dates = tuple(_inclusive_dates(self.window))
        record_count = len(forecast_dates) * len(self.window.time_slots) * len(active_routes)
        id_digits = max(MINIMUM_DEMAND_ID_DIGITS, len(str(record_count)))
        demands: list[DemandInstance] = []
        method_counts: Counter[ForecastMethod] = Counter()
        for forecast_date in forecast_dates:
            for slot in self.window.time_slots:
                for origin, destination in active_routes:
                    desi, method = means.resolve(
                        origin,
                        destination,
                        forecast_date.weekday(),
                        slot,
                    )
                    method_counts[method] += 1
                    demands.append(
                        self._build_demand(
                            len(demands) + 1,
                            id_digits,
                            forecast_date,
                            slot,
                            origin,
                            destination,
                            desi,
                        )
                    )
        return ForecastBatch(
            tuple(demands),
            _forecast_statistics(demands, active_routes, method_counts),
        )

    def _build_demand(
        self,
        sequence: int,
        id_digits: int,
        forecast_date: date,
        slot: time,
        origin: str,
        destination: str,
        desi: float,
    ) -> DemandInstance:
        """Create one finite, non-negative forecast record with a stable ID."""

        safe_desi = desi if math.isfinite(desi) and desi >= 0 else 0.0
        return DemandInstance(
            demand_id=f"{DEMAND_ID_PREFIX}{sequence:0{id_digits}d}",
            demand_date=forecast_date,
            time_slot=slot,
            origin=origin,
            destination=destination,
            desi=float(safe_desi),
            split_required=(
                safe_desi > self.static_data.maximum_vehicle_capacity_desi
            ),
        )


def forecast_window_from_manifest(manifest: Mapping[str, Any]) -> ForecastWindow:
    """Parse the central manifest's inclusive dates and ordered time slots.

    ``generate_development_forecast`` calls this function after ``load_manifest``.
    ISO dates and times are converted with the standard library; missing,
    malformed, duplicate or reversed values raise ``ValueError`` rather than
    silently selecting another forecast period.
    """

    rules = manifest.get("rules")
    forecast = rules.get("forecast_window") if isinstance(rules, Mapping) else None
    if not isinstance(forecast, Mapping):
        raise ValueError("Manifest rules.forecast_window nesne olmalıdır.")
    raw_slots = forecast.get("time_slots")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        raise ValueError("Manifest forecast time_slots liste olmalıdır.")
    try:
        start_date = date.fromisoformat(str(forecast["start_date"]))
        end_date = date.fromisoformat(str(forecast["end_date"]))
        slots = tuple(time.fromisoformat(str(value)) for value in raw_slots)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest forecast_window alanları geçersiz.") from exc
    return ForecastWindow(start_date, end_date, slots)


def _build_mean_index(history: tuple[DemandInstance, ...]) -> _HistoricalMeanIndex:
    """Index valid desi values without rounding or admitting future records."""

    primary: dict[tuple[str, str, int, time], list[float]] = defaultdict(list)
    od_time: dict[tuple[str, str, time], list[float]] = defaultdict(list)
    od_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    for demand in history:
        desi = float(demand.desi)
        if not math.isfinite(desi) or desi < 0:
            continue
        route = (demand.origin, demand.destination)
        primary[(*route, demand.demand_date.weekday(), demand.time_slot)].append(desi)
        od_time[(*route, demand.time_slot)].append(desi)
        od_overall[route].append(desi)
    return _HistoricalMeanIndex(
        {key: tuple(values) for key, values in primary.items()},
        {key: tuple(values) for key, values in od_time.items()},
        {key: tuple(values) for key, values in od_overall.items()},
    )


def _inclusive_dates(window: ForecastWindow) -> Iterator[date]:
    """Yield every local calendar date in the inclusive forecast window."""

    current = window.start_date
    while current <= window.end_date:
        yield current
        current += timedelta(days=1)


def _forecast_statistics(
    demands: list[DemandInstance],
    active_routes: list[tuple[str, str]],
    method_counts: Counter[ForecastMethod],
) -> ForecastStatistics:
    """Build immutable counts and full-precision total forecast desi."""

    complete_counts = {
        method: method_counts[method]
        for method in ("primary", "od_time", "od_overall", "zero")
    }
    return ForecastStatistics(
        record_count=len(demands),
        active_route_count=len(active_routes),
        total_forecast_desi=math.fsum(demand.desi for demand in demands),
        split_required_count=sum(demand.split_required for demand in demands),
        method_counts=complete_counts,
    )


__all__ = [
    "ForecastBatch",
    "ForecastMethod",
    "ForecastProvider",
    "ForecastStatistics",
    "ForecastWindow",
    "HistoricalMeanForecastProvider",
    "forecast_window_from_manifest",
]
