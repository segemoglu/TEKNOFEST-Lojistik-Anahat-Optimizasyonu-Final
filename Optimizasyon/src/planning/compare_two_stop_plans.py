"""CLI comparing 6B with the same-origin two-destination 6D baseline.

Run ``python -m src.planning.compare_two_stop_plans`` for the configured
development forecast or pass ``--forecast`` for another standard demand
workbook.  The command builds validated in-memory plans, reports economic and
physical metrics, and never writes a transport-plan workbook or raw input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean

from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader
from src.planning.compare_baseline_plans import (
    PlanComparison,
    compare_strategies,
)

from .baseline import BaselineBuildResult
from .two_stop import GreedySameOriginTwoStopBuilder


@dataclass(frozen=True, slots=True)
class TwoStopComparison:
    """Hold common-input conservative, 6A, 6B and validated 6D results.

    ``compare_two_stop_strategies`` creates this read-only record for CLI and
    smoke diagnostics.  Costs are SLA-inclusive TL values from each plan's
    ``CostCalculator`` result; no file output or optimization occurs.
    """

    baselines: PlanComparison
    two_stop: BaselineBuildResult


def build_parser() -> argparse.ArgumentParser:
    """Create Unicode/Windows-safe manifest and forecast CLI arguments.

    Returns:
        Parser accepting optional central config and standard forecast paths.
        Malformed arguments are rejected by ``argparse`` before any workbook
        is read; omitted paths use the existing manifest defaults.
    """

    parser = argparse.ArgumentParser(
        description="6B ile aynı-origin iki-duraklı 6D baseline'ı karşılaştırır."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    return parser


def compare_two_stop_strategies(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
) -> TwoStopComparison:
    """Build all baselines and 6D from one immutable problem instance.

    Args:
        manifest_path: Optional central rules/paths config.
        forecast_path: Optional standard forecast workbook; ``None`` selects
            ``paths.development_forecast`` from the manifest.

    Returns:
        ``TwoStopComparison`` containing validator-gated plans.

    Connected modules and errors:
        ``compare_strategies`` performs read-only loading and builds the three
        earlier strategies.  ``GreedySameOriginTwoStopBuilder`` consumes only
        their common ``ProblemData``/policy and raises ``BaselineBuildError``
        if strict savings or validation fails.  No exporter is called.
    """

    baselines = compare_strategies(manifest_path, forecast_path)
    loader = ExcelProblemLoader(manifest_path)
    evaluation_policy = EvaluationPolicy.from_manifest(loader.manifest)
    result = GreedySameOriginTwoStopBuilder(
        baselines.problem,
        evaluation_policy,
    ).build()
    return TwoStopComparison(baselines, result)


def render_comparison(comparison: TwoStopComparison) -> str:
    """Render validator, TL, trip, desi, km, waiting and capacity metrics.

    All values come from validated immutable plan models.  Savings include
    hourly, kilometre, SLA and any mandatory continuation TL.  Empty metric
    collections use zero instead of hidden assumptions or invented fees.
    """

    base = comparison.baselines.mandatory_filled
    result = comparison.two_stop
    metrics = _two_stop_metrics(result, comparison.baselines.problem)
    saving = base.total_cost_tl - result.total_cost_tl
    direct_base = _direct_spot_count(base)
    direct_result = _direct_spot_count(result)
    spot_km_base = _spot_kilometres(base)
    spot_km_result = _spot_kilometres(result)
    lines = [
        f"Fizibilite: {'GEÇERLİ VE DAHA UCUZ' if saving > 0 and metrics['trip_count'] else 'BAŞARISIZ'}",
        f"Validator (6B / 6D): {base.validation_report.is_valid} / {result.validation_report.is_valid}",
        f"Toplam maliyet TL (6B / 6D): {base.total_cost_tl:.2f} / {result.total_cost_tl:.2f}",
        f"6B tasarrufu: {saving:.2f} TL (%{_ratio(saving, base.total_cost_tl) * 100.0:.2f})",
        f"6A toplam tasarrufu: {comparison.baselines.consolidated.total_cost_tl - result.total_cost_tl:.2f} TL",
        f"Konservatif toplam tasarrufu: {comparison.baselines.conservative.total_cost_tl - result.total_cost_tl:.2f} TL",
        f"Doğrudan spot trip (6B / 6D): {direct_base} / {direct_result}",
        f"İki duraklı trip: {metrics['trip_count']}",
        f"İki duraklı taşınan desi: {metrics['carried_desi']:.6f} (%{metrics['demand_ratio'] * 100.0:.2f})",
        f"Birleşik doluluk ortalama / maksimum: %{metrics['average_fill'] * 100.0:.2f} / %{metrics['maximum_fill'] * 100.0:.2f}",
        f"Araç türleri: {dict(metrics['vehicle_types'])}",
        f"1. / 2. durak teslim desi: {metrics['first_delivery_desi']:.6f} / {metrics['second_delivery_desi']:.6f}",
        f"Spot km (6B / 6D / fark): {spot_km_base:.2f} / {spot_km_result:.2f} / {spot_km_result - spot_km_base:.2f}",
        f"Durak bekleme dakika toplam / ortalama / maksimum: {metrics['wait_total']:.6f} / {metrics['wait_average']:.6f} / {metrics['wait_maximum']:.6f}",
        f"SLA TL (6B / 6D / fark): {base.plan.reported_cost.sla_penalty_tl:.6f} / {result.plan.reported_cost.sla_penalty_tl:.6f} / {result.plan.reported_cost.sla_penalty_tl - base.plan.reported_cost.sla_penalty_tl:.6f}",
        f"Son teslim: {result.completion_at}",
        f"Zorunlu araç sayısı: {result.mandatory_vehicle_count}",
        f"En yüksek handling / Tır kota: %{_metric(result, 'peak_handling_capacity_ratio') * 100.0:.2f} / %{_metric(result, 'peak_truck_quota_ratio') * 100.0:.2f}",
        f"6D çalışma süresi: {result.build_seconds:.3f} saniye",
    ]
    return "\n".join(lines)


def _two_stop_metrics(result: BaselineBuildResult, problem) -> dict[str, object]:
    """Calculate two-stop desi, fill, type, delivery and waiting diagnostics."""

    trips = tuple(
        trip for trip in result.plan.trips
        if trip.ownership_type == "spot" and len(trip.legs) == 2
    )
    capacities = {item.name: item.capacity_desi for item in problem.static.vehicle_types}
    loads = [math.fsum(item.desi for item in trip.legs[0].assignments) for trip in trips]
    fills = [load / capacities[trip.vehicle_type] for load, trip in zip(loads, trips)]
    waits = [
        max(0.0, (operation.start_at - leg.arrival_at).total_seconds() / 60.0)
        for trip in trips
        for leg in trip.legs
        for operation in trip.handling_operations
        if leg.unloading_operation_id == operation.operation_id
    ]
    total_demand = math.fsum(part.desi for part in result.plan.demand_parts)
    carried = math.fsum(loads)
    return {
        "trip_count": len(trips), "carried_desi": carried,
        "demand_ratio": _ratio(carried, total_demand),
        "average_fill": mean(fills) if fills else 0.0,
        "maximum_fill": max(fills, default=0.0),
        "vehicle_types": Counter(trip.vehicle_type for trip in trips),
        "first_delivery_desi": _delivered_on_leg(trips, 0),
        "second_delivery_desi": _delivered_on_leg(trips, 1),
        "wait_total": math.fsum(waits),
        "wait_average": mean(waits) if waits else 0.0,
        "wait_maximum": max(waits, default=0.0),
    }


def _delivered_on_leg(trips: tuple, index: int) -> float:
    """Sum final-delivery desi on one zero-based stop leg."""

    return math.fsum(
        assignment.desi
        for trip in trips
        for assignment in trip.legs[index].assignments
        if assignment.is_final_delivery
    )


def _direct_spot_count(result: BaselineBuildResult) -> int:
    """Count single-leg spot trips in one validated baseline result."""

    return sum(
        trip.ownership_type == "spot" and len(trip.legs) == 1
        for trip in result.plan.trips
    )


def _spot_kilometres(result: BaselineBuildResult) -> float:
    """Sum every spot vehicle cost line's matrix distance in kilometres."""

    return math.fsum(
        line.distance_km
        for line in result.plan.reported_cost.vehicle_cost_lines
        if line.ownership_type == "spot"
    )


def _metric(result: BaselineBuildResult, key: str) -> float:
    """Read one numeric validator metric while preserving an explicit zero."""

    return float(result.validation_report.metrics.get(key, 0.0))


def _ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using zero when the denominator is zero."""

    return numerator / denominator if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    """Build and print the read-only 6B/6D comparison, returning process status.

    Args:
        argv: Optional CLI arguments without the executable name.

    Returns:
        ``0`` after valid strictly cheaper 6D output.  Loader, planning,
        costing or validation failures propagate with non-zero interpreter
        status and no workbook mutation.
    """

    args = build_parser().parse_args(argv)
    comparison = compare_two_stop_strategies(args.manifest, args.forecast)
    print(render_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
