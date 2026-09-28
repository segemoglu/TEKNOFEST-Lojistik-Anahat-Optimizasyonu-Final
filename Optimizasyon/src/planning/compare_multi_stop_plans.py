"""Compare validated two-stop 6D with bounded cheapest-insertion 6E.

Run ``python -m src.planning.compare_multi_stop_plans`` for the configured
development forecast or pass ``--forecast`` for another standard demand
workbook.  The command loads one immutable problem, builds 6D once, grows 6E
from it and reports complete CostCalculator/validator metrics without writing
an Excel file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path

from .baseline import BaselineBuildResult
from .multi_stop import GreedySameOriginMultiStopBuilder
from .two_stop import GreedySameOriginTwoStopBuilder


@dataclass(frozen=True, slots=True)
class MultiStopComparison:
    """Hold the common problem plus validator-approved 6D and 6E results."""

    problem: ProblemData
    two_stop: BaselineBuildResult
    multi_stop: BaselineBuildResult


def build_parser() -> argparse.ArgumentParser:
    """Create Unicode-safe manifest and forecast CLI arguments."""

    parser = argparse.ArgumentParser(
        description="İki-duraklı 6D ile üç/dört-duraklı 6E'yi karşılaştırır."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    return parser


def compare_multi_stop_strategies(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
) -> MultiStopComparison:
    """Load one forecast, build 6D once and grow strict-improvement 6E.

    Args:
        manifest_path: Optional central rules/paths config.
        forecast_path: Optional standard forecast workbook; ``None`` selects
            ``paths.development_forecast`` inside the inferred project root.

    Returns:
        Common ``ProblemData`` and validator-approved 6D/6E results.

    Raises:
        Loader, planning, cost or validation errors propagate.  6E raises a
        structured ``BaselineBuildError`` rather than reporting success when
        the full plan does not strictly beat 6D.
    """

    loader = ExcelProblemLoader(manifest_path)
    static = loader.load_static_problem_data()
    selected_forecast = (
        Path(forecast_path).resolve()
        if forecast_path is not None
        else configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=loader.project_root,
        )
    )
    demands = loader.load_demands_from_workbook(
        selected_forecast,
        static_data=static,
    )
    problem = ProblemData(static, demands)
    policy = EvaluationPolicy.from_manifest(loader.manifest)
    two_stop = GreedySameOriginTwoStopBuilder(problem, policy).build()
    multi_stop = GreedySameOriginMultiStopBuilder(
        problem,
        policy,
    ).build_from_two_stop(two_stop)
    return MultiStopComparison(problem, two_stop, multi_stop)


def render_comparison(comparison: MultiStopComparison) -> str:
    """Render validator, TL, route, fill, km, SLA, horizon and runtime metrics."""

    base = comparison.two_stop
    result = comparison.multi_stop
    base_metrics = _route_metrics(base, comparison.problem)
    result_metrics = _route_metrics(result, comparison.problem)
    saving = base.total_cost_tl - result.total_cost_tl
    lines = [
        f"Validator (6D / 6E): {base.validation_report.is_valid} / "
        f"{result.validation_report.is_valid}",
        f"Toplam maliyet TL (6D / 6E): {base.total_cost_tl:.2f} / "
        f"{result.total_cost_tl:.2f}",
        f"6D'ye göre tasarruf: {saving:.2f} TL "
        f"(%{_ratio(saving, base.total_cost_tl) * 100.0:.2f})",
        f"Spot araç (6D / 6E): {base.spot_vehicle_count} / "
        f"{result.spot_vehicle_count}",
        f"Bacak sayısı dağılımı 6D: {dict(base_metrics['leg_counts'])}",
        f"Bacak sayısı dağılımı 6E: {dict(result_metrics['leg_counts'])}",
        f"3/4 duraklı rota: {result_metrics['three_stop']} / "
        f"{result_metrics['four_stop']}",
        f"Çok-duraklı taşınan desi: {result_metrics['multi_stop_desi']:.6f} "
        f"(%{result_metrics['demand_ratio'] * 100.0:.2f})",
        f"Çok-duraklı doluluk ortalama / maksimum: "
        f"%{result_metrics['average_fill'] * 100.0:.2f} / "
        f"%{result_metrics['maximum_fill'] * 100.0:.2f}",
        f"Çok-duraklı araç türleri: {dict(result_metrics['vehicle_types'])}",
        f"Spot km (6D / 6E / fark): {base_metrics['spot_km']:.2f} / "
        f"{result_metrics['spot_km']:.2f} / "
        f"{result_metrics['spot_km'] - base_metrics['spot_km']:.2f}",
        f"SLA TL (6D / 6E / fark): {base_metrics['sla_tl']:.6f} / "
        f"{result_metrics['sla_tl']:.6f} / "
        f"{result_metrics['sla_tl'] - base_metrics['sla_tl']:.6f}",
        f"Son teslim (6D / 6E): {base.completion_at} / {result.completion_at}",
        f"Zorunlu araç (6D / 6E): {base.mandatory_vehicle_count} / "
        f"{result.mandatory_vehicle_count}",
        f"En yüksek handling / Tır kota: "
        f"%{_metric(result, 'peak_handling_capacity_ratio') * 100.0:.2f} / "
        f"%{_metric(result, 'peak_truck_quota_ratio') * 100.0:.2f}",
        f"6E ek çalışma süresi: {result.build_seconds:.3f} saniye",
    ]
    return "\n".join(lines)


def _route_metrics(
    result: BaselineBuildResult,
    problem: ProblemData,
) -> dict[str, object]:
    """Calculate spot route-size, initial-load, fill, type, km and SLA metrics."""

    spot = tuple(
        trip for trip in result.plan.trips if trip.ownership_type == "spot"
    )
    multi = tuple(trip for trip in spot if len(trip.legs) >= 3)
    capacities = {
        vehicle.name: vehicle.capacity_desi
        for vehicle in problem.static.vehicle_types
    }
    loads = [
        math.fsum(item.desi for item in trip.legs[0].assignments)
        for trip in multi
    ]
    fills = [
        load / capacities[trip.vehicle_type]
        for load, trip in zip(loads, multi, strict=True)
    ]
    total_demand = math.fsum(part.desi for part in result.plan.demand_parts)
    cost = result.plan.reported_cost
    return {
        "leg_counts": Counter(len(trip.legs) for trip in spot),
        "three_stop": sum(len(trip.legs) == 3 for trip in spot),
        "four_stop": sum(len(trip.legs) == 4 for trip in spot),
        "multi_stop_desi": math.fsum(loads),
        "demand_ratio": _ratio(math.fsum(loads), total_demand),
        "average_fill": mean(fills) if fills else 0.0,
        "maximum_fill": max(fills, default=0.0),
        "vehicle_types": Counter(trip.vehicle_type for trip in multi),
        "spot_km": math.fsum(
            line.distance_km
            for line in cost.vehicle_cost_lines
            if line.ownership_type == "spot"
        ),
        "sla_tl": cost.sla_penalty_tl,
    }


def _metric(result: BaselineBuildResult, key: str) -> float:
    """Read one numeric validator metric while preserving explicit zero."""

    return float(result.validation_report.metrics.get(key, 0.0))


def _ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using zero when the denominator is zero."""

    return numerator / denominator if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    """Build and print the 6D/6E comparison, returning zero on strict success."""

    args = build_parser().parse_args(argv)
    comparison = compare_multi_stop_strategies(args.manifest, args.forecast)
    print(render_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
