"""CLI comparing conservative, same-O-D and mandatory-filled baselines.

Run ``python -m src.planning.compare_baseline_plans`` with the configured
development forecast, or pass ``--forecast`` for another standard workbook.
All builders receive identical ``ProblemData``/``EvaluationPolicy`` and must
independently validate.  The command prints metrics only and writes no plan
workbook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

from src.domain import ProblemData, VehicleTrip
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path

from .baseline import (
    BaselineBuildError,
    BaselineBuildResult,
    ConservativeDirectBaselineBuilder,
)
from .consolidated import GreedySameODConsolidatedBuilder
from .mandatory_filled import GreedyMandatoryRentalFilledBuilder


@dataclass(frozen=True, slots=True)
class PlanComparison:
    """Hold three validated plans built from exactly the same problem instance.

    ``compare_strategies`` creates this record for the CLI and tests.  Costs
    are available in TL through each ``BaselineBuildResult``; the source
    ``ProblemData`` provides vehicle capacities for desi fill metrics.  No
    mutation, optimization or file output occurs in this model.
    """

    conservative: BaselineBuildResult
    consolidated: BaselineBuildResult
    mandatory_filled: BaselineBuildResult
    problem: ProblemData
    forecast_path: Path


@dataclass(frozen=True, slots=True)
class _MandatoryMetrics:
    """Internal mandatory vehicle, carried desi and fill diagnostics."""

    vehicle_count: int
    loaded_vehicle_count: int
    empty_vehicle_count: int
    multi_part_vehicle_count: int
    carried_desi: float
    carried_desi_ratio: float
    average_fill_ratio: float
    maximum_fill_ratio: float


def build_parser() -> argparse.ArgumentParser:
    """Create manifest and Unicode/Windows-safe forecast path arguments.

    ``argparse`` rejects malformed arguments before input loading.  An omitted
    forecast resolves from the manifest and neither selected workbook nor raw
    data is written.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Konservatif, aynı-O-D FFD ve zorunlu-kiralık doldurma "
            "başlangıç stratejilerini karşılaştırır."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Varsayılan: config/rules.yaml",
    )
    parser.add_argument(
        "--forecast",
        type=Path,
        default=None,
        help="Varsayılan: manifest paths.development_forecast",
    )
    return parser


def compare_strategies(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
) -> PlanComparison:
    """Build all three validated strategies from one forecast collection.

    Args:
        manifest_path: Optional central config path.
        forecast_path: Optional forecast workbook path; ``None`` uses the
            configured development forecast.

    Returns:
        ``PlanComparison`` with conservative, same-O-D consolidated and
        mandatory-filled results, common problem and absolute input path.

    Connected modules and errors:
        ``ExcelProblemLoader`` performs read-only domain conversion.  All
        planning builders cost with ``CostCalculator`` and require the
        independent validator to pass.  Loader/config/``BaselineBuildError``
        failures propagate to :func:`main`; no exporter is invoked.
    """

    loader = ExcelProblemLoader(manifest_path)
    static = loader.load_static_problem_data()
    selected_path = _forecast_path(loader, forecast_path)
    demands = loader.load_demands_from_workbook(
        selected_path,
        static_data=static,
    )
    problem = ProblemData(static, demands)
    policy = EvaluationPolicy.from_manifest(loader.manifest)
    conservative = ConservativeDirectBaselineBuilder(problem, policy).build()
    consolidated = GreedySameODConsolidatedBuilder(problem, policy).build()
    mandatory_filled = GreedyMandatoryRentalFilledBuilder(problem, policy).build()
    return PlanComparison(
        conservative,
        consolidated,
        mandatory_filled,
        problem,
        selected_path,
    )


def render_comparison(comparison: PlanComparison) -> str:
    """Render TL, vehicle, desi fill, completion, SLA and runtime differences.

    Calculations consume already validated plans.  Savings include SLA and all
    vehicle TL.  Mandatory fill ratios divide unchanged carried desi by vehicle
    capacity desi and use no rounded plan quantities.
    """

    conservative = comparison.conservative
    same_od = comparison.consolidated
    filled = comparison.mandatory_filled
    metrics = _mandatory_metrics(filled, comparison.problem)
    incremental_tl = same_od.total_cost_tl - filled.total_cost_tl
    incremental_ratio = _savings_ratio(incremental_tl, same_od.total_cost_tl)
    total_tl = conservative.total_cost_tl - filled.total_cost_tl
    total_ratio = _savings_ratio(total_tl, conservative.total_cost_tl)
    same_od_sla = same_od.plan.reported_cost.sla_penalty_tl
    filled_sla = filled.plan.reported_cost.sla_penalty_tl
    lines = [
        "Baseline strateji karşılaştırması; optimizasyon değildir.",
        f"Forecast: {comparison.forecast_path}",
        "Validator konservatif / 6A / 6B: "
        f"{_validity(conservative)} / {_validity(same_od)} / {_validity(filled)}",
        f"Toplam maliyet 6A / 6B (TL): {same_od.total_cost_tl:.2f} / {filled.total_cost_tl:.2f}",
        f"6B'nin 6A'ya göre tasarrufu (TL / yüzde): {incremental_tl:.2f} / %{incremental_ratio * 100.0:.2f}",
        f"6B'nin konservatife göre tasarrufu (TL / yüzde): {total_tl:.2f} / %{total_ratio * 100.0:.2f}",
        f"Spot araç 6A / 6B: {same_od.spot_vehicle_count} / {filled.spot_vehicle_count}",
        f"Zorunlu araç toplam / yüklü / boş: {metrics.vehicle_count} / {metrics.loaded_vehicle_count} / {metrics.empty_vehicle_count}",
        f"Zorunlu araçla taşınan desi / talep payı: {metrics.carried_desi:.10f} / %{metrics.carried_desi_ratio * 100.0:.2f}",
        "Zorunlu araç ortalama / maksimum doluluk: "
        f"%{metrics.average_fill_ratio * 100.0:.2f} / "
        f"%{metrics.maximum_fill_ratio * 100.0:.2f}",
        f"Birden fazla DemandPart taşıyan zorunlu araç: {metrics.multi_part_vehicle_count}",
        f"SLA cezası 6A / 6B (TL): {same_od_sla:.2f} / {filled_sla:.2f}",
        f"Son teslim 6A / 6B: {_completion(same_od)} / {_completion(filled)}",
        "6B en yüksek elleçleme / Tır kota oranı: "
        f"%{_metric_ratio(filled, 'peak_handling_capacity_ratio') * 100.0:.2f} / "
        f"%{_metric_ratio(filled, 'peak_truck_quota_ratio') * 100.0:.2f}",
        "Çalışma süresi konservatif / 6A / 6B (s): "
        f"{conservative.build_seconds:.3f} / {same_od.build_seconds:.3f} / "
        f"{filled.build_seconds:.3f}",
        "Resmi TAŞIMA PLANI.xlsx üretilmedi.",
    ]
    return "\n".join(lines)


def _forecast_path(
    loader: ExcelProblemLoader,
    forecast_path: str | Path | None,
) -> Path:
    """Resolve an explicit or manifest-selected forecast input path."""

    if forecast_path is not None:
        return Path(forecast_path).resolve()
    return configured_project_path(
        loader.manifest,
        "development_forecast",
        project_root=loader.project_root,
    )


def _mandatory_metrics(
    result: BaselineBuildResult,
    problem: ProblemData,
) -> _MandatoryMetrics:
    """Calculate mandatory counts, unchanged desi share and fill ratios."""

    capacities = {
        vehicle.name: vehicle.capacity_desi
        for vehicle in problem.static.vehicle_types
    }
    mandatory = tuple(
        trip
        for trip in result.plan.trips
        if trip.ownership_type == "mandatory_rented"
    )
    fill_ratios = tuple(
        _trip_fill_ratio(trip, capacities[trip.vehicle_type])
        for trip in mandatory
    )
    carried_desi = math.fsum(
        assignment.desi
        for trip in mandatory
        for assignment in trip.legs[0].assignments
    )
    total_desi = math.fsum(demand.desi for demand in problem.demands)
    loaded = sum(bool(trip.legs[0].assignments) for trip in mandatory)
    average = math.fsum(fill_ratios) / len(fill_ratios) if fill_ratios else 0.0
    return _MandatoryMetrics(
        len(mandatory),
        loaded,
        len(mandatory) - loaded,
        sum(len(trip.legs[0].assignments) > 1 for trip in mandatory),
        carried_desi,
        carried_desi / total_desi if total_desi else 0.0,
        average,
        max(fill_ratios, default=0.0),
    )


def _trip_fill_ratio(trip: VehicleTrip, capacity_desi: float) -> float:
    """Return one direct trip's carried desi / vehicle capacity desi."""

    if not trip.legs or capacity_desi <= 0:
        return 0.0
    carried_desi = math.fsum(
        assignment.desi for assignment in trip.legs[0].assignments
    )
    return carried_desi / capacity_desi


def _savings_ratio(savings_tl: float, reference_tl: float) -> float:
    """Return TL savings divided by reference TL without a zero division."""

    return savings_tl / reference_tl if reference_tl else 0.0


def _metric_ratio(result: BaselineBuildResult, key: str) -> float:
    """Read one numeric validator ratio for CLI percentage formatting."""

    return float(result.validation_report.metrics.get(key, 0.0))


def _validity(result: BaselineBuildResult) -> str:
    """Render the independent validator state for one strategy."""

    return "GEÇERLİ" if result.validation_report.is_valid else "GEÇERSİZ"


def _completion(result: BaselineBuildResult) -> str:
    """Render one strategy's final delivery timestamp."""

    return (
        result.completion_at.isoformat(sep=" ")
        if result.completion_at is not None
        else "-"
    )


def main(argv: list[str] | None = None) -> int:
    """Run comparison CLI and convert expected failures to non-zero status.

    ``argv`` supports tests while normal execution uses process arguments.
    Builder error codes are printed for diagnosis; other loader/config errors
    remain concise.  Failure paths do not save any plan or source workbook.
    """

    args = build_parser().parse_args(argv)
    try:
        print(render_comparison(compare_strategies(args.manifest, args.forecast)))
    except BaselineBuildError as exc:
        codes = ",".join(exc.error_codes) or "baseline_build_failed"
        print(f"Plan karşılaştırması başarısız [{codes}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # CLI boundary keeps input failures concise.
        print(f"Plan karşılaştırması başarısız: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PlanComparison",
    "build_parser",
    "compare_strategies",
    "main",
    "render_comparison",
]
