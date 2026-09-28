"""CLI for building and validating the non-optimized direct baseline plan.

Run ``python -m src.planning.generate_baseline_plan`` to load the configured
development forecast, or pass ``--forecast`` for another standard forecast
workbook.  The command prints diagnostics only: it never writes the official
``TAŞIMA PLANI.xlsx`` template or starts an optimization algorithm.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path

from .baseline import (
    BaselineBuildError,
    BaselineBuildResult,
    ConservativeDirectBaselineBuilder,
)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for manifest and forecast workbook selection.

    The forecast defaults to ``paths.development_forecast`` in the central
    manifest.  ``pathlib.Path`` preserves Windows and Unicode path handling;
    argument errors are handled by ``argparse`` before any workbook is read.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Bütün pozitif talepleri doğrudan taşıyan, optimize edilmemiş "
            "başlangıç planını kurar ve validator ile doğrular."
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


def generate_baseline_plan(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
) -> tuple[BaselineBuildResult, ProblemData, Path]:
    """Load a standard forecast and build the validated direct baseline.

    Args:
        manifest_path: Optional central JSON-compatible YAML config path.
        forecast_path: Optional standard forecast workbook; ``None`` resolves
            ``data/interim/development_forecast.xlsx`` through the manifest.

    Returns:
        Baseline result, its source-independent ``ProblemData`` and the
        absolute forecast path used.  The builder receives only ``ProblemData``
        and ``EvaluationPolicy`` and therefore has no Excel/provider coupling.

    Raises:
        Loader/config exceptions: For unreadable or invalid input workbooks.
        BaselineBuildError: For construction, costing or validator failures.
        No input or output workbook is modified.
    """

    loader = ExcelProblemLoader(manifest_path)
    static = loader.load_static_problem_data()
    selected_path = (
        Path(forecast_path).resolve()
        if forecast_path is not None
        else configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=loader.project_root,
        )
    )
    demands = loader.load_demands_from_workbook(
        selected_path,
        static_data=static,
    )
    problem = ProblemData(static, demands)
    policy = EvaluationPolicy.from_manifest(loader.manifest)
    result = ConservativeDirectBaselineBuilder(problem, policy).build()
    return result, problem, selected_path


def render_summary(
    result: BaselineBuildResult,
    problem: ProblemData,
    forecast_path: Path,
) -> str:
    """Render feasibility, desi, vehicle, capacity, TL and timing metrics.

    The function performs no I/O or recalculation.  Values come from the
    independent validation report, immutable plan and reported ``CostBreakdown``;
    costs are TL, quantities are desi and duration is seconds.
    """

    report = result.validation_report
    cost = result.plan.reported_cost
    spot_trips = [
        trip for trip in result.plan.trips if trip.ownership_type == "spot"
    ]
    distribution = Counter(trip.vehicle_type for trip in spot_trips)
    spot_truck_count = sum(
        trip.vehicle_type == "Tır" for trip in spot_trips
    )
    lines = [
        "Konservatif doğrudan başlangıç planı üretildi; optimize edilmemiştir.",
        f"Forecast: {forecast_path}",
        f"Validator: {'GEÇERLİ' if report.is_valid else 'GEÇERSİZ'}",
        f"Toplam talep / DemandPart: {len(problem.demands)} / {result.demand_part_count}",
        f"Teslim edilen toplam desi: {report.metrics.get('delivered_desi', 0.0):.10f}",
        f"Spot araç: {result.spot_vehicle_count} ({_format_distribution(distribution)})",
        f"Zorunlu kiralık araç: {result.mandatory_vehicle_count}",
        f"Spot Tır: {spot_truck_count}",
        f"Son teslim zamanı: {result.completion_at.isoformat(sep=' ') if result.completion_at else '-'}",
        f"Toplam maliyet (TL): {result.total_cost_tl:.2f}",
        f"SLA cezası (TL): {cost.sla_penalty_tl if cost else 0.0:.2f}",
        "En yüksek günlük elleçleme doluluk oranı: "
        + _format_ratio(report.metrics.get("peak_handling_capacity_ratio", 0.0)),
        "En yüksek günlük Tır kota oranı: "
        + _format_ratio(report.metrics.get("peak_truck_quota_ratio", 0.0)),
        f"Çalışma süresi (s): {result.build_seconds:.3f}",
        "Resmi TAŞIMA PLANI.xlsx üretilmedi.",
    ]
    return "\n".join(lines)


def _format_distribution(distribution: Counter[str]) -> str:
    """Render deterministic spot vehicle-type counts for the CLI."""

    if not distribution:
        return "yok"
    return ", ".join(
        f"{vehicle_type}={count}"
        for vehicle_type, count in sorted(distribution.items())
    )


def _format_ratio(value: object) -> str:
    """Render a numeric capacity ratio as percent without hiding infinity."""

    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(ratio):
        return "sonsuz"
    return f"%{ratio * 100.0:.2f}"


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return non-zero with structured build codes on failure.

    ``argv`` supports unit tests; normal execution uses process arguments.
    Expected baseline failures show their machine-readable codes and concise
    context on stderr.  No exception path writes a transport-plan workbook.
    """

    args = build_parser().parse_args(argv)
    try:
        result, problem, forecast_path = generate_baseline_plan(
            args.manifest,
            args.forecast,
        )
        print(render_summary(result, problem, forecast_path))
    except BaselineBuildError as exc:
        codes = ",".join(exc.error_codes) or "baseline_build_failed"
        print(f"Başlangıç planı üretilemedi [{codes}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # CLI boundary keeps loader failures concise.
        print(f"Başlangıç planı üretilemedi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
