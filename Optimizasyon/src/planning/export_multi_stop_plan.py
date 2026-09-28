"""Build validated 6E and export the official transport-plan workbook.

Run ``python -m src.planning.export_multi_stop_plan`` to build a 6D reference,
require strict three/four-stop improvement and write the configured official
workbook.  The separate 6D exporter remains available as a safe fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.io.transport_plan_workbook import (
    TransportPlanWorkbookExport,
    TransportPlanWorkbookExporter,
)

from .multi_stop import GreedySameOriginMultiStopBuilder
from .two_stop import GreedySameOriginTwoStopBuilder


def build_parser() -> argparse.ArgumentParser:
    """Create Unicode-safe manifest, forecast and generated-output arguments."""

    parser = argparse.ArgumentParser(
        description="Geçerli 6E planını resmî TAŞIMA PLANI.xlsx olarak üretir."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def export_multi_stop_plan(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> TransportPlanWorkbookExport:
    """Build strict-improvement 6E and write a semantic-round-trip workbook.

    Args:
        manifest_path: Optional central config path.
        forecast_path: Optional standard forecast workbook; ``None`` selects
            ``paths.development_forecast``.
        output_path: Optional generated target; ``None`` selects
            ``paths.transport_plan_output`` inside the repository.

    Returns:
        Saved path, reloaded typed rows and compact round-trip audit metrics.

    Connected modules and errors:
        One ``ExcelProblemLoader`` creates the problem, 6D is built once as the
        explicit fallback/reference, 6E must beat it through
        ``GreedySameOriginMultiStopBuilder.build_from_two_stop``, and the
        generic workbook exporter revalidates/costs/reloads the result.  Any
        failure propagates and prevents a misleading success message.
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
    selected_output = (
        Path(output_path).resolve()
        if output_path is not None
        else configured_project_path(
            loader.manifest,
            "transport_plan_output",
            project_root=loader.project_root,
        )
    )
    return TransportPlanWorkbookExporter(
        problem,
        policy,
        loader.manifest,
    ).write(multi_stop.plan, selected_output)


def render_export_summary(result: TransportPlanWorkbookExport) -> str:
    """Render path, leg/vehicle counts and reconciled 6E TL audit totals."""

    return "\n".join(
        (
            f"Taşıma planı: {result.path}",
            f"Satır / fiziksel araç: {result.audit.row_count} / "
            f"{result.audit.vehicle_count}",
            f"Toplam maliyet: {result.audit.total_cost_tl:.2f} TL",
            f"SLA cezası: {result.audit.sla_penalty_tl:.2f} TL",
            "Round-trip doğrulama: GEÇTİ",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Build, export, reload and report 6E; return zero only on full success."""

    args = build_parser().parse_args(argv)
    result = export_multi_stop_plan(args.manifest, args.forecast, args.output)
    print(render_export_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
