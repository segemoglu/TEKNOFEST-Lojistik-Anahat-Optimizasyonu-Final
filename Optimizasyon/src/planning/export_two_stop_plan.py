"""Build validated 6D and export the official transport-plan workbook.

Run ``python -m src.planning.export_two_stop_plan`` to use the configured
development forecast and ``outputs/TAŞIMA PLANI.xlsx`` target.  Optional
manifest, forecast and output paths support surprise-data and CI runs without
changing protected sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import (
    ExcelProblemLoader,
    configured_project_path,
)
from src.io.transport_plan_workbook import (
    TransportPlanWorkbookExport,
    TransportPlanWorkbookExporter,
)

from .two_stop import GreedySameOriginTwoStopBuilder


def build_parser() -> argparse.ArgumentParser:
    """Create Unicode-safe manifest, forecast and generated-output arguments.

    Returns:
        Parser whose omitted paths select the central manifest defaults.
        Malformed arguments are rejected before any workbook is read or saved.
    """

    parser = argparse.ArgumentParser(
        description="Geçerli 6D planını resmî TAŞIMA PLANI.xlsx olarak üretir."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def export_two_stop_plan(
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> TransportPlanWorkbookExport:
    """Build 6D from one forecast and write a validator-gated workbook.

    Args:
        manifest_path: Optional central config path.
        forecast_path: Optional standard forecast workbook.  ``None`` uses
            ``paths.development_forecast`` below the inferred project root.
        output_path: Optional generated target.  ``None`` uses
            ``paths.transport_plan_output``; raw inputs are never overwritten.

    Returns:
        Export result containing the absolute path, typed rows and read-back
        audit totals.

    Connected modules and errors:
        ``ExcelProblemLoader`` builds immutable input data,
        ``GreedySameOriginTwoStopBuilder`` produces validated 6D, and
        ``TransportPlanWorkbookExporter`` revalidates, costs and round-trips
        the official workbook.  Their explicit loader, planning and exporter
        exceptions propagate and prevent a success message.
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
    result = GreedySameOriginTwoStopBuilder(problem, policy).build()
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
    ).write(result.plan, selected_output)


def render_export_summary(result: TransportPlanWorkbookExport) -> str:
    """Render path, leg/vehicle counts and reconciled TL audit totals."""

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
    """Build, export, reload and report 6D; return zero only on full success.

    Args:
        argv: Optional CLI arguments excluding the executable/module name.

    Returns:
        ``0`` after a valid exact-schema workbook survives semantic round-trip.
        Loader, planning, validation, costing or I/O failures propagate and
        produce a non-zero interpreter status.
    """

    args = build_parser().parse_args(argv)
    result = export_two_stop_plan(args.manifest, args.forecast, args.output)
    print(render_export_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
