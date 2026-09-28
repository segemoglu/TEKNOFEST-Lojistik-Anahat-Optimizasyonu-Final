"""CLI for the deterministic historical-mean development forecast.

Run with ``python -m src.forecasting.generate_development_forecast``.  The
command reads validated historical/static workbooks, writes only the configured
interim forecast path, reloads that workbook, and never starts optimization,
route generation, serialized models or a transport-plan exporter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.domain import DemandInstance
from src.io import ExcelProblemLoader, configured_project_path

from .providers import (
    ForecastBatch,
    HistoricalMeanForecastProvider,
    forecast_window_from_manifest,
)
from .workbook import write_development_forecast_workbook


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for selecting the central manifest only.

    The output path and forecast window remain centralized in the manifest;
    callers may choose an alternate manifest for isolated tests.  ``argparse``
    reports malformed arguments and no source file is changed.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Geçmiş taleplerden deterministik geliştirme tahmini üretir; "
            "optimizasyon çalıştırmaz."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Varsayılan: config/rules.yaml",
    )
    return parser


def generate_development_forecast(
    manifest_path: str | Path | None = None,
) -> tuple[ForecastBatch, Path]:
    """Generate, save and reload-check the development forecast workbook.

    Args:
        manifest_path: Optional JSON-compatible YAML manifest; ``None`` uses
            the repository default.

    Returns:
        ``ForecastBatch`` with standard demands/statistics and the absolute
        ``data/interim/development_forecast.xlsx`` path selected by config.

    Raises:
        Loader, config, forecast or workbook exceptions are propagated to the
        CLI boundary.  A domain-record mismatch after Excel reload raises
        ``RuntimeError``.  Raw files and pickle models are never written/read.
    """

    loader = ExcelProblemLoader(manifest_path)
    static_data = loader.load_static_problem_data()
    history = loader.load_demands(
        "historical_demand",
        static_data=static_data,
    )
    provider = HistoricalMeanForecastProvider(
        history,
        static_data,
        forecast_window_from_manifest(loader.manifest),
    )
    batch = provider.generate_with_statistics()
    output_path = configured_project_path(
        loader.manifest,
        "development_forecast",
        project_root=loader.project_root,
    )
    write_development_forecast_workbook(
        batch.demands,
        output_path,
        loader.manifest,
    )
    reloaded = loader.load_demands_from_workbook(
        output_path,
        static_data=static_data,
    )
    _require_identical_round_trip(batch.demands, reloaded)
    return batch, output_path


def render_summary(batch: ForecastBatch, output_path: Path) -> str:
    """Render record, route, full-precision desi, split and fallback counts.

    The CLI prints this human-readable text after a successful Excel round
    trip.  It consumes ``ForecastBatch`` diagnostics and performs no I/O or
    recalculation; invalid batches have already been rejected by their models.
    """

    statistics = batch.statistics
    methods = statistics.method_counts
    lines = [
        "Geliştirme tahmini üretildi; yarışma tahmini değildir.",
        f"Çıktı: {output_path}",
        f"Kayıt sayısı: {statistics.record_count}",
        f"Aktif O-D hattı: {statistics.active_route_count}",
        f"Toplam tahmini desi: {statistics.total_forecast_desi:.10f}",
        f"split_required sayısı: {statistics.split_required_count}",
        (
            "Yöntem kullanımı: "
            f"primary={methods['primary']}, "
            f"od_time={methods['od_time']}, "
            f"od_overall={methods['od_overall']}, "
            f"zero={methods['zero']}"
        ),
        "Excel round-trip: başarılı",
        "Optimizasyon çalıştırılmadı.",
    ]
    return "\n".join(lines)


def _require_identical_round_trip(
    expected: tuple[DemandInstance, ...],
    actual: tuple[DemandInstance, ...],
) -> None:
    """Raise with the first differing index when Excel changes domain records."""

    if expected == actual:
        return
    mismatch_index = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(expected, actual, strict=False)
            )
            if left != right
        ),
        min(len(expected), len(actual)),
    )
    raise RuntimeError(
        "Forecast Excel round-trip DemandInstance kayıtlarını değiştirdi: "
        f"index={mismatch_index}, beklenen_adet={len(expected)}, "
        f"okunan_adet={len(actual)}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert generation failures to a non-zero exit code.

    ``argv`` supports unit tests and normal execution uses process arguments.
    The function prints a complete summary on success or one concise error to
    stderr on failure; underlying modules keep raw workbooks read-only.
    """

    args = build_parser().parse_args(argv)
    try:
        batch, output_path = generate_development_forecast(args.manifest)
        print(render_summary(batch, output_path))
    except Exception as exc:  # The CLI boundary preserves a concise terminal error.
        print(f"Geliştirme tahmini üretilemedi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
