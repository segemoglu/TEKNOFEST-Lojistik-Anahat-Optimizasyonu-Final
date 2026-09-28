"""Pick the cheapest valid per-seed plan and publish it as the deliverable.

The delivery run starts one process per random seed because the search result
varies far more with the seed than with most code changes, so each process writes
its own workbook.  This CLI compares those workbooks, checks each one is complete
and well formed, and copies the cheapest to the configured output path.

Run with ``python -m src.planning.select_best_plan``.

A workbook is only eligible if it reads back cleanly, carries the official column
count, covers every forecast demand and reports a positive total cost.  Ties keep
the lowest seed number so a rerun over the same candidates always publishes the
same plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

from src.io import ExcelProblemLoader, configured_project_path


# Written by the delivery run as ``outputs/kosu_seed<N>.xlsx``.
CANDIDATE_PATTERN = "kosu_seed*.xlsx"
SEED_IN_NAME = re.compile(r"kosu_seed(\d+)\.xlsx$", re.IGNORECASE)
OFFICIAL_COLUMN_COUNT = 16

# Column positions in the official transport-plan layout.
DEMAND_ID_COLUMN = 9
COST_COLUMN = 15


@dataclass(frozen=True, slots=True)
class Candidate:
    """One per-seed plan workbook with the numbers needed to rank it."""

    seed: int
    path: Path
    total_cost_tl: float
    row_count: int
    vehicle_count: int
    covered_demands: int


class PlanSelectionError(RuntimeError):
    """Explain why no candidate workbook could be published."""


def _root_demand_ids(series: pd.Series) -> set[str]:
    """Strip the ``-1``/``-2`` part suffixes back to original demand IDs."""

    text = series.astype(str)
    return set(text.str.replace(r"-\d+$", "", regex=True)) - {"0"}


def _inspect(path: Path, expected_demands: set[str]) -> Candidate:
    """Read one candidate workbook and reject it if anything is missing.

    Raises:
        PlanSelectionError: If the workbook cannot be read, has the wrong column
            count, leaves a forecast demand unplanned or reports a non-positive
            total cost.  Rejecting here is deliberate: publishing a cheap but
            incomplete plan would be worse than publishing a dearer complete one.
    """

    match = SEED_IN_NAME.search(path.name)
    if match is None:
        raise PlanSelectionError(f"Tohum numarası okunamadı: {path.name}")
    try:
        frame = pd.read_excel(path)
    except Exception as exc:
        raise PlanSelectionError(f"{path.name} okunamadı: {exc}") from exc
    if len(frame.columns) != OFFICIAL_COLUMN_COUNT:
        raise PlanSelectionError(
            f"{path.name}: sütun sayısı {len(frame.columns)}, "
            f"beklenen {OFFICIAL_COLUMN_COUNT}"
        )
    columns = frame.columns
    covered = _root_demand_ids(frame[columns[DEMAND_ID_COLUMN]])
    missing = expected_demands - covered
    if missing:
        raise PlanSelectionError(
            f"{path.name}: {len(missing)} talep planda yok "
            f"(örnek: {sorted(missing)[:3]})"
        )
    total = float(frame[columns[COST_COLUMN]].sum())
    if not total > 0:
        raise PlanSelectionError(f"{path.name}: toplam maliyet {total}")
    return Candidate(
        seed=int(match.group(1)),
        path=path,
        total_cost_tl=total,
        row_count=len(frame),
        vehicle_count=frame[columns[0]].nunique(),
        covered_demands=len(covered),
    )


def select_best_plan(
    outputs_dir: str | Path | None = None,
    forecast_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Candidate, tuple[Candidate, ...], tuple[str, ...]]:
    """Publish the cheapest complete per-seed plan as the official output.

    Args:
        outputs_dir: Directory holding ``kosu_seed*.xlsx``; defaults to the
            configured output workbook's own directory.
        forecast_path: Demand workbook whose IDs every plan must cover.
        manifest_path: Alternative config manifest.

    Returns:
        The published candidate, every accepted candidate, and one rejection
        message per discarded workbook.

    Raises:
        PlanSelectionError: If no candidate workbook exists or all were rejected.
    """

    loader = ExcelProblemLoader(manifest_path)
    destination = configured_project_path(
        loader.manifest,
        "transport_plan_output",
        project_root=loader.project_root,
    )
    directory = Path(outputs_dir) if outputs_dir else destination.parent
    forecast = (
        Path(forecast_path)
        if forecast_path
        else loader.project_root / "data" / "interim" / "talep_tahmini_ml.xlsx"
    )
    static_data = loader.load_static_problem_data()
    demands = loader.load_demands_from_workbook(forecast, static_data=static_data)
    expected = {item.demand_id for item in demands}

    paths = sorted(directory.glob(CANDIDATE_PATTERN))
    if not paths:
        raise PlanSelectionError(
            f"{directory} içinde {CANDIDATE_PATTERN} bulunamadı; "
            "önce koşuları çalıştırın."
        )

    accepted: list[Candidate] = []
    rejected: list[str] = []
    for path in paths:
        try:
            accepted.append(_inspect(path, expected))
        except PlanSelectionError as exc:
            rejected.append(str(exc))
    if not accepted:
        raise PlanSelectionError(
            "Hiçbir aday plan kabul edilmedi:\n  " + "\n  ".join(rejected)
        )

    # Cheapest wins; the lowest seed breaks a tie so the choice is reproducible.
    best = min(accepted, key=lambda item: (item.total_cost_tl, item.seed))
    shutil.copyfile(best.path, destination)
    return best, tuple(sorted(accepted, key=lambda item: item.total_cost_tl)), tuple(rejected)


def render_summary(
    best: Candidate,
    accepted: tuple[Candidate, ...],
    rejected: tuple[str, ...],
) -> str:
    """Render the seed comparison an operator needs to trust the choice."""

    lines = ["=== TOHUM KARŞILAŞTIRMASI ==="]
    for item in accepted:
        mark = "  <-- SEÇİLDİ" if item is best else ""
        lines.append(
            f"  tohum {item.seed:>4}: {item.total_cost_tl:>16,.2f} TL   "
            f"{item.row_count:>6} satır   {item.vehicle_count:>4} araç{mark}"
        )
    if len(accepted) > 1:
        spread = accepted[-1].total_cost_tl - accepted[0].total_cost_tl
        lines.append(f"  en iyi / en kötü fark: {spread:,.2f} TL")
    if rejected:
        lines.append("Reddedilen adaylar:")
        lines.extend(f"  {message}" for message in rejected)
    lines.append("")
    lines.append(f"Yayınlanan plan  : {best.total_cost_tl:,.2f} TL (tohum {best.seed})")
    lines.append(f"Kapsanan talep   : {best.covered_demands}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the candidate directory and forecast."""

    parser = argparse.ArgumentParser(
        description=(
            "Tohum başına üretilmiş planlardan en ucuz geçerli olanı "
            "resmî çıktı dosyasına kopyalar."
        )
    )
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and turn a selection failure into a non-zero exit code."""

    args = build_parser().parse_args(argv)
    try:
        best, accepted, rejected = select_best_plan(
            args.outputs_dir,
            args.forecast,
            args.manifest,
        )
        print(render_summary(best, accepted, rejected))
    except Exception as exc:  # CLI boundary keeps one concise terminal error.
        print(f"Plan seçilemedi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
