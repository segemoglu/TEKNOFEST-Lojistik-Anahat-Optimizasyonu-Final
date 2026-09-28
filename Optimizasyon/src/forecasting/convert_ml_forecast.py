"""Convert the team's ML shift forecasts into the official demand workbook.

The forecasting models emit one workbook per shift, keyed by an integer
``rota_id`` rather than by centre names, and without demand IDs.  This CLI joins
those two workbooks to the ``rota_id`` mapping, assigns competition-format demand
IDs and writes the manifest's official six-column template, so the planner can
consume it through ``--forecast`` exactly like the development forecast.

Run with ``python -m src.forecasting.convert_ml_forecast``.

Every check that can reject a bad input is applied before anything is written:
unknown centres, missing arcs, lanes absent from history, duplicated
lane/date/shift rows, negative or non-finite desi, a wrong date window and a
shift column that does not hold the expected time.  The written workbook is then
reloaded through ``ExcelProblemLoader`` and compared record by record.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
import math
from pathlib import Path
import sys
import unicodedata

import pandas as pd

from src.domain import DemandInstance, StaticProblemData
from src.io import ExcelProblemLoader, configured_project_path

from .workbook import write_development_forecast_workbook


# The model workbooks use their own column names; these are the ones this
# converter depends on and will refuse to run without.
ROTA_COLUMN = "rota_id"
DATE_COLUMN = "tarih"
SLOT_COLUMN = "talep_tamamlanma_saati"
DESI_COLUMN = "Tahmin_Edilen_Desi"
MAPPING_COLUMNS = ("rota_id", "cikis", "varis")

# Competition demand-ID format: D00001, D00002, ...
DEMAND_ID_TEMPLATE = "D{:05d}"


class ForecastConversionError(RuntimeError):
    """Explain why the ML workbooks cannot become an official forecast."""


def _normalise(value: object) -> str:
    """Compare Turkish centre names without tripping over Unicode forms.

    The model workbooks, the mapping workbook and the competition master data
    are produced by different tools; ``İstanbul`` can arrive composed or
    decomposed and would otherwise fail an equality check that looks correct.
    """

    return unicodedata.normalize("NFC", str(value)).strip()


def _read_mapping(path: Path) -> dict[int, tuple[str, str]]:
    """Load and sanity-check the ``rota_id`` to origin/destination mapping."""

    frame = pd.read_excel(path)
    missing = [name for name in MAPPING_COLUMNS if name not in frame.columns]
    if missing:
        raise ForecastConversionError(
            f"Eşleme dosyasında eksik sütun: {', '.join(missing)}"
        )
    mapping: dict[int, tuple[str, str]] = {}
    for row in frame.itertuples(index=False):
        rota_id = int(getattr(row, ROTA_COLUMN))
        if rota_id in mapping:
            raise ForecastConversionError(f"Eşlemede tekrar eden rota_id: {rota_id}")
        mapping[rota_id] = (_normalise(row.cikis), _normalise(row.varis))
    if not mapping:
        raise ForecastConversionError("Eşleme dosyası boş.")
    return mapping


def _read_shift(path: Path, expected_slot: time) -> pd.DataFrame:
    """Load one shift workbook and verify it holds only that shift."""

    frame = pd.read_excel(path)
    for name in (ROTA_COLUMN, DATE_COLUMN, SLOT_COLUMN, DESI_COLUMN):
        if name not in frame.columns:
            raise ForecastConversionError(
                f"{path.name}: beklenen sütun bulunamadı: {name!r}"
            )
    slots = {str(value)[:5] for value in frame[SLOT_COLUMN]}
    expected_text = expected_slot.strftime("%H:%M")
    if slots != {expected_text}:
        raise ForecastConversionError(
            f"{path.name}: vardiya sütunu {expected_text} dışında değer taşıyor: "
            f"{sorted(slots)}"
        )
    return frame


def _demand_rows(
    frame: pd.DataFrame,
    slot: time,
    mapping: dict[int, tuple[str, str]],
    path: Path,
) -> list[tuple[date, time, str, str, float]]:
    """Turn one shift workbook into normalised, validated tuples."""

    rows: list[tuple[date, time, str, str, float]] = []
    unknown: set[int] = set()
    for row in frame.itertuples(index=False):
        rota_id = int(getattr(row, ROTA_COLUMN))
        lane = mapping.get(rota_id)
        if lane is None:
            unknown.add(rota_id)
            continue
        desi = float(getattr(row, DESI_COLUMN))
        if not math.isfinite(desi) or desi < 0:
            raise ForecastConversionError(
                f"{path.name}: rota_id={rota_id} için geçersiz desi: {desi!r}"
            )
        rows.append(
            (
                pd.Timestamp(getattr(row, DATE_COLUMN)).date(),
                slot,
                lane[0],
                lane[1],
                desi,
            )
        )
    if unknown:
        raise ForecastConversionError(
            "Eşlemede bulunmayan rota_id değerleri: "
            + ", ".join(str(value) for value in sorted(unknown)[:10])
        )
    return rows


def _require_consistent(
    rows: list[tuple[date, time, str, str, float]],
    static_data: StaticProblemData,
    history_lanes: frozenset[tuple[str, str]],
    window: tuple[date, date],
) -> None:
    """Reject anything the planner or the competition rules would refuse.

    The competition forbids inventing a lane that never appeared in history and
    forbids dropping a lane that did, so both directions are checked here rather
    than discovered later as a silent cost difference.
    """

    centers = {_normalise(name) for name in static_data.center_names}
    arcs = {
        (_normalise(arc.origin), _normalise(arc.destination))
        for arc in static_data.travel_arcs
    }
    seen: set[tuple[date, time, str, str]] = set()
    lanes: set[tuple[str, str]] = set()
    for day, slot, origin, destination, _ in rows:
        if origin not in centers or destination not in centers:
            raise ForecastConversionError(
                f"Tanınmayan transfer merkezi: {origin!r} -> {destination!r}"
            )
        if (origin, destination) not in arcs:
            raise ForecastConversionError(
                f"Mesafe matrisinde bulunmayan hat: {origin!r} -> {destination!r}"
            )
        if not window[0] <= day <= window[1]:
            raise ForecastConversionError(
                f"Tahmin penceresi dışında tarih: {day} "
                f"(beklenen {window[0]} - {window[1]})"
            )
        key = (day, slot, origin, destination)
        if key in seen:
            raise ForecastConversionError(f"Tekrar eden tahmin satırı: {key}")
        seen.add(key)
        lanes.add((origin, destination))

    invented = sorted(lanes - history_lanes)
    if invented:
        raise ForecastConversionError(
            "Geçmiş veride bulunmayan hat için tahmin üretilmiş: "
            + ", ".join(f"{o}->{d}" for o, d in invented[:10])
        )
    dropped = sorted(history_lanes - lanes)
    if dropped:
        raise ForecastConversionError(
            "Geçmişte bulunan hat tahminde eksik: "
            + ", ".join(f"{o}->{d}" for o, d in dropped[:10])
        )


def convert_ml_forecast(
    morning_path: str | Path,
    evening_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[tuple[DemandInstance, ...], Path]:
    """Join the shift workbooks into one official-format forecast workbook.

    Args:
        morning_path: Model output holding only the 09:00 shift.
        evening_path: Model output holding only the 17:00 shift.
        mapping_path: Workbook with ``rota_id``, ``cikis`` and ``varis``.
        output_path: Destination ``.xlsx`` written with manifest headers.
        manifest_path: Alternative config manifest.

    Returns:
        The demand records written and the absolute output path.

    Raises:
        ForecastConversionError: For any structural or rule violation listed in
            the module docstring.
        RuntimeError: If reloading the written workbook does not reproduce the
            same domain records.
    """

    loader = ExcelProblemLoader(manifest_path)
    static_data = loader.load_static_problem_data()
    history = loader.load_demands("historical_demand", static_data=static_data)
    history_lanes = frozenset(
        (_normalise(item.origin), _normalise(item.destination)) for item in history
    )

    mapping = _read_mapping(Path(mapping_path))
    morning = _read_shift(Path(morning_path), time(9, 0))
    evening = _read_shift(Path(evening_path), time(17, 0))

    rows = _demand_rows(morning, time(9, 0), mapping, Path(morning_path))
    rows += _demand_rows(evening, time(17, 0), mapping, Path(evening_path))

    days = sorted({day for day, *_ in rows})
    _require_consistent(rows, static_data, history_lanes, (days[0], days[-1]))

    # Deterministic order: the demand IDs must be reproducible from the inputs
    # alone, otherwise a rerun renames every demand and the plan stops matching
    # the forecast the jury holds.
    rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    # ``split_required`` is derived, not supplied: both the loader
    # (``excel_loader``) and the development-forecast provider mark a demand that
    # exceeds the largest vehicle.  Setting it here keeps the round-trip check
    # meaningful — otherwise every oversized demand looks like a serialization
    # failure when the loader simply computed the flag.
    maximum_capacity = static_data.maximum_vehicle_capacity_desi
    demands = tuple(
        DemandInstance(
            demand_id=DEMAND_ID_TEMPLATE.format(index),
            demand_date=day,
            time_slot=slot,
            origin=origin,
            destination=destination,
            desi=desi,
            split_required=desi > maximum_capacity,
        )
        for index, (day, slot, origin, destination, desi) in enumerate(rows, start=1)
    )

    written = write_development_forecast_workbook(
        demands,
        output_path,
        loader.manifest,
    )
    reloaded = loader.load_demands_from_workbook(written, static_data=static_data)
    if reloaded != demands:
        raise RuntimeError(
            "Tahmin Excel round-trip DemandInstance kayıtlarını değiştirdi: "
            f"yazılan={len(demands)}, okunan={len(reloaded)}"
        )
    return demands, written


def render_summary(demands: tuple[DemandInstance, ...], output_path: Path) -> str:
    """Render the counts an operator needs before starting a planning run."""

    total = math.fsum(item.desi for item in demands)
    per_slot: dict[str, float] = {}
    per_day: dict[date, float] = {}
    for item in demands:
        key = item.time_slot.strftime("%H:%M")
        per_slot[key] = per_slot.get(key, 0.0) + item.desi
        per_day[item.demand_date] = per_day.get(item.demand_date, 0.0) + item.desi
    zero_count = sum(1 for item in demands if item.desi == 0)
    lines = [
        "ML talep tahmini resmî formata çevrildi.",
        f"Çıktı: {output_path}",
        f"Kayıt sayısı: {len(demands)}",
        f"Hat sayısı: {len({(i.origin, i.destination) for i in demands})}",
        f"Gün sayısı: {len(per_day)}",
        f"Toplam desi: {total:,.2f}",
        "Vardiya dağılımı: "
        + ", ".join(f"{k}={v:,.0f}" for k, v in sorted(per_slot.items())),
        f"Sıfır desili kayıt: {zero_count}",
        "Gün gün desi:",
    ]
    lines.extend(
        f"  {day}: {value:,.0f}" for day, value in sorted(per_day.items())
    )
    lines.append("Excel round-trip: başarılı")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the two shift workbooks and the mapping."""

    parser = argparse.ArgumentParser(
        description=(
            "ML vardiya tahminlerini resmî talep tahmini formatına çevirir; "
            "optimizasyon çalıştırmaz."
        )
    )
    parser.add_argument(
        "--morning",
        type=Path,
        default=Path("data/interim/ml_tahmin/SABAH.xlsx"),
        help="09:00 vardiyası tahmin dosyası",
    )
    parser.add_argument(
        "--evening",
        type=Path,
        default=Path("data/interim/ml_tahmin/AKSAM.xlsx"),
        help="17:00 vardiyası tahmin dosyası",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/interim/rota_id_eslesme.xlsx"),
        help="rota_id -> cikis/varis eşleme dosyası",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/ml_forecast.xlsx"),
        help="Resmî formatta yazılacak tahmin dosyası",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and turn any conversion failure into a non-zero exit code."""

    args = build_parser().parse_args(argv)
    try:
        demands, output_path = convert_ml_forecast(
            args.morning,
            args.evening,
            args.mapping,
            args.output,
            args.manifest,
        )
        print(render_summary(demands, output_path))
    except Exception as exc:  # CLI boundary keeps one concise terminal error.
        print(f"ML tahmini çevrilemedi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
