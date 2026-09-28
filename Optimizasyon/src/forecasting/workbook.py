"""Excel serialization for development forecast ``DemandInstance`` records.

The official demand-template headers come only from ``config/rules.yaml``.
This module writes a new interim workbook and never opens raw sources for save,
so competition inputs remain unchanged.  Reload validation is delegated to
``ExcelProblemLoader.load_demands_from_workbook``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.domain import DemandInstance
from src.io.manifest import ManifestError, file_spec


FORECAST_SCHEMA_NAME = "forecast_demand_template"
FORECAST_FIELDS = (
    "demand_id",
    "demand_date",
    "time_slot",
    "origin",
    "destination",
    "desi",
)
HEADER_FILL_COLOR = "1F4E78"
HEADER_FONT_COLOR = "FFFFFF"
COLUMN_WIDTHS = (14, 13, 24, 28, 28, 22)
DATE_NUMBER_FORMAT = "yyyy-mm-dd"
TIME_NUMBER_FORMAT = "hh:mm"
DESI_NUMBER_FORMAT = "0.##########"


def write_development_forecast_workbook(
    demands: Sequence[DemandInstance],
    output_path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write standard demand records with official forecast-template headers.

    Args:
        demands: Ordered ``DemandInstance`` collection from any provider.
        output_path: New interim ``.xlsx`` target; parent folders are created.
        manifest: Central manifest supplying sheet and exact Turkish headers.

    Returns:
        Absolute saved workbook path.  The CLI reloads it through
        ``ExcelProblemLoader`` to verify domain-level round-trip equality.

    Raises:
        ManifestError: If the template schema omits a required canonical field.
        OSError or openpyxl exceptions: If the interim file cannot be saved.
        No source workbook is modified, and forecast desi floats are not
        rounded before storage.
    """

    spec = file_spec(manifest, FORECAST_SCHEMA_NAME)
    column_mapping = spec["columns"]
    missing = [field for field in FORECAST_FIELDS if field not in column_mapping]
    if missing:
        raise ManifestError(
            "Forecast şablonu zorunlu alanları eksik: " + ", ".join(missing)
        )
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = str(spec["sheet"])
    headers = [str(column_mapping[field]) for field in FORECAST_FIELDS]
    worksheet.append(headers)
    _style_header(worksheet, len(headers))
    for row_index, demand in enumerate(demands, start=2):
        worksheet.append(
            [
                demand.demand_id,
                demand.demand_date,
                demand.time_slot,
                demand.origin,
                demand.destination,
                None,
            ]
        )
        desi_cell = worksheet.cell(row_index, 6)
        # The shortest round-trip decimal is stored with numeric cell type.
        # Passing a float directly makes openpyxl emit only 16 significant
        # digits and can change a DemandInstance by one binary-float ULP.
        desi_cell.value = repr(float(demand.desi))
        desi_cell.data_type = "n"
    _format_data_sheet(worksheet, len(demands), len(headers))
    try:
        workbook.save(path)
    finally:
        workbook.close()
    return path


def _style_header(worksheet: Any, column_count: int) -> None:
    """Apply a compact, readable style to the official six-column header."""

    fill = PatternFill("solid", fgColor=HEADER_FILL_COLOR)
    font = Font(color=HEADER_FONT_COLOR, bold=True)
    for cell in worksheet[1][:column_count]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    worksheet.row_dimensions[1].height = 24


def _format_data_sheet(worksheet: Any, record_count: int, column_count: int) -> None:
    """Set typed number formats, filters, widths and a frozen header row."""

    worksheet.freeze_panes = "A2"
    final_column = get_column_letter(column_count)
    worksheet.auto_filter.ref = f"A1:{final_column}{record_count + 1}"
    worksheet.sheet_view.showGridLines = False
    for index, width in enumerate(COLUMN_WIDTHS, start=1):
        worksheet.column_dimensions[worksheet.cell(1, index).column_letter].width = width
    for row_index in range(2, record_count + 2):
        worksheet.cell(row_index, 2).number_format = DATE_NUMBER_FORMAT
        worksheet.cell(row_index, 3).number_format = TIME_NUMBER_FORMAT
        worksheet.cell(row_index, 6).number_format = DESI_NUMBER_FORMAT
        worksheet.cell(row_index, 2).alignment = Alignment(horizontal="center")
        worksheet.cell(row_index, 3).alignment = Alignment(horizontal="center")
        worksheet.cell(row_index, 6).alignment = Alignment(horizontal="right")
    if column_count != len(COLUMN_WIDTHS):
        raise ManifestError("Forecast sütun genişliği sözleşmesi şemayla uyuşmuyor.")


__all__ = ["write_development_forecast_workbook"]
