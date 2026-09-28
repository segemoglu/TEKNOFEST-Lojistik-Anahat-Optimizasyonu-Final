"""Export validated transport plans to the official 16-column workbook.

The exporter keeps protected source workbooks read-only and creates a new
``.xlsx`` file with manifest-backed Turkish headers.

One output row represents **one demand part on one physical vehicle leg**.  A
leg carrying three demand parts therefore produces three rows that repeat the
vehicle, route and timestamps.  The competition organisers require this layout
and require cost and handling duration to be allocated across those rows in
proportion to desi (their example: 4.000 desi gets 40%, 6.000 desi gets 60%).

Allocation rules applied here:

* ``Toplam maliyet``  -- vehicle TL allocated by desi share within the leg.
* ``SLA cezası``      -- not split; each part carries its own penalty only.
* handling durations  -- allocated by desi share among the parts actually
  loaded (respectively unloaded) on that leg.
* ``Yolculuk süresi`` -- repeated whole; the journey is shared, not divided.

An empty mandatory movement still produces exactly one row using the
configured zero-desi placeholder.  The exported row total reconciles exactly to
the authoritative vehicle-level :class:`~src.domain.CostBreakdown`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
import math
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

from src.domain import CostBreakdown, ProblemData, TransportPlan, VehicleLeg, VehicleTrip
from src.evaluation import CostCalculator, EvaluationPolicy, TransportPlanValidator

from .manifest import ManifestError, file_spec


TRANSPORT_PLAN_SCHEMA_NAME = "transport_plan_template"
TRANSPORT_PLAN_FIELDS = (
    "vehicle_id",
    "vehicle_ownership_type",
    "vehicle_type",
    "origin",
    "destination",
    "departure_date",
    "departure_time",
    "arrival_date",
    "arrival_time",
    "demand_id",
    "carried_desi",
    "travel_hours",
    "arrival_handling_hours",
    "departure_handling_hours",
    "sla_penalty_tl",
    "total_cost_tl",
)
DATE_NUMBER_FORMAT = "yyyy-mm-dd"
TIME_NUMBER_FORMAT = "hh:mm"
DESI_NUMBER_FORMAT = "0.000"
DURATION_NUMBER_FORMAT = "0.000000"
COST_NUMBER_FORMAT = "0.00"
SECONDS_PER_HOUR = 3600.0
EXCEL_TIME_TOLERANCE_SECONDS = 0.001
NUMERIC_ROUND_TRIP_TOLERANCE = 1e-9
ZERO_DESI_TOLERANCE = 1e-12
DEFAULT_COST_TOLERANCE_TL = 0.01
COLUMN_WIDTHS = (
    13.0, 13.0, 13.0, 22.0, 22.0, 13.0, 13.0, 13.0,
    13.0, 38.0, 13.0, 16.0, 21.0, 21.0, 13.0, 16.0,
)


class TransportPlanWorkbookError(RuntimeError):
    """Report an unsafe export contract, invalid plan or round-trip mismatch."""


@dataclass(frozen=True, slots=True)
class TransportPlanWorkbookRow:
    """Represent one demand part on one physical leg, as stored in the workbook.

    ``demand_part_id`` is the single atomic ID visible to the evaluator;
    ``None`` marks an empty mandatory movement and is written with the
    configured zero-desi placeholder.  Timestamps are truncated to whole
    minutes so the stored value matches the ``hh:mm`` cell display exactly.
    Duration fields are decimal hours and cost fields are Turkish lira (TL).
    """

    vehicle_id: str
    vehicle_ownership_label: str
    vehicle_type: str
    origin: str
    destination: str
    departure_at: datetime
    arrival_at: datetime
    demand_part_id: str | None
    carried_desi: float
    travel_hours: float
    arrival_handling_hours: float
    departure_handling_hours: float
    sla_penalty_tl: float
    total_cost_tl: float


@dataclass(frozen=True, slots=True)
class TransportPlanWorkbookAudit:
    """Summarize a successful write-and-read verification of one workbook."""

    row_count: int
    vehicle_count: int
    total_cost_tl: float
    sla_penalty_tl: float


@dataclass(frozen=True, slots=True)
class TransportPlanWorkbookExport:
    """Return the saved path, immutable rows and round-trip audit metrics."""

    path: Path
    rows: tuple[TransportPlanWorkbookRow, ...]
    audit: TransportPlanWorkbookAudit


class TransportPlanWorkbookExporter:
    """Validate, cost, serialize and reload one complete transport plan.

    Args:
        problem: Static network plus the exact forecast demands being exported.
        policy: The same timing, tolerance and costing policy used by planning.
        manifest: Central config supplying official headers and output policy.

    Connected modules and errors:
        :class:`CostCalculator` and :class:`TransportPlanValidator` are rerun
        before writing.  Invalid or unreported plans, malformed output config,
        workbook I/O failures and any round-trip difference raise
        :class:`TransportPlanWorkbookError`; protected input files are never
        opened for save.
    """

    def __init__(
        self,
        problem: ProblemData,
        policy: EvaluationPolicy,
        manifest: Mapping[str, Any],
    ) -> None:
        """Index immutable arcs and checked manifest-backed output settings."""

        self.problem = problem
        self.policy = policy
        self.manifest = manifest
        self._arcs = {
            (arc.origin, arc.destination): arc
            for arc in problem.static.travel_arcs
        }
        self._headers, self._empty_demand_id = _output_contract(manifest)

    def build_rows(
        self,
        plan: TransportPlan,
    ) -> tuple[TransportPlanWorkbookRow, ...]:
        """Convert a validator-approved plan into deterministic leg rows.

        Vehicle IDs are remapped in first-appearance order to official
        ``V0001``-style IDs.  Each vehicle cost line is allocated over its legs
        without a second hourly ``ceil``; final-delivery SLA TL is attached to
        the corresponding destination leg.  The row total must reconcile to
        the authoritative complete-plan total within the configured TL
        tolerance or export stops.
        """

        cost = self._validated_cost(plan)
        tolerance = _cost_tolerance(self.manifest)
        vehicle_costs = _allocate_vehicle_costs(
            plan,
            cost,
            self._arcs,
            tolerance,
        )
        # Row building attributes each penalty to its own demand part; this call
        # is retained because it also guards that the leg-level SLA total still
        # reconciles to the authoritative breakdown.
        _sla_costs_by_leg(plan, cost, tolerance)
        sla_by_part = {
            line.demand_part_id: line.penalty_tl
            for line in cost.sla_penalty_lines
        }
        external_ids = _external_vehicle_ids(plan.trips)
        rows = tuple(
            row
            for trip in plan.trips
            for leg in trip.legs
            for row in self._rows_for_leg(
                trip,
                leg,
                external_ids[trip.vehicle_id],
                vehicle_costs[leg.leg_id],
                sla_by_part,
            )
        )
        if not math.isclose(
            math.fsum(row.total_cost_tl for row in rows),
            cost.total_cost_tl,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise TransportPlanWorkbookError(
                "Excel satır maliyetleri plan toplam maliyetiyle uzlaşmıyor."
            )
        return rows

    def write(
        self,
        plan: TransportPlan,
        output_path: str | Path,
    ) -> TransportPlanWorkbookExport:
        """Write a new workbook and require semantic round-trip equality.

        Args:
            plan: Complete plan already carrying a reported cost breakdown.
            output_path: New or replaceable generated ``.xlsx`` target.  The
                parent directory is created; raw source workbooks are untouched.

        Returns:
            Saved absolute path, reloaded rows and compact audit metrics.

        Raises:
            TransportPlanWorkbookError: If validation, costing, serialization
                or reload differs.  Lower-level permission errors are wrapped
                with the target path for diagnosis.
        """

        rows = self.build_rows(plan)
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_workbook(
                path,
                str(file_spec(self.manifest, TRANSPORT_PLAN_SCHEMA_NAME)["sheet"]),
                self._headers,
                rows,
                self._empty_demand_id,
            )
            actual = read_transport_plan_workbook(path, self.manifest)
        except (OSError, ValueError, TypeError) as exc:
            raise TransportPlanWorkbookError(
                f"Taşıma planı Excel'i yazılamadı veya geri okunamadı: {path}: {exc}"
            ) from exc
        _require_equal_rows(rows, actual, _cost_tolerance(self.manifest))
        return TransportPlanWorkbookExport(path, actual, _audit(actual))

    def _validated_cost(self, plan: TransportPlan) -> CostBreakdown:
        """Revalidate the plan and return an independently recalculated cost."""

        if plan.reported_cost is None:
            raise TransportPlanWorkbookError(
                "Raporlanmış CostBreakdown bulunmayan plan Excel'e aktarılamaz."
            )
        report = TransportPlanValidator(self.problem, self.policy).validate(plan)
        if not report.is_valid:
            codes = ", ".join(issue.code for issue in report.errors[:8])
            raise TransportPlanWorkbookError(
                f"Geçersiz plan Excel'e aktarılamaz: {codes}"
            )
        calculated = CostCalculator(self.problem, self.policy).calculate(plan)
        if not math.isclose(
            calculated.total_cost_tl,
            plan.reported_cost.total_cost_tl,
            rel_tol=0.0,
            abs_tol=_cost_tolerance(self.manifest),
        ):
            raise TransportPlanWorkbookError(
                "Raporlanan ve yeniden hesaplanan plan maliyetleri farklı."
            )
        return calculated

    def _rows_for_leg(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        external_vehicle_id: str,
        vehicle_cost_tl: float,
        sla_by_part: Mapping[str, float],
    ) -> tuple[TransportPlanWorkbookRow, ...]:
        """Create one row per demand part carried on this physical leg.

        Args:
            trip: Owning trip, used for ownership label and handling lookup.
            leg: The physical movement being reported.
            external_vehicle_id: Official ``V0001``-style identifier.
            vehicle_cost_tl: This leg's already-allocated vehicle TL.
            sla_by_part: Penalty TL per demand part from the cost breakdown.

        Returns:
            One row per assignment, or exactly one placeholder row when an
            empty mandatory vehicle moves without cargo.

        Allocation:
            Vehicle TL is divided by desi share across every part on the leg.
            Handling durations are divided only among the parts actually loaded
            (respectively unloaded) here, so a part continuing from an earlier
            leg reports zero loading time.  Travel hours are repeated whole
            because the journey is shared rather than divided.  SLA penalty is
            never split; it belongs to the delivered part alone.
        """

        operations = {item.operation_id: item for item in trip.handling_operations}
        load = operations.get(leg.loading_operation_id or "")
        unload = operations.get(leg.unloading_operation_id or "")
        ownership_label = _ownership_label(trip.ownership_type, self.manifest)
        travel_hours = _hours(leg.departure_at, leg.arrival_at)
        departure_at = _truncate_to_minute(leg.departure_at)
        arrival_at = _truncate_to_minute(leg.arrival_at)

        if not leg.assignments:
            return (
                TransportPlanWorkbookRow(
                    external_vehicle_id, ownership_label, trip.vehicle_type,
                    leg.origin, leg.destination, departure_at, arrival_at,
                    None, 0.0, travel_hours,
                    _operation_hours(unload), _operation_hours(load),
                    0.0, vehicle_cost_tl,
                ),
            )

        keys = tuple(str(index) for index in range(len(leg.assignments)))
        cost_share = _proportional(
            vehicle_cost_tl,
            keys,
            [item.desi for item in leg.assignments],
        )
        loading_share = _handling_share(
            _operation_hours(load),
            keys,
            [
                item.desi if item.loaded_at_origin else 0.0
                for item in leg.assignments
            ],
        )
        unloading_share = _handling_share(
            _operation_hours(unload),
            keys,
            [
                item.desi if item.unloaded_at_destination else 0.0
                for item in leg.assignments
            ],
        )

        rows: list[TransportPlanWorkbookRow] = []
        for key, assignment in zip(keys, leg.assignments, strict=True):
            penalty = (
                sla_by_part.get(assignment.demand_part_id, 0.0)
                if assignment.is_final_delivery
                else 0.0
            )
            rows.append(
                TransportPlanWorkbookRow(
                    external_vehicle_id, ownership_label, trip.vehicle_type,
                    leg.origin, leg.destination, departure_at, arrival_at,
                    assignment.demand_part_id, assignment.desi, travel_hours,
                    unloading_share[key], loading_share[key],
                    penalty, cost_share[key] + penalty,
                )
            )
        return tuple(rows)


def read_transport_plan_workbook(
    workbook_path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[TransportPlanWorkbookRow, ...]:
    """Read a generated workbook back into typed semantic output rows.

    The exact configured sheet and 16-header order are required.  Date/time
    cells must be real Excel date/time values, numerics must remain numeric,
    and the empty mandatory placeholder becomes ``demand_part_id=None``.  This
    is a generated-output reader only; input-domain loading remains in
    :mod:`src.io.excel_loader`.
    """

    headers, empty_demand_id = _output_contract(manifest)
    spec = file_spec(manifest, TRANSPORT_PLAN_SCHEMA_NAME)
    workbook = load_workbook(Path(workbook_path), read_only=True, data_only=True)
    try:
        if str(spec["sheet"]) not in workbook.sheetnames:
            raise TransportPlanWorkbookError(
                f"Taşıma planı sayfası bulunamadı: {spec['sheet']!r}"
            )
        worksheet = workbook[str(spec["sheet"])]
        actual_headers = tuple(worksheet.cell(1, index).value for index in range(1, 17))
        if actual_headers != headers:
            raise TransportPlanWorkbookError("Taşıma planı başlıkları manifestle uyuşmuyor.")
        return tuple(
            _parse_row(values, empty_demand_id, row_index)
            for row_index, values in enumerate(
                worksheet.iter_rows(min_row=2, max_col=16, values_only=True),
                start=2,
            )
            if any(value is not None for value in values)
        )
    finally:
        workbook.close()


def _output_contract(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], object]:
    """Return exact headers and the empty mandatory placeholder demand ID."""

    spec = file_spec(manifest, TRANSPORT_PLAN_SCHEMA_NAME)
    columns = spec["columns"]
    missing = [field for field in TRANSPORT_PLAN_FIELDS if field not in columns]
    if missing:
        raise ManifestError(
            "Taşıma planı şeması zorunlu alanları eksik: " + ", ".join(missing)
        )
    policy = manifest.get("rules", {}).get("temporary_output_policy", {})
    multiple = policy.get("multiple_demand_ids_per_physical_vehicle_row", {})
    empty = policy.get("empty_mandatory_vehicle", {})
    # The organisers require one row per demand ID with desi-proportional cost
    # and handling allocation; joining IDs into one cell is no longer legal.
    if multiple.get("mode") != "explode":
        raise ManifestError(
            "Taşıma planı Talep ID politikası 'explode' olmalıdır; "
            f"bulunan={multiple.get('mode')!r}."
        )
    if "demand_id" not in empty:
        raise ManifestError("Boş zorunlu araç Talep ID yer tutucusu eksik.")
    headers = tuple(str(columns[field]) for field in TRANSPORT_PLAN_FIELDS)
    return headers, empty["demand_id"]


def _ownership_label(ownership_type: str, manifest: Mapping[str, Any]) -> str:
    """Map internal ownership to the configured official Turkish label."""

    labels = manifest.get("rules", {}).get("temporary_output_policy", {}).get(
        "vehicle_ownership_labels", {}
    )
    label = labels.get(ownership_type)
    if not isinstance(label, str) or not label:
        raise ManifestError(f"Araç sahiplik çıktı etiketi eksik: {ownership_type!r}")
    return label


def _external_vehicle_ids(trips: Sequence[VehicleTrip]) -> dict[str, str]:
    """Assign deterministic official ``V0001`` IDs to physical internal IDs."""

    ordered = dict.fromkeys(trip.vehicle_id for trip in trips)
    return {
        internal_id: f"V{index:04d}"
        for index, internal_id in enumerate(ordered, start=1)
    }


def _allocate_vehicle_costs(
    plan: TransportPlan,
    cost: CostBreakdown,
    arcs: Mapping[tuple[str, str], Any],
    tolerance_tl: float,
) -> dict[str, float]:
    """Allocate already-ceiled vehicle TL to legs without changing its total."""

    indexed = _indexed_legs(plan)
    allocated = {leg_id: 0.0 for leg_id in indexed}
    for line in cost.vehicle_cost_lines:
        leg_ids = tuple(line.leg_ids)
        if not leg_ids or any(leg_id not in indexed for leg_id in leg_ids):
            raise TransportPlanWorkbookError(
                f"Maliyet satırı bilinmeyen veya boş leg listesi taşıyor: {line.cost_unit_id}"
            )
        active_weights = [
            _leg_activity_hours(*indexed[leg_id]) for leg_id in leg_ids
        ]
        distance_weights = [
            float(arcs[(indexed[leg_id][1].origin, indexed[leg_id][1].destination)].distance_km)
            for leg_id in leg_ids
        ]
        hourly = _proportional(line.hourly_cost_tl, leg_ids, active_weights)
        distance = _proportional(line.distance_cost_tl, leg_ids, distance_weights)
        for leg_id in leg_ids:
            allocated[leg_id] += hourly[leg_id] + distance[leg_id]
    if not math.isclose(
        math.fsum(allocated.values()),
        cost.vehicle_cost_tl,
        rel_tol=0.0,
        abs_tol=tolerance_tl,
    ):
        raise TransportPlanWorkbookError("Bacaklara ayrılan araç TL toplamı değişti.")
    return allocated


def _indexed_legs(plan: TransportPlan) -> dict[str, tuple[VehicleTrip, VehicleLeg]]:
    """Index unique leg IDs together with their owning trips."""

    indexed: dict[str, tuple[VehicleTrip, VehicleLeg]] = {}
    for trip in plan.trips:
        for leg in trip.legs:
            if leg.leg_id in indexed:
                raise TransportPlanWorkbookError(f"Tekrarlı leg_id: {leg.leg_id!r}")
            indexed[leg.leg_id] = (trip, leg)
    return indexed


def _leg_activity_hours(trip: VehicleTrip, leg: VehicleLeg) -> float:
    """Measure linked handling plus travel span used as hourly allocation weight."""

    operations = {item.operation_id: item for item in trip.handling_operations}
    start_at = leg.departure_at
    end_at = leg.arrival_at
    load = operations.get(leg.loading_operation_id or "")
    unload = operations.get(leg.unloading_operation_id or "")
    if load is not None:
        start_at = min(start_at, load.start_at)
    if unload is not None:
        end_at = max(end_at, unload.end_at)
    return _hours(start_at, end_at)


def _handling_share(
    total_hours: float,
    keys: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    """Allocate one handling duration only to the parts that operation covers.

    Unlike :func:`_proportional`, an all-zero weight vector returns zeros
    instead of an equal split: a part that was not loaded (or unloaded) on this
    leg must report no handling time for it.
    """

    if math.fsum(weights) <= 0.0:
        return {key: 0.0 for key in keys}
    return _proportional(total_hours, keys, weights)


def _truncate_to_minute(value: datetime) -> datetime:
    """Drop sub-minute precision from one exported timestamp.

    The competition accepts ``HH:MM`` resolution (Q&A 11.4), but plan
    timestamps carry fractional seconds because handling duration is derived
    from fractional desi.  Storing the truncated value keeps the cell and its
    ``hh:mm`` display identical: a reader opening the workbook visually and a
    reader parsing it programmatically both see the same minute.

    This affects the exported cell only.  Vehicle cost and SLA penalty are
    computed from the full-precision plan timestamps and are unchanged; the
    ceiling rules the competition applies to *durations* (billed hours, delay
    hours) are enforced in :mod:`src.evaluation.costs`, not here.
    """

    return value.replace(second=0, microsecond=0)


def _proportional(
    amount: float,
    keys: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    """Allocate one TL component by non-negative weights with exact remainder."""

    if len(keys) != len(weights) or not keys:
        raise TransportPlanWorkbookError("Maliyet dağıtım anahtarı/ağırlığı geçersiz.")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise TransportPlanWorkbookError("Maliyet dağıtım ağırlığı sonlu ve negatif olmamalı.")
    total_weight = math.fsum(weights)
    effective = tuple(weights) if total_weight > 0.0 else tuple(1.0 for _ in keys)
    denominator = math.fsum(effective)
    result: dict[str, float] = {}
    used = 0.0
    for key, weight in zip(keys[:-1], effective[:-1], strict=True):
        value = amount * weight / denominator
        result[key] = value
        used += value
    result[keys[-1]] = amount - used
    return result


def _sla_costs_by_leg(
    plan: TransportPlan,
    cost: CostBreakdown,
    tolerance_tl: float,
) -> dict[str, float]:
    """Attach each atomic SLA penalty to its single final-delivery leg."""

    by_part = {line.demand_part_id: line.penalty_tl for line in cost.sla_penalty_lines}
    result: dict[str, float] = {}
    for trip in plan.trips:
        for leg in trip.legs:
            result[leg.leg_id] = math.fsum(
                by_part.get(item.demand_part_id, 0.0)
                for item in leg.assignments
                if item.is_final_delivery
            )
    if not math.isclose(
        math.fsum(result.values()),
        cost.sla_penalty_tl,
        rel_tol=0.0,
        abs_tol=tolerance_tl,
    ):
        raise TransportPlanWorkbookError("Bacaklara bağlanan SLA TL toplamı değişti.")
    return result


def _write_workbook(
    path: Path,
    sheet_name: str,
    headers: tuple[str, ...],
    rows: Sequence[TransportPlanWorkbookRow],
    empty_demand_id: object,
) -> None:
    """Create a template-compatible workbook without resaving a raw source."""

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(headers)
    _style_header_and_columns(worksheet)
    for row in rows:
        worksheet.append(_excel_values(row, empty_demand_id))
    _style_data_rows(worksheet, len(rows))
    try:
        workbook.save(path)
    finally:
        workbook.close()


def _style_header_and_columns(worksheet: Any) -> None:
    """Match the supplied template's Arial header and practical widths."""

    for cell in worksheet[1]:
        cell.font = Font(name="Arial", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for index, width in enumerate(COLUMN_WIDTHS, start=1):
        worksheet.column_dimensions[worksheet.cell(1, index).column_letter].width = width
    worksheet.freeze_panes = "A2"


def _style_data_rows(worksheet: Any, row_count: int) -> None:
    """Apply semantic date, time, desi, duration and TL cell formats."""

    worksheet.auto_filter.ref = f"A1:P{row_count + 1}"
    for row_index in range(2, row_count + 2):
        for column in range(1, 17):
            worksheet.cell(row_index, column).font = Font(name="Arial", size=10)
        for column in (6, 8):
            worksheet.cell(row_index, column).number_format = DATE_NUMBER_FORMAT
        for column in (7, 9):
            worksheet.cell(row_index, column).number_format = TIME_NUMBER_FORMAT
        worksheet.cell(row_index, 11).number_format = DESI_NUMBER_FORMAT
        for column in (12, 13, 14):
            worksheet.cell(row_index, column).number_format = DURATION_NUMBER_FORMAT
        for column in (15, 16):
            worksheet.cell(row_index, column).number_format = COST_NUMBER_FORMAT


def _excel_values(
    row: TransportPlanWorkbookRow,
    empty_demand_id: object,
) -> list[object]:
    """Return typed Excel values in exact official column order."""

    demand_value: object = (
        row.demand_part_id if row.demand_part_id is not None else empty_demand_id
    )
    return [
        row.vehicle_id, row.vehicle_ownership_label, row.vehicle_type,
        row.origin, row.destination, row.departure_at.date(),
        row.departure_at.time(), row.arrival_at.date(), row.arrival_at.time(),
        demand_value, row.carried_desi, row.travel_hours,
        row.arrival_handling_hours, row.departure_handling_hours,
        row.sla_penalty_tl, row.total_cost_tl,
    ]


def _parse_row(
    values: Sequence[object],
    empty_demand_id: object,
    row_index: int,
) -> TransportPlanWorkbookRow:
    """Parse one non-empty 16-cell workbook row with typed error context."""

    if len(values) != 16:
        raise TransportPlanWorkbookError(f"Satır {row_index}: 16 sütun bekleniyor.")
    demand_part_id = _parse_demand_id(
        values[9],
        values[10],
        empty_demand_id,
        row_index,
    )
    return TransportPlanWorkbookRow(
        _text(values[0], row_index, "Araç ID"),
        _text(values[1], row_index, "Araç Tipi"),
        _text(values[2], row_index, "Araç türü"),
        _text(values[3], row_index, "Çıkış Transfer Merkezi"),
        _text(values[4], row_index, "Varış Transfer Merkezi"),
        datetime.combine(_date(values[5], row_index), _time(values[6], row_index)),
        datetime.combine(_date(values[7], row_index), _time(values[8], row_index)),
        demand_part_id,
        _number(values[10], row_index, "Taşınan Desi"),
        _number(values[11], row_index, "Yolculuk süresi"),
        _number(values[12], row_index, "Varış elleçleme süresi"),
        _number(values[13], row_index, "Çıkış Elleçleme süresi"),
        _number(values[14], row_index, "SLA cezası"),
        _number(values[15], row_index, "Toplam maliyet"),
    )


def _parse_demand_id(
    value: object,
    carried_desi: object,
    empty_demand_id: object,
    row_index: int,
) -> str | None:
    """Decode one atomic ID, or ``None`` for the zero-desi mandatory placeholder."""

    desi = _number(carried_desi, row_index, "Taşınan Desi")
    if str(value) == str(empty_demand_id) and math.isclose(
        desi,
        0.0,
        abs_tol=ZERO_DESI_TOLERANCE,
    ):
        return None
    if not isinstance(value, str) or not value.strip():
        raise TransportPlanWorkbookError(
            f"Satır {row_index}: pozitif taşıma satırında Talep ID eksik."
        )
    return value.strip()


def _require_equal_rows(
    expected: Sequence[TransportPlanWorkbookRow],
    actual: Sequence[TransportPlanWorkbookRow],
    cost_tolerance: float,
) -> None:
    """Require exact identifiers and tolerant Excel numeric/time equality."""

    if len(expected) != len(actual):
        raise TransportPlanWorkbookError(
            f"Excel round-trip satır sayısı değişti: {len(expected)} != {len(actual)}"
        )
    for index, (left, right) in enumerate(zip(expected, actual, strict=True), start=2):
        _require_equal_row(left, right, index, cost_tolerance)


def _require_equal_row(
    expected: TransportPlanWorkbookRow,
    actual: TransportPlanWorkbookRow,
    row_index: int,
    cost_tolerance: float,
) -> None:
    """Compare one row while allowing only Excel serialization precision."""

    exact_left = (
        expected.vehicle_id, expected.vehicle_ownership_label, expected.vehicle_type,
        expected.origin, expected.destination, expected.demand_part_id,
    )
    exact_right = (
        actual.vehicle_id, actual.vehicle_ownership_label, actual.vehicle_type,
        actual.origin, actual.destination, actual.demand_part_id,
    )
    if exact_left != exact_right:
        raise TransportPlanWorkbookError(f"Excel round-trip satır {row_index} kimliği değişti.")
    for left, right in (
        (expected.departure_at, actual.departure_at),
        (expected.arrival_at, actual.arrival_at),
    ):
        if abs((left - right).total_seconds()) > EXCEL_TIME_TOLERANCE_SECONDS:
            raise TransportPlanWorkbookError(f"Excel round-trip satır {row_index} zamanı değişti.")
    numeric_pairs = (
        (expected.carried_desi, actual.carried_desi, NUMERIC_ROUND_TRIP_TOLERANCE),
        (expected.travel_hours, actual.travel_hours, NUMERIC_ROUND_TRIP_TOLERANCE),
        (
            expected.arrival_handling_hours,
            actual.arrival_handling_hours,
            NUMERIC_ROUND_TRIP_TOLERANCE,
        ),
        (
            expected.departure_handling_hours,
            actual.departure_handling_hours,
            NUMERIC_ROUND_TRIP_TOLERANCE,
        ),
        (expected.sla_penalty_tl, actual.sla_penalty_tl, cost_tolerance),
        (expected.total_cost_tl, actual.total_cost_tl, cost_tolerance),
    )
    differs = any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=tol)
        for left, right, tol in numeric_pairs
    )
    if differs:
        raise TransportPlanWorkbookError(
            f"Excel round-trip satır {row_index} sayısı değişti."
        )


def _audit(rows: Sequence[TransportPlanWorkbookRow]) -> TransportPlanWorkbookAudit:
    """Aggregate row count, physical vehicles, total TL and SLA TL."""

    return TransportPlanWorkbookAudit(
        len(rows),
        len({row.vehicle_id for row in rows}),
        math.fsum(row.total_cost_tl for row in rows),
        math.fsum(row.sla_penalty_tl for row in rows),
    )


def _operation_hours(operation: Any | None) -> float:
    """Return linked handling duration in decimal hours or zero when absent."""

    return 0.0 if operation is None else _hours(operation.start_at, operation.end_at)


def _hours(start_at: datetime, end_at: datetime) -> float:
    """Convert a non-negative datetime interval to decimal hours."""

    seconds = (end_at - start_at).total_seconds()
    if seconds < 0.0:
        raise TransportPlanWorkbookError("Negatif süre Excel'e aktarılamaz.")
    return seconds / SECONDS_PER_HOUR


def _cost_tolerance(manifest: Mapping[str, Any]) -> float:
    """Read the existing plan cost comparison tolerance in Turkish lira."""

    value = manifest.get("validation", {}).get("solution", {}).get(
        "cost_tolerance_tl", DEFAULT_COST_TOLERANCE_TL
    )
    if not isinstance(value, (int, float)) or value < 0.0:
        raise ManifestError("validation.solution.cost_tolerance_tl geçersiz.")
    return float(value)


def _text(value: object, row_index: int, field: str) -> str:
    """Require one non-empty text cell with row/field context."""

    if not isinstance(value, str) or not value.strip():
        raise TransportPlanWorkbookError(f"Satır {row_index}: {field} metin olmalıdır.")
    return value.strip()


def _number(value: object, row_index: int, field: str) -> float:
    """Require one finite numeric cell without accepting numeric text."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransportPlanWorkbookError(f"Satır {row_index}: {field} sayısal olmalıdır.")
    result = float(value)
    if not math.isfinite(result):
        raise TransportPlanWorkbookError(f"Satır {row_index}: {field} sonlu olmalıdır.")
    return result


def _date(value: object, row_index: int) -> date:
    """Require a typed Excel date cell and return its calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TransportPlanWorkbookError(f"Satır {row_index}: tarih hücresi typed olmalıdır.")


def _time(value: object, row_index: int) -> time:
    """Require a typed Excel time cell and return its local wall-clock time."""

    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time):
        return value
    raise TransportPlanWorkbookError(f"Satır {row_index}: saat hücresi typed olmalıdır.")


__all__ = [
    "TransportPlanWorkbookAudit",
    "TransportPlanWorkbookError",
    "TransportPlanWorkbookExport",
    "TransportPlanWorkbookExporter",
    "TransportPlanWorkbookRow",
    "read_transport_plan_workbook",
]
