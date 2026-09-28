"""Read-only Excel loaders for the linehaul problem data.

Only this module knows workbook names, sheets and source headers, and even
those values come from ``config/rules.yaml``.  The loader never saves,
renames, moves or otherwise mutates a source workbook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import time
from pathlib import Path

import pandas as pd

from src.domain import (
    Center,
    DailyTruckCapacity,
    DemandInstance,
    HandlingCapacity,
    MandatoryRentalRoute,
    ProblemData,
    StaticProblemData,
    TravelArc,
    VehicleType,
)

from .manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestError,
    file_spec,
    load_manifest,
    project_root_for,
    raw_data_directory,
    resolve_source_file,
)
from .validation import (
    DataValidationError,
    ValidationIssue,
    canonicalize_columns,
    excel_row_number,
    parse_date_column,
    parse_numeric_column,
    parse_text_column,
    parse_time_column,
    parse_time_value,
    validate_complete_directed_network,
    validate_duplicates,
    validate_known_centers,
)


STATIC_DATASETS = (
    "network",
    "vehicle_types",
    "mandatory_rentals",
    "handling_capacities",
    "truck_capacities",
)

DEMAND_FIELDS = (
    "demand_id",
    "demand_date",
    "time_slot",
    "origin",
    "destination",
    "desi",
)


class SourceReadError(RuntimeError):
    """Raised when a configured workbook or sheet cannot be read."""


class ExcelProblemLoader:
    """Load and validate source workbooks into source-agnostic dataclasses."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        """Load the central manifest and resolve its protected raw directory."""

        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
        )
        self.manifest = load_manifest(self.manifest_path)
        inferred_root = project_root_for(self.manifest_path)
        self.project_root = Path(project_root or inferred_root).resolve()
        self.raw_directory = raw_data_directory(
            self.manifest, project_root=self.project_root
        )

    def source_path(self, logical_name: str) -> Path:
        """Return the resolved physical path for a logical manifest key."""

        spec = file_spec(self.manifest, logical_name)
        return resolve_source_file(self.raw_directory, str(spec["filename"]))

    def read_canonical_frame(
        self, logical_name: str
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Read one sheet and apply its configured canonical column names."""

        return self._read_canonical_frame_at_path(
            self.source_path(logical_name),
            logical_name,
        )

    def read_canonical_frame_from_workbook(
        self,
        workbook_path: str | Path,
        schema_logical_name: str,
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Read a generated workbook using one manifest schema.

        Args:
            workbook_path: Existing Excel file outside the protected raw-data
                resolution flow, normally ``data/interim`` or a test folder.
            schema_logical_name: Manifest file entry that supplies sheet and
                source-to-canonical column names.

        Returns:
            Canonical ``DataFrame`` and non-throwing header issues.  Forecast
            round-trip checks call this method before domain construction.

        Raises:
            SourceReadError: If the workbook or configured sheet cannot be
                opened.  The method is strictly read-only.
        """

        return self._read_canonical_frame_at_path(
            Path(workbook_path).resolve(),
            schema_logical_name,
        )

    def _read_canonical_frame_at_path(
        self,
        path: Path,
        logical_name: str,
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Read one explicit Excel path with a manifest-backed schema."""

        spec = file_spec(self.manifest, logical_name)
        try:
            # pandas/openpyxl opens the workbook for reading and closes it after
            # parsing.  There is intentionally no ExcelWriter/save operation in
            # this package.
            frame = pd.read_excel(
                path,
                sheet_name=str(spec["sheet"]),
                engine="openpyxl",
                dtype=object,
            )
        except (OSError, ValueError, ImportError) as exc:
            raise SourceReadError(
                f"{logical_name!r} kaynağı okunamadı: {path} / "
                f"sayfa {spec['sheet']!r}: {exc}"
            ) from exc
        return canonicalize_columns(
            frame, spec["columns"], dataset=logical_name
        )

    def load_static_problem_data(self) -> StaticProblemData:
        """Load, cross-check and construct the immutable static data layer."""

        frames, issues = self._read_static_frames()
        if issues:
            raise DataValidationError(issues)
        typed_frames, parsing_issues = self._parse_static_frames(frames)
        issues.extend(parsing_issues)
        centers, travel_type_mapping, relation_issues = (
            self._validate_static_relationships(typed_frames)
        )
        issues.extend(relation_issues)
        if issues:
            raise DataValidationError(issues)
        return self._construct_static_data(
            typed_frames,
            centers,
            travel_type_mapping,
        )

    def _read_static_frames(
        self,
    ) -> tuple[dict[str, pd.DataFrame], list[ValidationIssue]]:
        """Read every static workbook so schema errors can be reported together."""

        frames: dict[str, pd.DataFrame] = {}
        issues: list[ValidationIssue] = []
        for dataset in STATIC_DATASETS:
            frame, frame_issues = self.read_canonical_frame(dataset)
            frames[dataset] = frame
            issues.extend(frame_issues)
        return frames, issues

    def _parse_static_frames(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], list[ValidationIssue]]:
        """Apply dataset-specific typed parsers to all static canonical frames."""

        parsers = {
            "network": self._parse_network,
            "vehicle_types": self._parse_vehicle_types,
            "mandatory_rentals": self._parse_mandatory_rentals,
            "handling_capacities": self._parse_handling_capacities,
            "truck_capacities": self._parse_truck_capacities,
        }
        typed: dict[str, pd.DataFrame] = {}
        issues: list[ValidationIssue] = []
        for dataset, parser in parsers.items():
            typed[dataset], found = parser(frames[dataset])
            issues.extend(found)
        return typed, issues

    def _validate_static_relationships(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[set[str], Mapping[str, str], list[ValidationIssue]]:
        """Cross-check center, vehicle and directed-network master relations."""

        issues = self._validate_static_business_keys(frames)
        centers = self._network_center_names(frames["network"])
        issues.extend(self._validate_static_center_relations(frames, centers))
        travel_type_mapping, vehicle_issues = self._validate_vehicle_relations(frames)
        issues.extend(vehicle_issues)
        validation_config = self.manifest.get("validation", {})
        if validation_config.get("require_complete_directed_network", True):
            issues.extend(
                validate_complete_directed_network(
                    frames["network"], centers, dataset="network"
                )
            )
        return centers, travel_type_mapping, issues

    def _validate_static_center_relations(
        self,
        frames: dict[str, pd.DataFrame],
        centers: set[str],
    ) -> list[ValidationIssue]:
        """Validate static center references and required capacity coverage."""

        issues: list[ValidationIssue] = []
        if not centers:
            issues.append(
                ValidationIssue(
                    "empty_dataset", "network", "Ağ en az bir merkez içermelidir."
                )
            )
        for dataset, fields in (
            ("mandatory_rentals", ("origin", "destination")),
            ("handling_capacities", ("center",)),
            ("truck_capacities", ("center",)),
        ):
            issues.extend(
                validate_known_centers(
                    frames[dataset], fields, centers, dataset=dataset
                )
            )
        issues.extend(
            self._validate_center_coverage(
                frames["handling_capacities"],
                centers,
                dataset="handling_capacities",
                value_name="elleçleme kapasitesi",
            )
        )
        issues.extend(
            self._validate_center_coverage(
                frames["truck_capacities"],
                centers,
                dataset="truck_capacities",
                value_name="günlük Tır kapasitesi",
            )
        )
        return issues

    def _validate_vehicle_relations(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[Mapping[str, str], list[ValidationIssue]]:
        """Validate mandatory vehicle names and matrix time-profile mapping."""

        vehicle_names = set(frames["vehicle_types"]["name"].dropna())
        issues = self._validate_known_values(
            frames["mandatory_rentals"],
            "vehicle_type",
            vehicle_names,
            dataset="mandatory_rentals",
            code="unknown_vehicle_type",
            noun="araç türü",
        )
        mapping = file_spec(self.manifest, "network").get(
            "travel_time_vehicle_types", {}
        )
        if not isinstance(mapping, Mapping):
            raise ManifestError(
                "files.network.travel_time_vehicle_types bir nesne/map olmalıdır."
            )
        mapped_names = set(mapping.values())
        for vehicle_name in sorted(vehicle_names - mapped_names):
            issues.append(
                ValidationIssue(
                    "missing_travel_time_profile",
                    "network",
                    f"Araç türü için seyahat süresi sütunu eşlenmemiş: {vehicle_name!r}",
                    value=vehicle_name,
                )
            )
        for vehicle_name in sorted(mapped_names - vehicle_names):
            issues.append(
                ValidationIssue(
                    "unknown_vehicle_type",
                    "network",
                    f"Seyahat süresi eşlemesi araç ana verisinde yok: {vehicle_name!r}",
                    value=vehicle_name,
                )
            )
        return mapping, issues

    def _construct_static_data(
        self,
        frames: dict[str, pd.DataFrame],
        centers: set[str],
        travel_type_mapping: Mapping[str, str],
    ) -> StaticProblemData:
        """Convert fully validated static frames to immutable domain models."""

        typed_network = frames["network"]
        typed_vehicles = frames["vehicle_types"]
        typed_rentals = frames["mandatory_rentals"]
        typed_handling = frames["handling_capacities"]
        typed_truck = frames["truck_capacities"]
        travel_arcs = tuple(
            TravelArc(
                origin=row.origin,
                destination=row.destination,
                distance_km=float(row.distance_km),
                travel_hours_by_vehicle={
                    str(vehicle_type): float(getattr(row, canonical_field))
                    for canonical_field, vehicle_type in travel_type_mapping.items()
                },
                target_delivery_days=int(row.target_delivery_days),
            )
            for row in typed_network.itertuples(index=False)
        )
        vehicle_types = tuple(
            VehicleType(
                name=row.name,
                capacity_desi=float(row.capacity_desi),
                rented_hourly_cost_tl=float(row.rented_hourly_cost_tl),
                rented_cost_per_km_tl=float(row.rented_cost_per_km_tl),
                spot_hourly_cost_tl=float(row.spot_hourly_cost_tl),
                spot_cost_per_km_tl=float(row.spot_cost_per_km_tl),
            )
            for row in typed_vehicles.itertuples(index=False)
        )
        rentals = tuple(
            MandatoryRentalRoute(
                origin=row.origin,
                destination=row.destination,
                vehicle_type=row.vehicle_type,
                vehicle_count=int(row.vehicle_count),
            )
            for row in typed_rentals.itertuples(index=False)
        )
        handling = tuple(
            HandlingCapacity(
                center=row.center,
                max_daily_desi=float(row.max_daily_desi),
            )
            for row in typed_handling.itertuples(index=False)
        )
        truck_capacities = tuple(
            DailyTruckCapacity(
                center=row.center,
                max_daily_truck_visits=int(row.max_daily_truck_visits),
            )
            for row in typed_truck.itertuples(index=False)
        )
        return StaticProblemData(
            centers=tuple(Center(name) for name in sorted(centers)),
            travel_arcs=travel_arcs,
            vehicle_types=vehicle_types,
            mandatory_rental_routes=rentals,
            handling_capacities=handling,
            daily_truck_capacities=truck_capacities,
        )

    def load_demands(
        self,
        logical_name: str = "historical_demand",
        *,
        static_data: StaticProblemData | None = None,
    ) -> tuple[DemandInstance, ...]:
        """Load one configured demand workbook and mark oversized rows.

        Exceeding the largest vehicle capacity is intentionally not a
        validation error.  Such rows receive ``split_required=True``.
        """

        self._validate_demand_schema_role(logical_name)
        static = static_data or self.load_static_problem_data()
        frame, issues = self.read_canonical_frame(logical_name)
        return self._construct_demands_from_frame(
            frame,
            issues,
            dataset=logical_name,
            static=static,
        )

    def load_demands_from_workbook(
        self,
        workbook_path: str | Path,
        *,
        schema_logical_name: str = "forecast_demand_template",
        static_data: StaticProblemData | None = None,
    ) -> tuple[DemandInstance, ...]:
        """Reload a generated demand workbook as standard domain records.

        Args:
            workbook_path: Excel path produced by the development forecast CLI
                or supplied by another forecast provider.
            schema_logical_name: Manifest schema defining sheet and official
                demand headers.
            static_data: Optional preloaded network and vehicle master data.

        Returns:
            Validated immutable ``DemandInstance`` tuple with ``split_required``
            recomputed from the largest vehicle capacity in desi.

        Raises:
            ManifestError, SourceReadError, DataValidationError: For an invalid
            role, unreadable workbook, schema/type issue or unknown O-D line.
            The source and generated workbooks are never modified.
        """

        self._validate_demand_schema_role(schema_logical_name)
        static = static_data or self.load_static_problem_data()
        frame, issues = self.read_canonical_frame_from_workbook(
            workbook_path,
            schema_logical_name,
        )
        return self._construct_demands_from_frame(
            frame,
            issues,
            dataset=schema_logical_name,
            static=static,
        )

    def _validate_demand_schema_role(self, logical_name: str) -> None:
        """Reject manifest entries that are not demand-shaped datasets."""

        spec = file_spec(self.manifest, logical_name)
        if spec.get("role") not in (None, "demand", "demand_template"):
            raise ManifestError(
                f"{logical_name!r} talep kaynağı olarak işaretlenmemiş."
            )

    def _construct_demands_from_frame(
        self,
        frame: pd.DataFrame,
        issues: list[ValidationIssue],
        *,
        dataset: str,
        static: StaticProblemData,
    ) -> tuple[DemandInstance, ...]:
        """Parse and validate one canonical demand frame into domain models."""

        if issues:
            raise DataValidationError(issues)
        missing_canonical = [field for field in DEMAND_FIELDS if field not in frame]
        if missing_canonical:
            raise ManifestError(
                f"files.{dataset}.columns talep alanlarını eksik eşliyor: "
                + ", ".join(missing_canonical)
            )

        typed, parsing_issues = self._parse_demands(frame, dataset=dataset)
        issues.extend(parsing_issues)
        issues.extend(self._validate_demand_relations(typed, dataset, static))
        if issues:
            raise DataValidationError(issues)

        maximum_capacity = static.maximum_vehicle_capacity_desi
        return tuple(
            DemandInstance(
                demand_id=row.demand_id,
                demand_date=row.demand_date,
                time_slot=row.time_slot,
                origin=row.origin,
                destination=row.destination,
                desi=float(row.desi),
                split_required=float(row.desi) > maximum_capacity,
            )
            for row in typed.itertuples(index=False)
        )

    def _validate_demand_relations(
        self,
        frame: pd.DataFrame,
        dataset: str,
        static: StaticProblemData,
    ) -> list[ValidationIssue]:
        """Validate demand keys, centers, directed arcs, slots and O-D shape."""

        issues = self._validate_business_key(frame, dataset)
        issues.extend(
            validate_known_centers(
                frame,
                ("origin", "destination"),
                static.center_names,
                dataset=dataset,
            )
        )
        available_arcs = {
            (arc.origin, arc.destination) for arc in static.travel_arcs
        }
        for index, row in frame.iterrows():
            route = (row.get("origin"), row.get("destination"))
            if None not in route and route not in available_arcs:
                issues.append(
                    ValidationIssue(
                        "missing_arc",
                        dataset,
                        "Talep O-D çifti için mesafe/seyahat süresi kaydı yok: "
                        f"{route[0]!r} -> {route[1]!r}",
                        row_number=excel_row_number(index),
                        value=route,
                    )
                )
            if route[0] is not None and route[0] == route[1]:
                issues.append(
                    ValidationIssue(
                        "invalid_route",
                        dataset,
                        "Talep çıkış ve varış merkezi aynı olamaz.",
                        row_number=excel_row_number(index),
                        value=route[0],
                    )
                )
        issues.extend(self._validate_allowed_time_slots(frame, dataset=dataset))
        return issues

    def load_problem_data(
        self, demand_logical_name: str = "historical_demand"
    ) -> ProblemData:
        """Load validated static data and one variable demand instance."""

        static = self.load_static_problem_data()
        demands = self.load_demands(demand_logical_name, static_data=static)
        return ProblemData(static=static, demands=demands)

    def _parse_network(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Convert canonical network columns to typed values for validation."""

        dataset = "network"
        output = frame.copy()
        issues: list[ValidationIssue] = []
        for field in ("origin", "destination"):
            output[field], found = parse_text_column(
                output[field], dataset=dataset, field=field
            )
            issues.extend(found)
        numeric_fields = (
            "distance_km",
            "travel_hours_truck",
            "travel_hours_medium_truck",
            "travel_hours_light_truck",
            "travel_hours_van",
        )
        for field in numeric_fields:
            output[field], found = parse_numeric_column(
                output[field], dataset=dataset, field=field
            )
            issues.extend(found)
        output["target_delivery_days"], found = parse_numeric_column(
            output["target_delivery_days"],
            dataset=dataset,
            field="target_delivery_days",
            integer=True,
        )
        issues.extend(found)
        return output, issues

    def _parse_vehicle_types(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Convert canonical vehicle capacity and cost columns to numbers."""

        dataset = "vehicle_types"
        output = frame.copy()
        issues: list[ValidationIssue] = []
        output["name"], found = parse_text_column(
            output["name"], dataset=dataset, field="name"
        )
        issues.extend(found)
        for field in (
            "capacity_desi",
            "rented_hourly_cost_tl",
            "rented_cost_per_km_tl",
            "spot_hourly_cost_tl",
            "spot_cost_per_km_tl",
        ):
            output[field], found = parse_numeric_column(
                output[field], dataset=dataset, field=field
            )
            issues.extend(found)
        return output, issues

    def _parse_mandatory_rentals(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Parse fixed-route mandatory rental counts and text identifiers."""

        dataset = "mandatory_rentals"
        output = frame.copy()
        issues: list[ValidationIssue] = []
        for field in ("origin", "destination", "vehicle_type"):
            output[field], found = parse_text_column(
                output[field], dataset=dataset, field=field
            )
            issues.extend(found)
        output["vehicle_count"], found = parse_numeric_column(
            output["vehicle_count"],
            dataset=dataset,
            field="vehicle_count",
            integer=True,
        )
        issues.extend(found)
        return output, issues

    def _parse_handling_capacities(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Parse daily centre handling capacity records."""

        dataset = "handling_capacities"
        output = frame.copy()
        issues: list[ValidationIssue] = []
        output["center"], found = parse_text_column(
            output["center"], dataset=dataset, field="center"
        )
        issues.extend(found)
        output["max_daily_desi"], found = parse_numeric_column(
            output["max_daily_desi"], dataset=dataset, field="max_daily_desi"
        )
        issues.extend(found)
        return output, issues

    def _parse_truck_capacities(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Parse daily integer Tır visit quotas for each centre."""

        dataset = "truck_capacities"
        output = frame.copy()
        issues: list[ValidationIssue] = []
        output["center"], found = parse_text_column(
            output["center"], dataset=dataset, field="center"
        )
        issues.extend(found)
        output["max_daily_truck_visits"], found = parse_numeric_column(
            output["max_daily_truck_visits"],
            dataset=dataset,
            field="max_daily_truck_visits",
            integer=True,
        )
        issues.extend(found)
        return output, issues

    def _parse_demands(
        self, frame: pd.DataFrame, *, dataset: str
    ) -> tuple[pd.DataFrame, list[ValidationIssue]]:
        """Parse IDs, dates, slots, centres and desi for one demand source."""

        output = frame.copy()
        issues: list[ValidationIssue] = []
        for field in ("demand_id", "origin", "destination"):
            output[field], found = parse_text_column(
                output[field], dataset=dataset, field=field
            )
            issues.extend(found)
        output["demand_date"], found = parse_date_column(
            output["demand_date"], dataset=dataset, field="demand_date"
        )
        issues.extend(found)
        output["time_slot"], found = parse_time_column(
            output["time_slot"], dataset=dataset, field="time_slot"
        )
        issues.extend(found)
        output["desi"], found = parse_numeric_column(
            output["desi"], dataset=dataset, field="desi"
        )
        issues.extend(found)
        return output, issues

    def _validate_static_business_keys(
        self, frames: Mapping[str, pd.DataFrame]
    ) -> list[ValidationIssue]:
        """Apply manifest-defined uniqueness keys to every static table."""

        issues: list[ValidationIssue] = []
        for dataset, frame in frames.items():
            issues.extend(self._validate_business_key(frame, dataset))
        return issues

    def _validate_business_key(
        self, frame: pd.DataFrame, dataset: str
    ) -> list[ValidationIssue]:
        """Validate one canonical frame against its configured business key."""

        keys = (
            self.manifest.get("validation", {})
            .get("business_keys", {})
            .get(dataset)
        )
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            raise ManifestError(
                f"validation.business_keys.{dataset} liste olarak tanımlanmalıdır."
            )
        missing_keys = [key for key in keys if key not in frame.columns]
        if missing_keys:
            raise ManifestError(
                f"validation.business_keys.{dataset} bilinmeyen alan içeriyor: "
                + ", ".join(missing_keys)
            )
        return validate_duplicates(frame, keys, dataset=dataset)

    @staticmethod
    def _network_center_names(frame: pd.DataFrame) -> set[str]:
        """Derive the canonical centre universe from directed network arcs."""

        return set(frame["origin"].dropna()) | set(frame["destination"].dropna())

    @staticmethod
    def _validate_center_coverage(
        frame: pd.DataFrame,
        known_centers: set[str],
        *,
        dataset: str,
        value_name: str,
    ) -> list[ValidationIssue]:
        """Report network centres missing a required capacity parameter."""

        present = set(frame["center"].dropna())
        return [
            ValidationIssue(
                "missing_center_parameter",
                dataset,
                f"Merkez için {value_name} kaydı eksik: {center!r}",
                value=center,
            )
            for center in sorted(known_centers - present)
        ]

    @staticmethod
    def _validate_known_values(
        frame: pd.DataFrame,
        field: str,
        known_values: set[str],
        *,
        dataset: str,
        code: str,
        noun: str,
    ) -> list[ValidationIssue]:
        """Check a canonical column against a static master-value set."""

        issues: list[ValidationIssue] = []
        for index, value in frame[field].items():
            if value is not None and value not in known_values:
                issues.append(
                    ValidationIssue(
                        code,
                        dataset,
                        f"Bilinmeyen {noun}: {value!r}",
                        field,
                        excel_row_number(index),
                        value,
                    )
                )
        return issues

    def _validate_allowed_time_slots(
        self, frame: pd.DataFrame, *, dataset: str
    ) -> list[ValidationIssue]:
        """Enforce manifest-defined demand completion time slots."""

        configured = self.manifest.get("validation", {}).get(
            "allowed_demand_time_slots"
        )
        if configured is None:
            return []
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise ManifestError("validation.allowed_demand_time_slots liste olmalıdır.")
        allowed: set[time] = set()
        for value in configured:
            parsed = parse_time_value(value)
            if parsed is None:
                raise ManifestError(
                    f"Manifestte geçersiz izinli saat dilimi: {value!r}"
                )
            allowed.add(parsed)
        issues: list[ValidationIssue] = []
        for index, value in frame["time_slot"].items():
            if value is not None and value not in allowed:
                issues.append(
                    ValidationIssue(
                        "invalid_time_slot",
                        dataset,
                        f"İzinli saat dilimlerinden biri bekleniyor: "
                        f"{', '.join(sorted(slot.strftime('%H:%M') for slot in allowed))}",
                        "time_slot",
                        excel_row_number(index),
                        value,
                    )
                )
        return issues


def load_static_problem_data(
    manifest_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> StaticProblemData:
    """Convenience API for loading only static master data."""

    return ExcelProblemLoader(
        manifest_path, project_root=project_root
    ).load_static_problem_data()


def load_demand_instances(
    logical_name: str = "historical_demand",
    manifest_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[DemandInstance, ...]:
    """Convenience API for one configured demand source."""

    return ExcelProblemLoader(manifest_path, project_root=project_root).load_demands(
        logical_name
    )


def load_problem_data(
    demand_logical_name: str = "historical_demand",
    manifest_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> ProblemData:
    """Convenience API for the combined static + variable data layer."""

    return ExcelProblemLoader(manifest_path, project_root=project_root).load_problem_data(
        demand_logical_name
    )
