"""Reusable schema and cross-table validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable data problem found while loading a dataset."""

    code: str
    dataset: str
    message: str
    field: str | None = None
    row_number: int | None = None
    value: Any = None

    def format(self) -> str:
        """Render one loader issue as a concise human-readable line."""

        location = self.dataset
        if self.row_number is not None:
            location += f" satır {self.row_number}"
        if self.field:
            location += f" / {self.field}"
        return f"[{self.code}] {location}: {self.message}"


class DataValidationError(ValueError):
    """Raised with all validation issues collected in one pass."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        """Store every issue and build a bounded exception message preview."""

        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("DataValidationError en az bir issue gerektirir.")
        preview = "\n".join(issue.format() for issue in self.issues[:20])
        suffix = (
            f"\n... ve {len(self.issues) - 20} sorun daha"
            if len(self.issues) > 20
            else ""
        )
        super().__init__(
            f"Veri doğrulaması {len(self.issues)} sorunla başarısız oldu:\n"
            f"{preview}{suffix}"
        )


def excel_row_number(index: Any) -> int | None:
    """Translate a zero-based default DataFrame index to an Excel row number."""

    try:
        return int(index) + 2
    except (TypeError, ValueError):
        return None


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataset: str,
) -> list[ValidationIssue]:
    """Report missing or repeated Excel headers before typed parsing."""

    issues: list[ValidationIssue] = []
    duplicate_headers = frame.columns[frame.columns.duplicated()].tolist()
    for header in duplicate_headers:
        issues.append(
            ValidationIssue(
                "duplicate_column",
                dataset,
                f"Sütun adı birden fazla kez bulunuyor: {header!r}",
                field=str(header),
            )
        )
    for column in required_columns:
        if column not in frame.columns:
            issues.append(
                ValidationIssue(
                    "missing_column",
                    dataset,
                    f"Zorunlu sütun bulunamadı: {column!r}",
                    field=column,
                )
            )
    return issues


def canonicalize_columns(
    frame: pd.DataFrame,
    column_mapping: Mapping[str, str],
    *,
    dataset: str,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Select configured source columns and rename them to canonical names."""

    issues = validate_required_columns(frame, column_mapping.values(), dataset=dataset)
    if issues:
        return pd.DataFrame(), issues
    source_names = list(column_mapping.values())
    if len(source_names) != len(set(source_names)):
        issues.append(
            ValidationIssue(
                "duplicate_column_mapping",
                dataset,
                "Bir Excel sütunu birden fazla kanonik alana eşlenmiş.",
            )
        )
        return pd.DataFrame(), issues
    canonical = frame.loc[:, source_names].rename(
        columns={source: canonical for canonical, source in column_mapping.items()}
    )
    return canonical, issues


def _is_blank(value: Any) -> bool:
    """Treat null-like and whitespace-only source cells as missing."""

    if value is None or value is pd.NA:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def parse_text_column(
    series: pd.Series, *, dataset: str, field: str
) -> tuple[pd.Series, list[ValidationIssue]]:
    """Normalize a required text column and collect row-level type issues."""

    issues: list[ValidationIssue] = []
    parsed: list[str | None] = []
    for index, value in series.items():
        if _is_blank(value):
            parsed.append(None)
            issues.append(
                ValidationIssue(
                    "missing_value",
                    dataset,
                    "Boş değer kabul edilmez.",
                    field,
                    excel_row_number(index),
                    value,
                )
            )
        elif not isinstance(value, str):
            parsed.append(None)
            issues.append(
                ValidationIssue(
                    "invalid_type",
                    dataset,
                    "Metin değer bekleniyor.",
                    field,
                    excel_row_number(index),
                    value,
                )
            )
        else:
            parsed.append(value.strip())
    return pd.Series(parsed, index=series.index, dtype="object"), issues


def parse_numeric_column(
    series: pd.Series,
    *,
    dataset: str,
    field: str,
    integer: bool = False,
    allow_negative: bool = False,
) -> tuple[pd.Series, list[ValidationIssue]]:
    """Parse finite numeric cells while enforcing integer and sign rules."""

    issues: list[ValidationIssue] = []
    parsed: list[float | int | None] = []
    for index, value in series.items():
        parsed_value, found = _parse_numeric_cell(
            value,
            dataset=dataset,
            field=field,
            row_number=excel_row_number(index),
            integer=integer,
            allow_negative=allow_negative,
        )
        parsed.append(parsed_value)
        issues.extend(found)
    return pd.Series(parsed, index=series.index, dtype="object"), issues


def _parse_numeric_cell(
    value: Any,
    *,
    dataset: str,
    field: str,
    row_number: int,
    integer: bool,
    allow_negative: bool,
) -> tuple[float | int | None, list[ValidationIssue]]:
    """Parse one numeric cell and retain all relevant sign/type issues."""

    def issue(code: str, message: str) -> ValidationIssue:
        """Attach repeated dataset/field/row context to one cell issue."""

        return ValidationIssue(
            code,
            dataset,
            message,
            field,
            row_number,
            value,
        )

    if _is_blank(value):
        return None, [issue("missing_value", "Boş sayısal değer kabul edilmez.")]
    if isinstance(value, bool):
        return None, [
            issue(
                "invalid_type",
                "Sayısal değer bekleniyor; boolean kabul edilmez.",
            )
        ]
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, [issue("invalid_type", "Sayısal değer bekleniyor.")]
    if not math.isfinite(number):
        return None, [issue("invalid_type", "Sonlu bir sayı bekleniyor.")]

    issues = []
    if not allow_negative and number < 0:
        issues.append(issue("negative_value", "Negatif değer kabul edilmez."))
    if integer and not number.is_integer():
        issues.append(issue("invalid_type", "Tamsayı değer bekleniyor."))
        return None, issues
    return (int(number) if integer else number), issues


def parse_date_value(value: Any) -> date | None:
    """Parse supported Excel, ISO and Turkish day-first date values."""

    if _is_blank(value) or isinstance(value, time):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        try:
            return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def parse_date_column(
    series: pd.Series, *, dataset: str, field: str
) -> tuple[pd.Series, list[ValidationIssue]]:
    """Parse a required date column and return structured row issues."""

    issues: list[ValidationIssue] = []
    parsed: list[date | None] = []
    for index, value in series.items():
        result = parse_date_value(value)
        parsed.append(result)
        if result is None:
            issues.append(
                ValidationIssue(
                    "invalid_date",
                    dataset,
                    "Geçerli bir tarih bekleniyor.",
                    field,
                    excel_row_number(index),
                    value,
                )
            )
    return pd.Series(parsed, index=series.index, dtype="object"), issues


def parse_time_value(value: Any) -> time | None:
    """Parse Excel day fractions and common text/datetime time values."""

    if _is_blank(value) or isinstance(value, date) and not isinstance(value, datetime):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.time().replace(tzinfo=None)
    if isinstance(value, datetime):
        return value.time().replace(tzinfo=None)
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
        if not 0 <= seconds < 86_400:
            return None
        whole_seconds = round(seconds)
        return (datetime.min + timedelta(seconds=whole_seconds)).time()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric < 1:
            return None
        whole_seconds = round(numeric * 86_400)
        if whole_seconds == 86_400:
            whole_seconds = 0
        return (datetime.min + timedelta(seconds=whole_seconds)).time()
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = time.fromisoformat(text)
        except ValueError:
            parsed = None
            for fmt in ("%H:%M", "%H:%M:%S", "%H:%M:%S.%f"):
                try:
                    parsed = datetime.strptime(text, fmt).time()
                    break
                except ValueError:
                    continue
            if parsed is None:
                return None
        return parsed.replace(tzinfo=None)
    return None


def parse_time_column(
    series: pd.Series, *, dataset: str, field: str
) -> tuple[pd.Series, list[ValidationIssue]]:
    """Parse a required time column and return structured row issues."""

    issues: list[ValidationIssue] = []
    parsed: list[time | None] = []
    for index, value in series.items():
        result = parse_time_value(value)
        parsed.append(result)
        if result is None:
            issues.append(
                ValidationIssue(
                    "invalid_time",
                    dataset,
                    "Geçerli bir saat bekleniyor.",
                    field,
                    excel_row_number(index),
                    value,
                )
            )
    return pd.Series(parsed, index=series.index, dtype="object"), issues


def validate_duplicates(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    dataset: str,
) -> list[ValidationIssue]:
    """Report every row participating in a duplicated business key."""

    if frame.empty or any(column not in frame.columns for column in key_columns):
        return []
    duplicate_mask = frame.duplicated(subset=list(key_columns), keep=False)
    issues: list[ValidationIssue] = []
    for index in frame.index[duplicate_mask]:
        key = {column: frame.at[index, column] for column in key_columns}
        issues.append(
            ValidationIssue(
                "duplicate_record",
                dataset,
                f"İş anahtarı tekrar ediyor: {key!r}",
                row_number=excel_row_number(index),
                value=key,
            )
        )
    return issues


def validate_known_centers(
    frame: pd.DataFrame,
    center_columns: Sequence[str],
    known_centers: Iterable[str],
    *,
    dataset: str,
) -> list[ValidationIssue]:
    """Verify selected columns reference only centres from the network."""

    known = set(known_centers)
    issues: list[ValidationIssue] = []
    for column in center_columns:
        if column not in frame.columns:
            continue
        for index, value in frame[column].items():
            if value is not None and value not in known:
                issues.append(
                    ValidationIssue(
                        "unknown_center",
                        dataset,
                        f"Merkez ağ ana verisinde bulunmuyor: {value!r}",
                        column,
                        excel_row_number(index),
                        value,
                    )
                )
    return issues


def validate_complete_directed_network(
    frame: pd.DataFrame,
    known_centers: Iterable[str],
    *,
    dataset: str = "network",
) -> list[ValidationIssue]:
    """Check every ordered pair of distinct centers has distance and times."""

    if frame.empty or not {"origin", "destination"}.issubset(frame.columns):
        return []
    centers = set(known_centers)
    actual = set(zip(frame["origin"], frame["destination"], strict=False))
    expected = {(origin, destination) for origin in centers for destination in centers if origin != destination}
    missing = sorted(expected - actual)
    issues = [
        ValidationIssue(
            "missing_arc",
            dataset,
            f"Mesafe/seyahat süresi kaydı eksik: {origin!r} -> {destination!r}",
            value=(origin, destination),
        )
        for origin, destination in missing
    ]
    for index, row in frame.iterrows():
        if row.get("origin") == row.get("destination"):
            issues.append(
                ValidationIssue(
                    "self_arc",
                    dataset,
                    "Çıkış ve varış merkezi aynı olamaz.",
                    row_number=excel_row_number(index),
                    value=row.get("origin"),
                )
            )
    return issues
