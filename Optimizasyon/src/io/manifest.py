"""Manifest loading and Unicode-safe source file resolution.

The project keeps the logical-to-physical Excel file mapping in
``config/rules.yaml``.  That file deliberately uses the JSON subset of YAML,
so reading it does not require PyYAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import unicodedata
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "config" / "rules.yaml"


class ManifestError(ValueError):
    """Raised when the project data manifest is missing or malformed."""


def _normalized_filename(value: str) -> str:
    """Return a comparison key tolerant of composed/decomposed Unicode."""

    return unicodedata.normalize("NFC", value).casefold()


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Read and minimally validate the JSON-compatible YAML manifest.

    A defensive copy is returned so callers can adapt local settings without
    mutating state shared with another loader.
    """

    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    try:
        raw_text = manifest_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManifestError(f"Manifest okunamadı: {manifest_path}: {exc}") from exc

    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"Manifest JSON-uyumlu YAML olmalıdır: {manifest_path} "
            f"(satır {exc.lineno}, sütun {exc.colno})"
        ) from exc

    if not isinstance(document, dict):
        raise ManifestError("Manifest kökü bir nesne/map olmalıdır.")

    paths = document.get("paths")
    files = document.get("files")
    if not isinstance(paths, Mapping) or not isinstance(
        paths.get("raw_data_dir"), str
    ):
        raise ManifestError("Manifestte paths.raw_data_dir metin olarak tanımlanmalıdır.")
    if not isinstance(files, Mapping) or not files:
        raise ManifestError("Manifestte en az bir files kaydı bulunmalıdır.")

    for logical_name, entry in files.items():
        if not isinstance(logical_name, str) or not isinstance(entry, Mapping):
            raise ManifestError("Her files kaydı adlandırılmış bir nesne olmalıdır.")
        if not isinstance(entry.get("filename"), str):
            raise ManifestError(f"files.{logical_name}.filename eksik veya geçersiz.")
        if not isinstance(entry.get("sheet"), str):
            raise ManifestError(f"files.{logical_name}.sheet eksik veya geçersiz.")
        columns = entry.get("columns")
        if not isinstance(columns, Mapping) or not columns:
            raise ManifestError(f"files.{logical_name}.columns eksik veya boş.")
        if not all(
            isinstance(canonical, str) and isinstance(source, str)
            for canonical, source in columns.items()
        ):
            raise ManifestError(
                f"files.{logical_name}.columns anahtar ve değerleri metin olmalıdır."
            )

    return deepcopy(document)


def project_root_for(manifest_path: str | Path | None = None) -> Path:
    """Infer the repository root from a conventional ``config`` manifest."""

    if manifest_path is None:
        return PROJECT_ROOT
    path = Path(manifest_path).resolve()
    return path.parent.parent if path.parent.name.casefold() == "config" else Path.cwd()


def raw_data_directory(
    manifest: Mapping[str, Any], *, project_root: str | Path = PROJECT_ROOT
) -> Path:
    """Resolve the configured raw directory while keeping it inside the repo."""

    root = Path(project_root).resolve()
    configured = Path(str(manifest["paths"]["raw_data_dir"]))
    candidate = configured.resolve() if configured.is_absolute() else (root / configured).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(
            f"Ham veri klasörü proje kökü dışında olamaz: {candidate}"
        ) from exc
    if not candidate.is_dir():
        raise ManifestError(f"Ham veri klasörü bulunamadı: {candidate}")
    return candidate


def configured_project_path(
    manifest: Mapping[str, Any],
    path_key: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve one configured project path without allowing repo escape.

    Args:
        manifest: Mapping returned by :func:`load_manifest`.
        path_key: Key below ``paths`` such as ``development_forecast``.
        project_root: Repository boundary used for relative resolution.

    Returns:
        Absolute path inside ``project_root``.  Forecast CLI modules use this
        helper so output paths do not spread through algorithm code.

    Raises:
        ManifestError: If the key is missing, is not text, or resolves outside
            the repository.  The function does not create or modify the path.
    """

    configured = manifest.get("paths", {}).get(path_key)
    if not isinstance(configured, str) or not configured.strip():
        raise ManifestError(f"Manifestte paths.{path_key} metin olmalıdır.")
    root = Path(project_root).resolve()
    path = Path(configured)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(
            f"Yapılandırılmış proje yolu repo dışında olamaz: {candidate}"
        ) from exc
    return candidate


def resolve_source_file(raw_directory: str | Path, configured_filename: str) -> Path:
    """Resolve one direct child using Unicode-normalized, casefolded matching.

    Exact path lookup is attempted first.  The fallback handles source names
    whose Turkish characters are stored in NFD on one platform and NFC on
    another.  No recursive/glob lookup is used, which keeps the manifest
    authoritative and prevents an accidental file from another directory from
    being loaded.
    """

    directory = Path(raw_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Ham veri klasörü bulunamadı: {directory}")

    configured_path = Path(configured_filename)
    if configured_path.name != configured_filename or configured_path.is_absolute():
        raise ManifestError(
            "Manifest dosya adı yalnızca doğrudan bir dosya adı içermelidir: "
            f"{configured_filename!r}"
        )

    exact = directory / configured_filename
    if exact.is_file():
        return exact

    comparison_key = _normalized_filename(configured_filename)
    matches = [
        child
        for child in directory.iterdir()
        if child.is_file() and _normalized_filename(child.name) == comparison_key
    ]
    if not matches:
        raise FileNotFoundError(
            f"Manifestte tanımlı kaynak bulunamadı: {configured_filename!r} "
            f"({directory})"
        )
    if len(matches) > 1:
        names = ", ".join(repr(path.name) for path in matches)
        raise ManifestError(
            f"Unicode-normalize dosya adı birden fazla kaynakla eşleşti: {names}"
        )
    return matches[0]


def file_spec(manifest: Mapping[str, Any], logical_name: str) -> Mapping[str, Any]:
    """Return a checked logical file specification."""

    files = manifest.get("files", {})
    if logical_name not in files:
        available = ", ".join(sorted(str(name) for name in files))
        raise ManifestError(
            f"Bilinmeyen mantıksal veri adı {logical_name!r}. Mevcut: {available}"
        )
    return files[logical_name]
