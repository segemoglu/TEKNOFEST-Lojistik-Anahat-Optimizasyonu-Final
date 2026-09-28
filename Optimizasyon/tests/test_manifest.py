"""Manifest path-safety, Unicode resolution and fixed-policy config checks."""

from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
import unicodedata
import unittest

from src.io.manifest import (
    ManifestError,
    configured_project_path,
    load_manifest,
    resolve_source_file,
)
from src.evaluation import EvaluationPolicy, load_evaluation_policy


class ManifestTests(unittest.TestCase):
    """Verify central manifest safety and Unicode filename resolution."""

    def test_default_manifest_is_central_and_covers_all_workbooks(self) -> None:
        """The default manifest must enumerate every source workbook."""

        manifest = load_manifest()

        self.assertEqual(manifest["paths"]["raw_data_dir"], "raw/current")
        self.assertEqual(
            manifest["paths"]["development_forecast"],
            "data/interim/development_forecast.xlsx",
        )
        self.assertEqual(len(manifest["files"]), 8)
        self.assertTrue(manifest["manifest_conventions"]["source_workbooks_are_read_only"])

    def test_unicode_decomposed_filename_resolves_from_composed_manifest_name(self) -> None:
        """NFC manifest names must match NFD physical filenames."""

        with tempfile.TemporaryDirectory() as temp_dir:
            physical_name = unicodedata.normalize("NFD", "Araç.xlsx")
            physical_path = Path(temp_dir) / physical_name
            physical_path.touch()

            resolved = resolve_source_file(temp_dir, "Araç.xlsx")

            self.assertEqual(resolved, physical_path.resolve())

    def test_manifest_filename_cannot_escape_raw_directory(self) -> None:
        """Path traversal in a manifest filename must be rejected."""

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ManifestError):
                resolve_source_file(temp_dir, "../outside.xlsx")

    def test_configured_output_path_cannot_escape_project_root(self) -> None:
        """A forecast output path outside the selected repo must be rejected."""

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = {"paths": {"development_forecast": "../outside.xlsx"}}

            with self.assertRaises(ManifestError):
                configured_project_path(
                    manifest,
                    "development_forecast",
                    project_root=temp_dir,
                )

    def test_evaluation_assumptions_are_loaded_from_central_config(self) -> None:
        """Cost and validator assumptions must not remain hidden constants."""

        policy = load_evaluation_policy()

        self.assertEqual(policy.hourly_cost_granularity, "vehicle")
        self.assertEqual(policy.continuous_visit_midnight_policy, "arrival_day")
        self.assertEqual(policy.desi_tolerance, 1e-6)

    def test_fixed_rule_cannot_be_silently_reversed_in_config(self) -> None:
        """A NoOverlap config edit must fail instead of changing a fixed rule."""

        manifest = deepcopy(load_manifest())
        manifest["rules"]["handling"]["no_overlap_constraint_enabled"] = True

        with self.assertRaises(ValueError):
            EvaluationPolicy.from_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
