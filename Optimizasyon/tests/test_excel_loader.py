"""Synthetic read-only Excel loader and cross-table validation checks."""

from __future__ import annotations

from datetime import date, time
import unittest
from unittest.mock import patch

import pandas as pd

from src.io.excel_loader import ExcelProblemLoader
from src.io.validation import DataValidationError

from tests.test_domain import make_static_data


def make_in_memory_loader() -> ExcelProblemLoader:
    """Construct a loader shell whose Excel read is mocked by each test."""

    loader = object.__new__(ExcelProblemLoader)
    loader.manifest = {
        "files": {
            "demand": {
                "role": "demand",
                "filename": "unused.xlsx",
                "sheet": "Sheet1",
                "columns": {
                    "demand_id": "demand_id",
                    "demand_date": "demand_date",
                    "time_slot": "time_slot",
                    "origin": "origin",
                    "destination": "destination",
                    "desi": "desi",
                },
            }
        },
        "validation": {
            "business_keys": {"demand": ["demand_id"]},
            "allowed_demand_time_slots": ["09:00", "17:00"],
        },
    }
    return loader


class ExcelLoaderTests(unittest.TestCase):
    """Verify demand loading rules without writing temporary workbooks."""

    def test_oversized_demand_is_marked_for_split_instead_of_rejected(self) -> None:
        """A demand over vehicle capacity must request splitting, not fail."""

        loader = make_in_memory_loader()
        frame = pd.DataFrame(
            {
                "demand_id": ["D1"],
                "demand_date": [date(2026, 6, 29)],
                "time_slot": [time(9)],
                "origin": ["A"],
                "destination": ["B"],
                "desi": [150.0],
            }
        )

        with patch.object(loader, "read_canonical_frame", return_value=(frame, [])):
            demands = loader.load_demands("demand", static_data=make_static_data())

        self.assertEqual(len(demands), 1)
        self.assertTrue(demands[0].split_required)

    def test_unknown_center_is_a_validation_error(self) -> None:
        """Demand centres absent from static data must be rejected."""

        loader = make_in_memory_loader()
        frame = pd.DataFrame(
            {
                "demand_id": ["D1"],
                "demand_date": [date(2026, 6, 29)],
                "time_slot": [time(9)],
                "origin": ["UNKNOWN"],
                "destination": ["B"],
                "desi": [10.0],
            }
        )

        with patch.object(loader, "read_canonical_frame", return_value=(frame, [])):
            with self.assertRaises(DataValidationError) as context:
                loader.load_demands("demand", static_data=make_static_data())

        self.assertIn("unknown_center", {issue.code for issue in context.exception.issues})


if __name__ == "__main__":
    unittest.main()
