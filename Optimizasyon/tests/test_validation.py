"""Dataset-independent schema, type, key, date and network validation tests."""

from __future__ import annotations

from datetime import date, time
import unittest

import pandas as pd

from src.io.validation import (
    parse_date_column,
    parse_date_value,
    parse_numeric_column,
    parse_time_column,
    parse_time_value,
    validate_complete_directed_network,
    validate_duplicates,
    validate_known_centers,
    validate_required_columns,
)


class GeneralValidationTests(unittest.TestCase):
    """Verify dataset-independent schema validation helpers."""

    def test_required_column_and_duplicate_record_are_reported(self) -> None:
        """Missing headers and duplicate business keys must be visible."""

        frame = pd.DataFrame({"id": ["D1", "D1"]})

        missing = validate_required_columns(frame, ["id", "desi"], dataset="demand")
        duplicates = validate_duplicates(frame, ["id"], dataset="demand")

        self.assertEqual([issue.code for issue in missing], ["missing_column"])
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(issue.code == "duplicate_record" for issue in duplicates))

    def test_invalid_type_and_negative_number_are_reported(self) -> None:
        """Numeric parsing must distinguish sign and type failures."""

        series = pd.Series([10, -1, "bozuk", True])

        _, issues = parse_numeric_column(series, dataset="demand", field="desi")

        self.assertEqual({issue.code for issue in issues}, {"negative_value", "invalid_type"})
        self.assertEqual(sum(issue.code == "invalid_type" for issue in issues), 2)

    def test_unknown_center_and_missing_directed_arc_are_reported(self) -> None:
        """Cross-table centre and network completeness checks must run."""

        demands = pd.DataFrame({"origin": ["A", "X"], "destination": ["B", "A"]})
        network = pd.DataFrame(
            {
                "origin": ["A"],
                "destination": ["B"],
                "distance_km": [1.0],
                "travel_hours_truck": [1.0],
            }
        )

        unknown = validate_known_centers(
            demands, ["origin", "destination"], {"A", "B"}, dataset="demand"
        )
        missing = validate_complete_directed_network(network, {"A", "B"})

        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0].code, "unknown_center")
        self.assertEqual([(issue.code, issue.value) for issue in missing], [("missing_arc", ("B", "A"))])

    def test_date_and_time_parsers_accept_excel_and_text_forms(self) -> None:
        """Supported Excel/text date-time representations must normalize."""

        self.assertEqual(parse_date_value("29.06.2026"), date(2026, 6, 29))
        self.assertEqual(parse_time_value("09:00"), time(9))
        self.assertEqual(parse_time_value(17 / 24), time(17))
        self.assertIsNone(parse_date_value("31.02.2026"))
        self.assertIsNone(parse_time_value("25:00"))

    def test_missing_distance_and_invalid_date_time_are_reported(self) -> None:
        """Missing travel values and malformed dates/times must be issues."""

        _, numeric_issues = parse_numeric_column(
            pd.Series([None]), dataset="network", field="travel_hours_truck"
        )
        _, date_issues = parse_date_column(
            pd.Series(["31.02.2026"]), dataset="demand", field="demand_date"
        )
        _, time_issues = parse_time_column(
            pd.Series(["25:00"]), dataset="demand", field="time_slot"
        )

        self.assertEqual([issue.code for issue in numeric_issues], ["missing_value"])
        self.assertEqual([issue.code for issue in date_issues], ["invalid_date"])
        self.assertEqual([issue.code for issue in time_issues], ["invalid_time"])


if __name__ == "__main__":
    unittest.main()
