"""Regression-only checks for known current workbook snapshot dimensions."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.io import ExcelProblemLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DATA_AVAILABLE = (
    (PROJECT_ROOT / "config" / "rules.yaml").is_file()
    and (PROJECT_ROOT / "raw" / "current" / "teknofest26_gelismis.xlsx").is_file()
)


@unittest.skipUnless(
    CURRENT_DATA_AVAILABLE,
    "Mevcut yarışma veri paketi yok; genel testler bundan etkilenmez.",
)
class CurrentDatasetSmokeTests(unittest.TestCase):
    """Regression checks for this repository's current data snapshot only."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the current snapshot once for all expensive smoke checks."""

        cls.loader = ExcelProblemLoader()
        cls.problem = cls.loader.load_problem_data()

    def test_all_manifest_workbooks_resolve(self) -> None:
        """Every logical workbook name must resolve on this data snapshot."""

        resolved = {
            logical_name: self.loader.source_path(logical_name)
            for logical_name in self.loader.manifest["files"]
        }

        self.assertEqual(len(resolved), 8)
        self.assertTrue(all(path.is_file() for path in resolved.values()))

    def test_historical_snapshot_has_66024_rows(self) -> None:
        """Protect the known historical demand row count as a regression."""

        self.assertEqual(len(self.problem.demands), 66_024)

    def test_current_snapshot_marks_848_oversized_demands_for_split(self) -> None:
        """Protect the known oversized-demand count as a regression."""

        self.assertEqual(
            sum(demand.split_required for demand in self.problem.demands),
            848,
        )

    def test_current_static_snapshot_has_seven_zero_truck_quotas(self) -> None:
        """Protect the seven known zero daily Tır quotas in static data."""

        zero_centers = {
            capacity.center
            for capacity in self.problem.static.daily_truck_capacities
            if capacity.max_daily_truck_visits == 0
        }

        self.assertEqual(
            zero_centers,
            {"Bilecik", "Denizli", "Isparta", "Karaman", "Kütahya", "Sivas", "Zonguldak"},
        )


if __name__ == "__main__":
    unittest.main()
