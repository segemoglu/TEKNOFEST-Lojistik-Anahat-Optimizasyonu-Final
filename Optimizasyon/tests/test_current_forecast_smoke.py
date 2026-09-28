"""Regression checks for the current 2026 historical-demand forecast snapshot."""

from __future__ import annotations

from collections import Counter
from datetime import date, time
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.forecasting import (
    HistoricalMeanForecastProvider,
    forecast_window_from_manifest,
    write_development_forecast_workbook,
)
from src.io import ExcelProblemLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "raw" / "current"
CURRENT_DATA_AVAILABLE = (
    (PROJECT_ROOT / "config" / "rules.yaml").is_file()
    and (RAW_DIRECTORY / "teknofest26_gelismis.xlsx").is_file()
)
EXPECTED_FORECAST_DATES = {
    date(2026, 6, 29),
    date(2026, 6, 30),
    date(2026, 7, 1),
    date(2026, 7, 2),
    date(2026, 7, 3),
    date(2026, 7, 4),
    date(2026, 7, 5),
}
EXPECTED_TIME_SLOTS = {time(9), time(17)}


def _raw_hashes() -> dict[str, str]:
    """Return SHA-256 hashes for every immutable raw source file."""

    hashes: dict[str, str] = {}
    for path in sorted(RAW_DIRECTORY.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


@unittest.skipUnless(
    CURRENT_DATA_AVAILABLE,
    "Mevcut yarışma veri paketi yok; genel forecast testleri etkilenmez.",
)
class CurrentForecastSmokeTests(unittest.TestCase):
    """Protect only known current-data dimensions, never surprise-data sizes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load and forecast the current snapshot once for all smoke checks."""

        cls.loader = ExcelProblemLoader()
        cls.problem = cls.loader.load_problem_data()
        cls.window = forecast_window_from_manifest(cls.loader.manifest)
        cls.batch = HistoricalMeanForecastProvider(
            cls.problem.demands,
            cls.problem.static,
            cls.window,
        ).generate_with_statistics()
        cls.historical_routes = {
            (item.origin, item.destination)
            for item in cls.problem.demands
            if item.demand_date < cls.window.start_date
        }

    def test_current_history_has_289_active_routes(self) -> None:
        """The current pre-window snapshot must expose 289 observed O-D lines."""

        self.assertEqual(len(self.historical_routes), 289)
        self.assertEqual(self.batch.statistics.active_route_count, 289)

    def test_current_forecast_has_4046_records(self) -> None:
        """289 active lines times seven dates times two slots must equal 4,046."""

        self.assertEqual(len(self.batch.demands), 4_046)
        self.assertEqual(self.batch.statistics.record_count, 4_046)

    def test_each_active_route_has_exactly_fourteen_records(self) -> None:
        """Every observed O-D must have seven dates times two ordered slots."""

        counts = Counter(
            (item.origin, item.destination) for item in self.batch.demands
        )

        self.assertEqual(set(counts), self.historical_routes)
        self.assertEqual(set(counts.values()), {14})

    def test_forecast_dates_and_slots_match_the_window(self) -> None:
        """Current output must contain only 29 June-5 July and 09:00/17:00."""

        self.assertEqual(
            {item.demand_date for item in self.batch.demands},
            EXPECTED_FORECAST_DATES,
        )
        self.assertEqual(
            {item.time_slot for item in self.batch.demands},
            EXPECTED_TIME_SLOTS,
        )

    def test_no_unseen_static_network_route_is_generated(self) -> None:
        """Forecast routes must equal history, not all static-network arcs."""

        forecast_routes = {
            (item.origin, item.destination) for item in self.batch.demands
        }

        self.assertEqual(forecast_routes, self.historical_routes)

    def test_forecast_demand_ids_are_unique(self) -> None:
        """Sequential D-prefixed identifiers must not collide in 4,046 rows."""

        identifiers = [item.demand_id for item in self.batch.demands]

        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(identifiers[0], "D00001")
        self.assertEqual(identifiers[-1], "D04046")

    def test_forecast_write_does_not_change_any_raw_hash(self) -> None:
        """Writing an interim workbook must leave every raw Excel/PDF byte intact."""

        before = _raw_hashes()
        with tempfile.TemporaryDirectory() as temp_dir:
            write_development_forecast_workbook(
                self.batch.demands,
                Path(temp_dir) / "development_forecast.xlsx",
                self.loader.manifest,
            )
        after = _raw_hashes()

        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
