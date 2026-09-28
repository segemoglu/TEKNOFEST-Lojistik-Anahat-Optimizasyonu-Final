"""Synthetic tests for deterministic historical-mean development forecasts."""

from __future__ import annotations

from datetime import date, time
import math
from pathlib import Path
import tempfile
import unittest

from src.domain import DemandInstance
from src.forecasting import (
    ForecastWindow,
    HistoricalMeanForecastProvider,
    write_development_forecast_workbook,
)
from src.io import ExcelProblemLoader, load_manifest

from tests.test_domain import make_static_data


FORECAST_START = date(2026, 6, 29)


def historical_demand(
    demand_id: str,
    demand_date: date,
    slot: time,
    desi: float,
    *,
    origin: str = "A",
    destination: str = "B",
) -> DemandInstance:
    """Create one synthetic historical row without precomputing split flags."""

    return DemandInstance(
        demand_id,
        demand_date,
        slot,
        origin,
        destination,
        desi,
    )


def make_provider(
    history: tuple[DemandInstance, ...],
    *,
    end_date: date = FORECAST_START,
    slots: tuple[time, ...] = (time(9),),
) -> HistoricalMeanForecastProvider:
    """Build the provider with a 100-desi synthetic vehicle capacity."""

    return HistoricalMeanForecastProvider(
        history,
        make_static_data(),
        ForecastWindow(FORECAST_START, end_date, slots),
    )


class HistoricalMeanForecastTests(unittest.TestCase):
    """Verify arithmetic means, fallbacks, leakage and deterministic output."""

    def test_primary_od_weekday_and_time_mean_is_exact(self) -> None:
        """Matching O-D, Monday and 09:00 values 10/30 must average to 20 desi."""

        history = (
            historical_demand("H1", date(2026, 6, 15), time(9), 10.0),
            historical_demand("H2", date(2026, 6, 22), time(9), 30.0),
            historical_demand("H3", date(2026, 6, 23), time(9), 500.0),
        )

        batch = make_provider(history).generate_with_statistics()

        self.assertEqual(batch.demands[0].desi, 20.0)
        self.assertEqual(batch.statistics.method_counts["primary"], 1)

    def test_fallback_order_reaches_od_time_od_overall_and_zero(self) -> None:
        """Absent primary data must use each documented fallback in order."""

        history = (
            historical_demand("H1", date(2026, 6, 23), time(9), 20.0),
            historical_demand("H2", date(2026, 6, 24), time(9), 40.0),
            historical_demand(
                "H3",
                date(2026, 6, 23),
                time(9),
                float("nan"),
                origin="C",
                destination="D",
            ),
        )
        batch = make_provider(history, slots=(time(9), time(17))).generate_with_statistics()
        indexed = {
            (item.origin, item.destination, item.time_slot): item.desi
            for item in batch.demands
        }

        self.assertEqual(indexed[("A", "B", time(9))], 30.0)
        self.assertEqual(indexed[("A", "B", time(17))], 30.0)
        self.assertEqual(indexed[("C", "D", time(9))], 0.0)
        self.assertEqual(batch.statistics.method_counts["od_time"], 1)
        self.assertEqual(batch.statistics.method_counts["od_overall"], 1)
        self.assertEqual(batch.statistics.method_counts["zero"], 2)

    def test_records_on_or_after_forecast_start_are_excluded(self) -> None:
        """A leaked 29 June value must not influence the pre-window mean."""

        history = (
            historical_demand("H1", date(2026, 6, 22), time(9), 10.0),
            historical_demand("LEAK", FORECAST_START, time(9), 1_000.0),
            historical_demand("FUTURE", date(2026, 7, 6), time(9), 2_000.0),
        )

        forecast = make_provider(history).generate()

        self.assertEqual(len(forecast), 1)
        self.assertEqual(forecast[0].desi, 10.0)

    def test_negative_and_nan_history_never_produce_invalid_forecast(self) -> None:
        """Invalid historical values are ignored and zero remains finite/non-negative."""

        history = (
            historical_demand("NEG", date(2026, 6, 22), time(9), -5.0),
            historical_demand("NAN", date(2026, 6, 23), time(9), float("nan")),
        )

        forecast = make_provider(history).generate()

        self.assertTrue(forecast)
        self.assertTrue(all(math.isfinite(item.desi) for item in forecast))
        self.assertTrue(all(item.desi >= 0 for item in forecast))
        self.assertEqual({item.desi for item in forecast}, {0.0})

    def test_ids_and_order_are_identical_across_runs(self) -> None:
        """Date, configured slot order and sorted O-D order must be repeatable."""

        history = (
            historical_demand(
                "H1",
                date(2026, 6, 22),
                time(17),
                10.0,
                origin="B",
                destination="A",
            ),
            historical_demand("H2", date(2026, 6, 22), time(17), 20.0),
        )
        provider = make_provider(
            history,
            end_date=date(2026, 6, 30),
            slots=(time(17), time(9)),
        )

        first = provider.generate()
        second = provider.generate()

        self.assertEqual(first, second)
        self.assertEqual([item.demand_id for item in first], [f"D{i:05d}" for i in range(1, 9)])
        self.assertEqual(
            [(item.demand_date, item.time_slot, item.origin, item.destination) for item in first[:4]],
            [
                (FORECAST_START, time(17), "A", "B"),
                (FORECAST_START, time(17), "B", "A"),
                (FORECAST_START, time(9), "A", "B"),
                (FORECAST_START, time(9), "B", "A"),
            ],
        )

    def test_split_required_uses_maximum_vehicle_capacity(self) -> None:
        """A 150-desi mean exceeds the synthetic 100-desi maximum capacity."""

        history = (
            historical_demand("H1", date(2026, 6, 22), time(9), 150.0),
        )

        forecast = make_provider(history).generate()

        self.assertTrue(forecast[0].split_required)

    def test_excel_round_trip_preserves_demand_instances(self) -> None:
        """Typed dates, times, IDs and full desi must reload to equal records."""

        history = (
            historical_demand("H1", date(2026, 6, 22), time(9), 100.0),
            historical_demand("H2", date(2026, 6, 15), time(9), 120.0),
            historical_demand("H3", date(2026, 6, 8), time(9), 126.0),
        )
        expected = make_provider(history).generate()
        manifest = load_manifest()
        loader = ExcelProblemLoader()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "forecast.xlsx"
            write_development_forecast_workbook(expected, output_path, manifest)
            actual = loader.load_demands_from_workbook(
                output_path,
                static_data=make_static_data(),
            )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
