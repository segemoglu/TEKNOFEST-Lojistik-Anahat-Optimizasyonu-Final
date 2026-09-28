"""Public development-forecast API for the linehaul project.

Callers may depend on ``ForecastProvider`` and standard ``DemandInstance``
tuples.  The historical-mean implementation is a temporary development aid,
not the official competition forecast and not an optimizer dependency.
"""

from .providers import (
    ForecastBatch,
    ForecastMethod,
    ForecastProvider,
    ForecastStatistics,
    ForecastWindow,
    HistoricalMeanForecastProvider,
    forecast_window_from_manifest,
)
from .workbook import write_development_forecast_workbook

__all__ = [
    "ForecastBatch",
    "ForecastMethod",
    "ForecastProvider",
    "ForecastStatistics",
    "ForecastWindow",
    "HistoricalMeanForecastProvider",
    "forecast_window_from_manifest",
    "write_development_forecast_workbook",
]
