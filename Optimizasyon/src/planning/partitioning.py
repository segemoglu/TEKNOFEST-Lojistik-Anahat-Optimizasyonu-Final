"""Split positive forecast demands into atomic non-Tır-sized cargo parts.

The module works only with domain records.  ``ConservativeDirectBaselineBuilder`` calls it
before vehicle selection; no Excel, forecasting, routing or optimization
dependency is introduced.
"""

from __future__ import annotations

import math

from src.domain import DemandInstance, DemandPart


class DemandPartitionError(ValueError):
    """Explain demand data that cannot form finite positive atomic parts.

    The baseline builder converts this exception to ``BaselineBuildError``.
    Negative/non-finite demand desi and a non-positive maximum part capacity
    are rejected because silently dropping them would invalidate demand
    balance; zero desi is valid and intentionally creates no part.
    """


def partition_demands(
    demands: tuple[DemandInstance, ...],
    maximum_part_desi: float,
    desi_tolerance: float,
) -> tuple[DemandPart, ...]:
    """Return deterministic atomic parts no larger than a desi capacity.

    Args:
        demands: Standard forecast records to process in their supplied order.
        maximum_part_desi: Largest permitted atom size in desi, normally the
            largest non-Tır vehicle capacity from static data.
        desi_tolerance: Non-negative comparison tolerance in desi.  A parent
            may stay in fewer parts when its final atom exceeds nominal vehicle
            capacity by no more than this value.

    Returns:
        Positive ``DemandPart`` records named ``<demand_id>-1``, ``-2`` and so
        on.  Even an unsplit positive demand receives the ``-1`` suffix; a
        zero-desi demand receives no part.

    Connected modules and rules:
        ``ConservativeDirectBaselineBuilder`` supplies demands sorted by ready
        time and uses
        each result in exactly one assignment.  The final part is calculated
        as the original desi minus preceding parts, preserving the parent
        balance without decimal rounding.  Tolerance suppresses only numerical
        dust parts; it never reduces the original demand desi.

    Raises:
        DemandPartitionError: For non-finite/negative desi or tolerance,
            invalid capacity, or a numerical result outside
            ``(0, maximum_part_desi + desi_tolerance]`` desi.
    """

    if not math.isfinite(maximum_part_desi) or maximum_part_desi <= 0:
        raise DemandPartitionError(
            "Atomik parça kapasitesi sonlu ve pozitif desi olmalıdır."
        )
    if not math.isfinite(desi_tolerance) or desi_tolerance < 0:
        raise DemandPartitionError(
            "Desi toleransı sonlu ve negatif olmayan değer olmalıdır."
        )
    parts: list[DemandPart] = []
    for demand in demands:
        parts.extend(
            _partition_one_demand(
                demand,
                maximum_part_desi,
                desi_tolerance,
            )
        )
    return tuple(parts)


def _partition_one_demand(
    demand: DemandInstance,
    maximum_part_desi: float,
    desi_tolerance: float,
) -> tuple[DemandPart, ...]:
    """Split one demand and preserve its exact total after tolerance use."""

    if not math.isfinite(demand.desi) or demand.desi < 0:
        raise DemandPartitionError(
            f"Talep desisi sonlu ve negatif olmayan değer olmalıdır: "
            f"{demand.demand_id!r}={demand.desi!r} desi."
        )
    if demand.desi == 0:
        return ()
    # Subtracting the allowed tolerance before ceil prevents a microscopic
    # floating-point excess from creating a separate physical vehicle.  The
    # original demand value remains in the final part and is never rounded.
    tolerance_adjusted_desi = max(demand.desi - desi_tolerance, 0.0)
    part_count = max(
        1,
        math.ceil(tolerance_adjusted_desi / maximum_part_desi),
    )
    values = [maximum_part_desi] * max(part_count - 1, 0)
    values.append(demand.desi - math.fsum(values))
    if any(
        value <= 0 or value > maximum_part_desi + desi_tolerance
        for value in values
    ):
        raise DemandPartitionError(
            f"Talep güvenli atomik parçalara ayrılamadı: {demand.demand_id!r}."
        )
    return tuple(
        DemandPart(
            part_id=f"{demand.demand_id}-{index}",
            demand_id=demand.demand_id,
            desi=value,
        )
        for index, value in enumerate(values, start=1)
    )


__all__ = ["DemandPartitionError", "partition_demands"]
