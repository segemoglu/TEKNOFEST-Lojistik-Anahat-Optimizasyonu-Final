"""Reserve calendar-day handling desi without creating dock queues.

``ConservativeDirectBaselineBuilder`` uses this ledger for origin loading and
destination unloading.  Capacity is one shared centre/day desi pool, while
operation timestamps may overlap across vehicles because the competition has
no simultaneous ramp constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math

from src.domain import HandlingCapacity


MINUTES_PER_DAY = 24.0 * 60.0


class HandlingCapacityError(ValueError):
    """Explain why a handling reservation cannot fit on any calendar day.

    Unknown centres, non-finite/negative desi, insufficient daily capacity and
    operations lasting more than one day are structural baseline failures.
    ``ConservativeDirectBaselineBuilder`` exposes them as
    ``BaselineBuildError`` rather than
    looping indefinitely or exceeding capacity silently.
    """


@dataclass(frozen=True, slots=True)
class HandlingUsage:
    """One centre/day ledger snapshot measured in desi.

    ``BaselineBuildResult`` exposes these immutable records to the CLI and
    tests.  ``used_desi / capacity_desi`` is the daily utilization ratio;
    zero-capacity days never appear because they cannot accept an operation.
    """

    center: str
    operation_date: date
    used_desi: float
    capacity_desi: float

    @property
    def utilization_ratio(self) -> float:
        """Return used/capacity desi, or infinity for invalid zero capacity."""

        if self.capacity_desi <= 0:
            return math.inf if self.used_desi > 0 else 0.0
        return self.used_desi / self.capacity_desi


class HandlingCapacityLedger:
    """Track and reserve shared daily handling capacity by centre.

    Args:
        capacities: Static centre capacities in desi per calendar day.
        minutes_per_desi: Configured operation duration factor in minutes/desi.
        desi_tolerance: Comparison tolerance in desi.

    Connected modules and rules:
        The baseline builder calls :meth:`schedule` independently for each
        load/unload.  A reservation stays within one calendar day and moves to
        the earliest later day if either duration or remaining desi capacity
        does not fit.  Parallel operations remain legal; no dock timeline is
        stored.  Mandatory-filling strategy calls :meth:`try_schedule_before`
        when moving a load to tomorrow would invalidate its fixed service day.
        Invalid inputs raise ``HandlingCapacityError``.
    """

    def __init__(
        self,
        capacities: tuple[HandlingCapacity, ...],
        minutes_per_desi: float,
        desi_tolerance: float,
    ) -> None:
        """Index validated capacities and initialize an empty usage ledger."""

        self._capacities = {
            item.center: item.max_daily_desi for item in capacities
        }
        self._minutes_per_desi = minutes_per_desi
        self._tolerance = desi_tolerance
        self._used: dict[tuple[str, date], float] = {}

    def schedule(
        self,
        center: str,
        earliest_at: datetime,
        desi: float,
    ) -> tuple[datetime, datetime]:
        """Reserve the earliest single-day operation and return its interval.

        Args:
            center: Transfer centre consuming the common daily pool.
            earliest_at: Earliest legal local start timestamp.
            desi: Load or unload quantity in desi.

        Returns:
            ``(start_at, end_at)`` with duration
            ``desi × minutes_per_desi`` minutes.  The interval never crosses
            midnight and its full desi is charged to ``start_at.date()``.

        Raises:
            HandlingCapacityError: If the centre/input is invalid or the
                operation can never fit in one calendar day.
        """

        capacity = self._capacity_for(center)
        duration = self._duration_for(desi)
        if desi > capacity + self._tolerance:
            raise HandlingCapacityError(
                f"{center!r} günlük kapasitesi {desi} desilik işlemi karşılamıyor; "
                f"kapasite={capacity} desi."
            )
        candidate = earliest_at
        while True:
            start_at, end_at = self._same_day_interval(candidate, duration)
            key = (center, start_at.date())
            used = self._used.get(key, 0.0)
            if used + desi <= capacity + self._tolerance:
                self._used[key] = used + desi
                return start_at, end_at
            candidate = datetime.combine(
                start_at.date() + timedelta(days=1),
                time.min,
            )

    def snapshot(self) -> tuple[HandlingUsage, ...]:
        """Return deterministic immutable centre/day usage records in desi."""

        return tuple(
            HandlingUsage(center, operation_date, used, self._capacities[center])
            for (center, operation_date), used in sorted(self._used.items())
        )

    def fork(self) -> "HandlingCapacityLedger":
        """Return an independent copy for transactional schedule trials.

        Returns:
            A ledger with identical centre capacities, minutes/desi duration,
            desi tolerance and current centre/day usage.  Later reservations
            on either instance do not affect the other.

        Connected modules and rules:
            Multi-stop planning evaluates alternative stop orders on forks so
            a rejected load or unload sequence cannot consume real handling
            capacity.  :meth:`commit_from` applies only the selected trial.
            Invalid master data would already have failed construction, so
            this method performs no new scheduling or file access.
        """

        clone = HandlingCapacityLedger(
            tuple(
                HandlingCapacity(center, capacity)
                for center, capacity in self._capacities.items()
            ),
            self._minutes_per_desi,
            self._tolerance,
        )
        clone._used = dict(self._used)
        return clone

    def commit_from(self, trial: "HandlingCapacityLedger") -> None:
        """Atomically replace usage with a compatible successful trial.

        Args:
            trial: Fork whose complete centre/day desi reservations should
                become the live state.

        Connected modules and errors:
            The two-stop greedy builder calls this only after real cost and
            feasibility checks accept a candidate.  A ledger with different
            capacities, minutes/desi or tolerance raises
            ``HandlingCapacityError``; the receiving state remains unchanged.
        """

        compatible = (
            self._capacities == trial._capacities
            and self._minutes_per_desi == trial._minutes_per_desi
            and self._tolerance == trial._tolerance
        )
        if not compatible:
            raise HandlingCapacityError(
                "Farklı kapasite veya süre politikasındaki ledger commit edilemez."
            )
        self._used = dict(trial._used)

    def try_schedule_before(
        self,
        center: str,
        earliest_at: datetime,
        desi: float,
        latest_end_at: datetime,
    ) -> tuple[datetime, datetime] | None:
        """Atomically reserve one operation only if it ends before a deadline.

        Args:
            center: Transfer centre consuming the shared daily desi pool.
            earliest_at: Earliest legal loading timestamp on the service day.
            desi: Positive grouped quantity in desi.
            latest_end_at: Exclusive local deadline for operation completion.

        Returns:
            ``(start_at, end_at)`` when the operation fits remaining capacity,
            stays on ``earliest_at``'s calendar day and satisfies
            ``end_at < latest_end_at``.  Returns ``None`` otherwise.

        Connected modules and failure behaviour:
            ``GreedyMandatoryRentalFilledBuilder`` uses the exclusive deadline
            to preserve same-day departure and, for Tır, same-day arrival.  A
            ``None`` result never mutates ledger usage, so rejected cargo can
            safely return to the spot pool.  Unknown/invalid master values
            still raise ``HandlingCapacityError`` rather than being hidden.
        """

        capacity = self._capacity_for(center)
        duration = self._duration_for(desi)
        end_at = earliest_at + duration
        key = (center, earliest_at.date())
        used = self._used.get(key, 0.0)
        fits_time = (
            end_at.date() == earliest_at.date()
            and end_at < latest_end_at
        )
        fits_capacity = used + desi <= capacity + self._tolerance
        if not fits_time or not fits_capacity:
            return None
        self._used[key] = used + desi
        return earliest_at, end_at

    def record_completed(
        self,
        center: str,
        start_at: datetime,
        end_at: datetime,
        desi: float,
    ) -> None:
        """Charge an already-scheduled operation to the days it really spans.

        Args:
            center: Transfer centre whose daily pool absorbs the work.
            start_at: Local start timestamp of the finished operation.
            end_at: Local end timestamp; may fall on a later calendar day.
            desi: Quantity handled across the whole interval.

        Connected modules and rules:
            ``ALNSMaterializer`` calls this for the fixed mandatory operations
            it did not schedule itself.  Those operations exist whether or not
            capacity remains, so this method **records** rather than reserves:
            it never refuses and never moves the work to another day.  Use
            :meth:`schedule` or :meth:`try_schedule_before` to place new work.

            An interval crossing midnight is split in proportion to the time
            spent on each calendar day, which is the competition's stated rule
            (Q&A: 10.000 desi started 23:30 charges 3.000 desi to that day and
            7.000 desi to the next).  Today no mandatory operation crosses
            midnight, so this is currently equivalent to charging the start day
            — it keeps the ledger correct if that ever changes.

        Raises:
            HandlingCapacityError: For an unknown centre or a non-finite or
                negative desi value.
        """

        self._capacity_for(center)
        if not math.isfinite(desi) or desi < 0:
            raise HandlingCapacityError(
                f"Kaydedilen elleçleme desisi sonlu ve negatif olmamalıdır: {desi!r}."
            )
        total_seconds = (end_at - start_at).total_seconds()
        if total_seconds <= 0:
            self._charge(center, start_at.date(), desi)
            return
        cursor = start_at
        while cursor < end_at:
            next_midnight = datetime.combine(
                cursor.date() + timedelta(days=1),
                time.min,
            )
            segment_end = min(next_midnight, end_at)
            share = (segment_end - cursor).total_seconds() / total_seconds
            self._charge(center, cursor.date(), desi * share)
            cursor = segment_end

    def _charge(self, center: str, operation_date: date, desi: float) -> None:
        """Add desi to one centre/day cell without any capacity check."""

        key = (center, operation_date)
        self._used[key] = self._used.get(key, 0.0) + desi

    def _capacity_for(self, center: str) -> float:
        """Resolve a finite non-negative daily centre capacity in desi."""

        if center not in self._capacities:
            raise HandlingCapacityError(
                f"Elleçleme kapasitesi bulunmayan merkez: {center!r}."
            )
        capacity = self._capacities[center]
        if not math.isfinite(capacity) or capacity < 0:
            raise HandlingCapacityError(
                f"Geçersiz günlük elleçleme kapasitesi: {center!r}={capacity!r}."
            )
        return capacity

    def _duration_for(self, desi: float) -> timedelta:
        """Convert a finite positive desi reservation to configured minutes."""

        if not math.isfinite(desi) or desi <= 0:
            raise HandlingCapacityError(
                f"Elleçleme desisi sonlu ve pozitif olmalıdır: {desi!r}."
            )
        duration_minutes = desi * self._minutes_per_desi
        if not math.isfinite(duration_minutes) or duration_minutes >= MINUTES_PER_DAY:
            raise HandlingCapacityError(
                "Tek elleçleme işlemi aynı takvim gününde tamamlanamıyor: "
                f"{duration_minutes!r} dakika."
            )
        return timedelta(minutes=duration_minutes)

    @staticmethod
    def _same_day_interval(
        earliest_at: datetime,
        duration: timedelta,
    ) -> tuple[datetime, datetime]:
        """Move a midnight-crossing candidate to the next day's 00:00 time."""

        next_midnight = datetime.combine(
            earliest_at.date() + timedelta(days=1),
            time.min,
        )
        if earliest_at + duration <= next_midnight:
            return earliest_at, earliest_at + duration
        start_at = next_midnight
        return start_at, start_at + duration


__all__ = [
    "HandlingCapacityError",
    "HandlingCapacityLedger",
    "HandlingUsage",
]
