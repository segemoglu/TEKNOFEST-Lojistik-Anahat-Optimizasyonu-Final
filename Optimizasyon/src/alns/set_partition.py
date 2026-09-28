"""Recombine the routes ALNS built into the cheapest exact cover of all cargo.

The search judges whole *states*: a route that would be excellent in some other
combination is discarded because the state containing it was not an improvement.
This module keeps every distinct feasible route the search produced and asks
CP-SAT the global question instead — *which subset of these routes carries every
atom exactly once for the least money?*

The incumbent's own routes are always in the pool, so the model can never be
infeasible and its answer is never worse than the plan ALNS finished with.  The
selected combination is still materialized, validated and costed by the
authoritative pipeline before anything is exported; the objective here uses proxy
route costs, which are optimistic about handling deferral, so only the exact
audit decides whether the result is really cheaper.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import logging
import math
import time

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.alns.evaluator import ProxyEvaluator
from src.alns.state import ALNSSolutionState, ALNSSpotRoute

logger = logging.getLogger(__name__)

# CP-SAT works in integers; TL costs are scaled by this factor so that kuruş
# differences still order two candidate combinations correctly.
COST_SCALE = 100


class RouteSelectionUnavailable(RuntimeError):
    """Signal that no better exact cover could be produced."""


def optimize_route_selection(
    pool: dict[tuple, tuple[ALNSSpotRoute, float]],
    incumbent: ALNSSolutionState,
    problem: ProblemData,
    policy: EvaluationPolicy,
    proxy: ProxyEvaluator,
    time_limit_seconds: float = 300.0,
    workers: int = 4,
    candidate_count: int = 1,
) -> tuple[ALNSSolutionState, ...]:
    """Return candidate states whose routes exactly cover all spot cargo.

    Args:
        pool: Distinct candidate routes keyed canonically, with proxy costs.
        incumbent: The state ALNS finished with; its routes seed the hint and
            guarantee feasibility.
        problem: Static data supplying capacities and quotas.
        policy: Shared tolerance and Tır-type policy.
        proxy: Evaluator reused for per-stop day attribution.
        time_limit_seconds: Wall-clock ceiling shared by every solve.
        workers: Search worker count.
        candidate_count: How many distinct covers to return.  Measured on a
            4.092-route pool, twelve covers shared the **same** proxy objective
            while their real costs spread over 335.000 TL: the proxy cannot rank
            alternative optima, but the exact audit can, so the cheap move is to
            hand it many of them rather than trying to model the optimism away.

    Returns:
        ``ALNSSolutionState`` values in increasing proxy cost.  Frozen mandatory
        trips and atoms are carried over untouched.

    Raises:
        RouteSelectionUnavailable: If OR-Tools is missing, the model proves
            infeasible, or the solver returns nothing usable.  Callers keep their
            existing plan in that case.
    """

    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RouteSelectionUnavailable("OR-Tools bulunamadı.") from exc

    mandatory_part_ids = {
        assignment.demand_part_id
        for trip in incumbent.mandatory_trips
        for leg in trip.legs
        for assignment in leg.assignments
    }
    required = tuple(
        part.part_id
        for part in incumbent.parts
        if part.part_id not in mandatory_part_ids
    )
    if not required:
        raise RouteSelectionUnavailable("Spot tarafında kapsanacak parça yok.")

    routes = _usable_routes(pool, frozenset(required))
    if not routes:
        raise RouteSelectionUnavailable("Havuzda kullanılabilir rota yok.")

    incumbent_keys = _incumbent_keys(incumbent)
    model = cp_model.CpModel()
    chosen = [model.NewBoolVar(f"r{index}") for index in range(len(routes))]

    covering: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(routes):
        for part_id in entry.part_ids:
            covering[part_id].append(index)
    for part_id in required:
        options = covering.get(part_id)
        if not options:
            raise RouteSelectionUnavailable(
                f"{part_id!r} hiçbir havuz rotasında taşınmıyor."
            )
        model.AddExactlyOne(chosen[index] for index in options)

    _add_truck_quota(model, chosen, routes, problem, policy, proxy)
    _add_handling_headroom(model, chosen, routes, incumbent, problem, proxy)

    model.Minimize(
        sum(
            int(round(entry.cost_tl * COST_SCALE)) * chosen[index]
            for index, entry in enumerate(routes)
        )
    )
    for index, entry in enumerate(routes):
        model.AddHint(chosen[index], 1 if entry.key in incumbent_keys else 0)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(workers)
    # Determinism: the same pool and hint must always produce the same plans.
    solver.parameters.random_seed = 0

    states: list[ALNSSolutionState] = []
    budget = float(time_limit_seconds)
    for round_index in range(max(candidate_count, 1)):
        if budget <= 0:
            break
        solver.parameters.max_time_in_seconds = budget
        started = time.monotonic()
        status = solver.Solve(model)
        budget -= time.monotonic() - started
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if not states:
                raise RouteSelectionUnavailable(
                    f"Çözücü kullanılabilir sonuç vermedi: "
                    f"{solver.StatusName(status)}"
                )
            break
        picked = [index for index in range(len(routes)) if solver.Value(chosen[index])]
        logger.info(
            "Set-partitioning aday %d: %d rotadan %d seçildi, %s, proxy %.2f TL",
            round_index + 1,
            len(routes),
            len(picked),
            solver.StatusName(status),
            solver.ObjectiveValue() / COST_SCALE,
        )
        states.append(
            ALNSSolutionState(
                incumbent.parts,
                incumbent.mandatory_trips,
                tuple(routes[index].route for index in picked),
            )
        )
        # No-good cut: at least one of these routes must be dropped next time.
        # Forbidding only the picked set is enough here — any superset of an
        # exact cover is infeasible anyway — and keeps the clause short.
        model.AddBoolOr([chosen[index].Not() for index in picked])
    return tuple(states)


class _RouteEntry:
    """One pool route with the facts the model needs, computed once."""

    __slots__ = ("key", "route", "cost_tl", "part_ids", "stop_days")

    def __init__(
        self,
        key: tuple,
        route: ALNSSpotRoute,
        cost_tl: float,
        part_ids: frozenset[str],
        stop_days: tuple[date, ...],
    ) -> None:
        self.key = key
        self.route = route
        self.cost_tl = cost_tl
        self.part_ids = part_ids
        self.stop_days = stop_days


def _usable_routes(
    pool: dict[tuple, tuple[ALNSSpotRoute, float]],
    required: frozenset[str],
) -> tuple[_RouteEntry, ...]:
    """Keep pool routes that only carry cargo the spot side still owes.

    A route picking up an atom already assigned to a mandatory vehicle would
    double-deliver it, so it is dropped rather than repaired.
    """

    entries: list[_RouteEntry] = []
    for key, (route, cost_tl) in pool.items():
        picked = frozenset(
            part.part_id for stop in route.stops for part in stop.pickups
        )
        if not picked or not picked <= required:
            continue
        if not math.isfinite(cost_tl):
            continue
        entries.append(_RouteEntry(key, route, cost_tl, picked, ()))
    return tuple(entries)


def _incumbent_keys(state: ALNSSolutionState) -> frozenset[tuple]:
    """Return the canonical keys of the routes the search finished with."""

    return frozenset(
        (
            route.service_date,
            route.vehicle_type,
            tuple(stop.stop_key for stop in route.stops),
        )
        for route in state.spot_routes
    )


def _add_truck_quota(
    model,
    chosen: list,
    routes: tuple[_RouteEntry, ...],
    problem: ProblemData,
    policy: EvaluationPolicy,
    proxy: ProxyEvaluator,
) -> None:
    """Limit Tır visits per centre and day, mandatory schedule included first.

    Each stop of a Tır route is one visit, matching the validator.  Stop days
    come from :meth:`ProxyEvaluator.stop_dates`, so a route spanning midnight is
    charged to the days it really touches.
    """

    quotas = {
        item.center: item.max_daily_truck_visits
        for item in problem.static.daily_truck_capacities
    }
    demand_terms: dict[tuple[str, date], list] = defaultdict(list)
    for index, entry in enumerate(routes):
        if entry.route.vehicle_type != policy.truck_vehicle_type:
            continue
        days = proxy.stop_dates(
            entry.route.service_date,
            entry.route.stops,
            entry.route.vehicle_type,
        )
        for stop, day in zip(entry.route.stops, days, strict=True):
            demand_terms[(stop.center, day)].append(chosen[index])
    for (center, day), terms in demand_terms.items():
        quota = quotas.get(center, 0)
        used = proxy.mandatory_truck_usage.get((center, day), 0)
        model.Add(sum(terms) <= max(quota - used, 0))


def _add_handling_headroom(
    model,
    chosen: list,
    routes: tuple[_RouteEntry, ...],
    incumbent: ALNSSolutionState,
    problem: ProblemData,
    proxy: ProxyEvaluator,
) -> None:
    """Stop the solver piling handling onto days that are already full.

    Why this is needed at all:
        Route costs are priced in isolation.  A combination that looks cheapest
        per route can overload a few centre-days; materialization then defers
        those operations to the next day, the vehicles wait, and the billed hours
        grow.  Measured on the first working probe: the solver's proxy objective
        was 13.850.162 TL while the same plan really cost 14.638.750 TL — a 5,7%
        optimism that wiped out the whole gain.

    Why it cannot repeat the earlier INFEASIBLE:
        The first attempt bounded each centre-day by its nominal capacity and
        attributed every stop to the route's ``service_date``.  That rejected the
        incumbent's own routes (the proxy skips this check for
        the same reason).  Here the day comes from
        :meth:`ProxyEvaluator.stop_dates`, and the bound is the **larger** of the
        remaining nominal capacity and what the incumbent itself already puts on
        that day.  The incumbent therefore always satisfies every constraint, so
        the model stays feasible by construction and the bound is data-derived
        rather than a tuned margin.
    """

    capacities = {
        item.center: item.max_daily_desi
        for item in problem.static.handling_capacities
    }
    incumbent_keys = _incumbent_keys(incumbent)
    incumbent_indices = {
        index for index, entry in enumerate(routes) if entry.key in incumbent_keys
    }

    # Integer weights and the bound must come from the *same* arithmetic.  Taking
    # the bound from a float sum while rounding each stop separately let the
    # incumbent break its own constraint (three 0,6 desi stops weigh 3 but sum to
    # 1), which is what made the first headroom attempt INFEASIBLE.
    terms: dict[tuple[str, date], list[tuple[int, int]]] = defaultdict(list)
    incumbent_load: dict[tuple[str, date], int] = defaultdict(int)
    for index, entry in enumerate(routes):
        days = proxy.stop_dates(
            entry.route.service_date,
            entry.route.stops,
            entry.route.vehicle_type,
        )
        for stop, day in zip(entry.route.stops, days, strict=True):
            desi = stop.dropoff_desi + stop.pickup_desi
            if desi <= 0:
                continue
            weight = int(math.ceil(desi))
            key = (stop.center, day)
            terms[key].append((weight, index))
            if index in incumbent_indices:
                incumbent_load[key] += weight

    for key, items in terms.items():
        center, _ = key
        capacity = capacities.get(center)
        if capacity is None:
            continue
        nominal = capacity - proxy.mandatory_handling_usage.get(key, 0.0)
        bound = max(int(math.floor(nominal)), incumbent_load.get(key, 0))
        model.Add(sum(weight * chosen[index] for weight, index in items) <= bound)


__all__ = ["RouteSelectionUnavailable", "optimize_route_selection"]
