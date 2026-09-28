"""ALNS main engine."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Tuple

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy, CostCalculator, TransportPlanValidator
from src.alns.state import ALNSSolutionState, ALNSSpotRoute
from src.alns.materializer import ALNSMaterializer
from src.alns.destroy import random_removal, worst_removal, shaw_removal
from src.alns.repair import greedy_insertion, regret_2_insertion
from src.alns.evaluator import ProxyEvaluator
from src.alns.exceptions import RepairInfeasibleError
from src.alns.candidate_archive import CandidateArchive
from src.alns.set_partition import (
    RouteSelectionUnavailable,
    optimize_route_selection,
)
from src.domain.solution_models import TransportPlan

logger = logging.getLogger(__name__)


class ALNSEngine:
    """Manages the ALNS iterations, simulated annealing, candidate archiving, and exact audit tracking."""

    def __init__(
        self,
        problem: ProblemData,
        policy: EvaluationPolicy,
        initial_state: ALNSSolutionState,
        initial_plan: TransportPlan,
        initial_cost: float,
        archive_capacity: int = 8,
        exact_check_interval: int = 25,
        seed: int = 42,
    ):
        self.problem = problem
        self.policy = policy
        self.best_ever_cost = initial_cost
        self.best_ever_plan = initial_plan
        # The state behind best_ever_plan.  Set-partitioning needs the state, not
        # the materialized plan, to know which routes the incumbent used.
        self.best_ever_state = initial_state
        
        self.archive_capacity = archive_capacity
        self.exact_check_interval = exact_check_interval
        self.archive = CandidateArchive(capacity=archive_capacity)
        self.audited_signatures: set[str] = set()

        # Every feasible route the search ever builds, keyed canonically.  The
        # loop throws away thousands of good routes because the *state* they
        # belonged to was not an improvement; a set-partitioning pass can still
        # combine them into something better than any single state visited.
        self.route_pool: dict[tuple, tuple[ALNSSpotRoute, float]] = {}
        self.set_partition_outcome: str = "not_run"
        self._set_partition_notes: list[str] = []

        # Diagnostics.  Without these the loop silently discards every rejected
        # candidate, which is how a systematically infeasible search once ran
        # 1500 iterations while improving nothing.
        self.completed_iterations = 0
        self.audit_attempts = 0
        self.audit_passed = 0
        self.audit_improved = 0
        self.audit_rejections: Counter[str] = Counter()
        self.proxy_infeasible_candidates = 0
        self.repair_infeasible_candidates = 0
        
        # Calculate initial proxy cost accurately
        self.proxy = ProxyEvaluator(problem, policy, initial_state)
        initial_proxy_cost = self.proxy.evaluate_state(initial_state)
        self.best_ever_proxy_cost = initial_proxy_cost
        self.best_proxy_state = initial_state
        
        if not math.isinf(initial_proxy_cost):
            self.archive.add(initial_state, initial_proxy_cost)
        
        self.current_state = initial_state
        self.current_cost = initial_proxy_cost

        self.materializer = ALNSMaterializer(problem, policy)
        self.calculator = CostCalculator(problem, policy)
        self.validator = TransportPlanValidator(problem, policy)
        

        self.rng = random.Random(seed)
        
        # Adaptive Mechanism
        self.destroy_ops = [random_removal, worst_removal, shaw_removal]
        self.repair_ops = [greedy_insertion, regret_2_insertion]
        
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        
        self._reset_scores()

    def _reset_scores(self):
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_calls = [0] * len(self.destroy_ops)
        self.repair_calls = [0] * len(self.repair_ops)

    def _update_weights(self, reaction_factor: float = 0.1):
        for i in range(len(self.destroy_ops)):
            if self.destroy_calls[i] > 0:
                self.destroy_weights[i] = (1 - reaction_factor) * self.destroy_weights[i] + reaction_factor * (self.destroy_scores[i] / self.destroy_calls[i])
        
        for i in range(len(self.repair_ops)):
            if self.repair_calls[i] > 0:
                self.repair_weights[i] = (1 - reaction_factor) * self.repair_weights[i] + reaction_factor * (self.repair_scores[i] / self.repair_calls[i])
                
        self._reset_scores()

    def _audit_state_exact(self, state: ALNSSolutionState) -> bool:
        """Perform exact audit (materialize -> validate -> calculate) for a candidate state.

        Strict rules:
        1. Never re-audit the same canonical state signature twice.
        2. Reject candidate on Materializer, Validator, or CostCalculator Exception or invalid/non-finite cost, without crashing ALNS.
        3. Never catch KeyboardInterrupt, SystemExit, or BaseException.
        4. Update best_ever_plan and best_ever_cost ONLY if materialized, valid, finite, and strictly cheaper than current exact incumbent.
        """
        sig = state.signature()
        if sig in self.audited_signatures:
            return False
        self.audited_signatures.add(sig)
        self.audit_attempts += 1

        try:
            plan = self.materializer.materialize(state)
        except Exception as e:
            self.audit_rejections["materializer_exception"] += 1
            logger.warning(f"Exact Audit materializer exception for state {sig[:8]}: {e}")
            return False

        try:
            report = self.validator.validate(plan)
        except Exception as e:
            self.audit_rejections["validator_exception"] += 1
            logger.warning(f"Exact Audit validator exception for state {sig[:8]}: {e}")
            return False

        if not report.is_valid:
            # Count every distinct rule the candidate broke, not just the first,
            # so the run summary shows which constraint dominates.  Diagnostics
            # must never crash the search, so an unexpected issue shape falls
            # back to its string form instead of assuming the dataclass.
            codes = {
                getattr(issue, "code", None) or str(issue)
                for issue in report.errors
            }
            for code in codes:
                self.audit_rejections[code] += 1
            logger.warning(f"Exact Audit FAILED for state {sig[:8]}: {sorted(codes)}")
            return False

        try:
            cost_result = self.calculator.calculate(plan)
            exact_cost = cost_result.total_cost_tl
        except Exception as e:
            self.audit_rejections["calculator_exception"] += 1
            logger.warning(f"Exact Audit calculator exception for state {sig[:8]}: {e}")
            return False

        if math.isnan(exact_cost) or math.isinf(exact_cost):
            self.audit_rejections["non_finite_cost"] += 1
            logger.warning(f"Exact Audit non-finite cost ({exact_cost}) for state {sig[:8]}")
            return False

        self.audit_passed += 1
        if exact_cost >= self.best_ever_cost:
            self.audit_rejections["valid_but_not_cheaper"] += 1

        if exact_cost < self.best_ever_cost:
            self.audit_improved += 1
            logger.info(
                f"Exact Audit PASSED! New best exact cost: {exact_cost:,.4f} TL "
                f"(prev: {self.best_ever_cost:,.4f} TL)"
            )
            self.best_ever_cost = exact_cost
            self.best_ever_plan = plan
            self.best_ever_state = state
            return True
        else:
            logger.info(
                f"Exact Audit PASSED but cost {exact_cost:,.4f} TL is not lower than best {self.best_ever_cost:,.4f} TL."
            )
            return False

    def _audit_state(self, state: ALNSSolutionState) -> bool:
        """Alias for _audit_state_exact for backward compatibility."""
        return self._audit_state_exact(state)

    def _collect_routes(self, state: ALNSSolutionState) -> None:
        """Remember every distinct feasible route this candidate contains.

        The proxy cost is memoized, so re-reading it here is nearly free.  Routes
        are keyed by day, vehicle type and the exact stop sequence, which makes
        the pool a set of *distinct* routes rather than one entry per iteration.

        Only called when set-partitioning is enabled, and it skips any route
        whose vehicle type the proxy does not know: collecting routes is
        bookkeeping for an optional post-pass and must never stop the search.
        """

        for route in state.spot_routes:
            if route.vehicle_type not in self.proxy.vehicles:
                continue
            key = (
                route.service_date,
                route.vehicle_type,
                tuple(stop.stop_key for stop in route.stops),
            )
            if key in self.route_pool:
                continue
            cost = self.proxy.evaluate_route(
                route.service_date, route.stops, route.vehicle_type
            )
            if not math.isinf(cost):
                self.route_pool[key] = (route, cost)

    def release(self) -> None:
        """Free the memory this search accumulated, keeping its results.

        Route cache, audited signatures and the candidate archive are only
        useful during a run.  A caller that searches several seeds in one
        process calls this between runs; without it the retained memory
        measurably slowed later seeds (0,51 -> 1,10 seconds per iteration) until
        they hit their time limit.  ``best_ever_plan``, ``best_ever_cost`` and
        the diagnostics counters survive so the result stays readable.
        """

        self.proxy.clear_cache()
        self.audited_signatures.clear()
        self.archive.clear()
        self.route_pool.clear()

    def refine_with_set_partitioning(
        self,
        time_limit_seconds: float = 300.0,
        candidate_count: int = 60,
    ) -> bool:
        """Recombine the collected routes and keep the result only if cheaper.

        Args:
            time_limit_seconds: Wall-clock ceiling for the CP-SAT solve.

        Returns:
            ``True`` when the recombined plan became the new incumbent.

        Why this can only help:
            The chosen combination goes through :meth:`_audit_state_exact`, the
            same gate every search candidate passes: materialize, validate,
            cost, accept only if strictly cheaper.  A solver answer that the
            authoritative pipeline dislikes is discarded and the ALNS plan
            stands.  The objective uses proxy route costs, which ignore handling
            deferral, so this gate is doing real work rather than rubber-stamping.
        """

        if not self.route_pool:
            logger.info("Set-partitioning atlandı: rota havuzu boş.")
            return False
        try:
            candidates = optimize_route_selection(
                self.route_pool,
                self.best_ever_state,
                self.problem,
                self.policy,
                self.proxy,
                time_limit_seconds=time_limit_seconds,
                candidate_count=candidate_count,
            )
        except RouteSelectionUnavailable as exc:
            logger.info("Set-partitioning kullanılamadı: %s", exc)
            self.set_partition_outcome = f"unavailable: {exc}"
            return False
        before = self.best_ever_cost
        self._set_partition_notes = []
        improved_any = False
        for order, recombined in enumerate(candidates, start=1):
            if self._try_recombination(recombined, before, order):
                improved_any = True
        gain = before - self.best_ever_cost
        self.set_partition_outcome = (
            f"{len(candidates)} aday, kazanç {gain:,.2f} TL | "
            + " ".join(self._set_partition_notes)
        )
        if improved_any:
            logger.info(
                "Set-partitioning kazancı: %.2f TL (%.4f%%)",
                gain,
                gain / before * 100.0,
            )
        return improved_any

    def _try_recombination(
        self,
        recombined: ALNSSolutionState,
        reference_cost: float,
        order: int,
    ) -> bool:
        """Report and gate one recombined candidate through the exact pipeline."""

        before = self.best_ever_cost
        # Report *why* a recombination was refused.  "Not cheaper" means the proxy
        # objective was optimistic; a validator code means the routes came from
        # states whose schedules cannot coexist in one timeline.  Without this
        # distinction the two look identical in the log.
        try:
            plan = self.materializer.materialize(recombined)
            report = self.validator.validate(plan)
            if not report.is_valid:
                codes = sorted(
                    {getattr(issue, "code", str(issue)) for issue in report.errors}
                )
                logger.info(
                    "Aday %d geçersiz: %s", order, ", ".join(codes)
                )
                self._set_partition_notes.append(f"#{order} invalid:{codes[0]}")
                return False
            exact_cost = self.calculator.calculate(plan).total_cost_tl
            logger.info(
                "Aday %d kesin maliyeti: %.2f TL (mevcut en iyi: %.2f TL)",
                order,
                exact_cost,
                before,
            )
        except Exception as exc:
            logger.info("Aday %d materialize/cost edilemedi: %s", order, exc)
            self._set_partition_notes.append(f"#{order} exception")
            return False

        improved = self._audit_state_exact(recombined)
        self._set_partition_notes.append(
            f"#{order} {'improved' if improved else 'not_cheaper'}"
            f" {exact_cost:,.0f}"
        )
        return improved

    def diagnostics(self) -> dict[str, object]:
        """Return why candidates were discarded during the last run.

        Returns:
            Iteration count, proxy/repair rejection counts, exact-audit
            attempt/pass/improve counts and a rejection breakdown keyed by
            validator issue code.

        Why this exists:
            A search that is systematically steered into infeasible space looks
            identical to a search that simply cannot improve.  The breakdown
            distinguishes them: a single dominant issue code means a constraint
            the proxy does not model, while ``valid_but_not_cheaper`` means the
            search is genuinely converged.
        """

        return {
            "completed_iterations": self.completed_iterations,
            "repair_infeasible": self.repair_infeasible_candidates,
            "proxy_infeasible": self.proxy_infeasible_candidates,
            "audit_attempts": self.audit_attempts,
            "audit_valid": self.audit_passed,
            "audit_improved": self.audit_improved,
            "rejections_by_reason": dict(self.audit_rejections.most_common()),
        }

    def render_diagnostics(self) -> str:
        """Render :meth:`diagnostics` as an operator-readable summary block."""

        data = self.diagnostics()
        lines = [
            f"Tamamlanan iterasyon      : {data['completed_iterations']}",
            f"Onarım başarısız          : {data['repair_infeasible']}",
            f"Proxy fizibil bulmadı     : {data['proxy_infeasible']}",
            f"Exact audit denemesi      : {data['audit_attempts']}",
            f"  geçerli bulunan         : {data['audit_valid']}",
            f"  maliyeti düşüren        : {data['audit_improved']}",
            f"Rota havuzu               : {len(self.route_pool)}",
            f"Set-partitioning          : {self.set_partition_outcome}",
        ]
        rejections = data["rejections_by_reason"]
        if rejections:
            lines.append("Ret sebepleri:")
            lines.extend(
                f"  {reason:<32s} {count}"
                for reason, count in rejections.items()
            )
        else:
            lines.append("Ret sebebi kaydedilmedi.")
        return "\n".join(lines)

    def run(
        self,
        max_iterations: int = 10,
        start_temperature: float = 10000.0,
        end_temperature: float = 1.0,
        segment_size: int = 100,
        time_limit_seconds: float = 600.0,
        stagnation_limit: int = 250,
        set_partition_seconds: float = 0.0,
    ) -> tuple[TransportPlan, float]:
        """Run the ALNS loop and return the best exact plan found.

        ``set_partition_seconds`` budgets one CP-SAT recombination of every route
        the search built (see :meth:`refine_with_set_partitioning`).  It defaults
        to **off** because it was measured to gain nothing: route costs are not
        additive in reality, so the solver exploits the proxy's optimism instead
        of finding real savings.
        """
        import time
        start_time = time.time()
        logger.info("Starting ALNS execution with candidate archiving and exact audit policy...")
        
        cooling_rate = (end_temperature / start_temperature) ** (1.0 / max_iterations)
        temperature = start_temperature
        
        iterations_since_best_proxy = 0

        for iteration in range(1, max_iterations + 1):
            if time.time() - start_time > time_limit_seconds:
                logger.info(f"Iteration {iteration}: Time limit reached ({time_limit_seconds}s). Stopping.")
                break
                
            num_stops = sum(len(r.stops) for r in self.current_state.spot_routes)
            if num_stops == 0:
                logger.debug(f"Iteration {iteration}: No spot routes to optimize.")
                break

            self.completed_iterations = iteration
            num_parts = len(self.current_state.parts)
            # The proportion is the design; the old min(25, ...) silently broke
            # it.  On 4.178 parts the intended 0,5-1,5% band is 20-62 atoms, but
            # the cap clipped every draw to 20-25, so destruction never exceeded
            # 0,6%.  That is far too small to dismantle a long milk-run, which is
            # exactly what the search now has to restructure after S-24.
            n_remove = max(2, int(num_parts * self.rng.uniform(0.005, 0.015)))
            
            # Select Operators
            d_idx = self.rng.choices(range(len(self.destroy_ops)), weights=self.destroy_weights)[0]
            r_idx = self.rng.choices(range(len(self.repair_ops)), weights=self.repair_weights)[0]
            
            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]
            
            self.destroy_calls[d_idx] += 1
            self.repair_calls[r_idx] += 1
            
            # 1. Destroy
            if destroy_op == random_removal:
                destroyed_state, unassigned = destroy_op(
                    self.current_state, n_remove, self.rng
                )
            else:
                destroyed_state, unassigned = destroy_op(
                    self.current_state, n_remove, self.rng, self.problem, self.policy, self.proxy
                )

            # 2. Repair
            try:
                candidate_state = repair_op(
                    destroyed_state, unassigned, self.problem, self.policy, self.proxy
                )
            except RepairInfeasibleError:
                self.repair_infeasible_candidates += 1
                logger.debug(f"Iteration {iteration}: Repair operator returned infeasible state.")
                continue

            # 3. Fast Proxy Evaluation
            proxy_cost = self.proxy.evaluate_state(candidate_state)

            # 4. Candidate Archiving
            if math.isinf(proxy_cost):
                self.proxy_infeasible_candidates += 1
            else:
                self.archive.add(candidate_state, proxy_cost)
                if set_partition_seconds > 0:
                    self._collect_routes(candidate_state)

            # 5. Accept/Reject via SA
            delta = proxy_cost - self.current_cost
            accepted = False
            score = 0

            if delta < 0:
                accepted = True
                if proxy_cost < self.best_ever_proxy_cost - 1e-3:
                    score = 5 # Global Best
                else:
                    score = 3 # Better than current
            elif proxy_cost < float("inf"):
                exponent = -delta / temperature
                if exponent < -700:
                    probability = 0.0
                else:
                    probability = math.exp(exponent)
                
                if self.rng.random() < probability:
                    accepted = True
                    score = 1 # Accepted worse
                    logger.debug(
                        f"Iteration {iteration}: Accepted worse solution (delta: +{delta:.2f}, prob: {probability:.3f})"
                    )

            is_new_proxy_record = False
            if not math.isinf(proxy_cost):
                if proxy_cost < self.best_ever_proxy_cost:
                    is_new_proxy_record = True
                    self.best_ever_proxy_cost = proxy_cost
                    self.best_proxy_state = candidate_state
                elif proxy_cost == self.best_ever_proxy_cost:
                    if candidate_state.canonical_tuple() < self.best_proxy_state.canonical_tuple():
                        is_new_proxy_record = True
                        self.best_proxy_state = candidate_state

            # Trigger 1: Immediate audit on new proxy record
            if is_new_proxy_record:
                self._audit_state_exact(candidate_state)

            # Trigger 2: Periodic audit every exact_check_interval iterations
            if iteration % self.exact_check_interval == 0:
                for _, st in self.archive.get_candidates():
                    self._audit_state_exact(st)

            # Apply Accepted State
            if accepted:
                self.current_state = candidate_state
                self.current_cost = proxy_cost
                self.destroy_scores[d_idx] += score
                self.repair_scores[r_idx] += score
            if is_new_proxy_record:
                iterations_since_best_proxy = 0
            else:
                iterations_since_best_proxy += 1
                
            temperature *= cooling_rate
            
            if iterations_since_best_proxy >= stagnation_limit:
                logger.debug(f"Iteration {iteration}: Stagnation limit reached ({stagnation_limit}). Reheating.")
                temperature = max(temperature, start_temperature / 2.0)
                iterations_since_best_proxy = 0
                
            if iteration % segment_size == 0:
                self._update_weights()
                
            if iteration % 500 == 0:
                logger.info(f"Iteration {iteration}: Proxy Cost = {self.best_ever_proxy_cost:,.2f} TL (Temp: {temperature:.2f})")
                
        # --- END OF MAIN LOOP ---
        
        # Trigger 3: Final sweep - audit all remaining un-audited candidates in archive
        logger.info("Main ALNS loop finished. Auditing all remaining un-audited candidates in candidate archive...")
        for _, st in self.archive.get_candidates():
            self._audit_state_exact(st)

        # Trigger 4: recombine every route the search built into the cheapest
        # exact cover.  Runs once, outside the loop, so it costs no iterations.
        if set_partition_seconds > 0:
            logger.info(
                "Rota havuzu: %d farklı rota. Set-partitioning başlıyor...",
                len(self.route_pool),
            )
            self.refine_with_set_partitioning(
                time_limit_seconds=set_partition_seconds,
            )

        return self.best_ever_plan, self.best_ever_cost

__all__ = ["ALNSEngine"]
