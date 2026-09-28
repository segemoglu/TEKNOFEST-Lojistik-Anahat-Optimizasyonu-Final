"""Build and export an optimized transport plan using ALNS.

Run ``python -m src.planning.run_alns`` to build a 6E reference,
optimize it using ALNS, and write the configured official workbook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

from src.domain import ProblemData
from src.evaluation import EvaluationPolicy
from src.io import ExcelProblemLoader, configured_project_path
from src.io.transport_plan_workbook import (
    TransportPlanWorkbookExport,
    TransportPlanWorkbookExporter,
)

from .baseline import (
    BaselineBuildError,
    BaselineBuildResult,
    ConservativeDirectBaselineBuilder,
)
from .consolidated import GreedySameODConsolidatedBuilder
from .mandatory_filled import GreedyMandatoryRentalFilledBuilder
from .multi_stop import GreedySameOriginMultiStopBuilder
from .two_stop import GreedySameOriginTwoStopBuilder
from src.alns.materializer import ALNSMaterializer
from src.alns.engine import ALNSEngine


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for ALNS run."""
    parser = argparse.ArgumentParser(
        description="ALNS ile optimize edilmiş TAŞIMA PLANI.xlsx üretir."
    )
    parser.add_argument("--iterations", type=int, default=200, help="ALNS döngü sayısı")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Tohum başına maksimum çalışma süresi (saniye)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Virgülle ayrılmış tohum listesi, örn. 42,7,123,999. "
            "Her tohum ayrı çalıştırılır ve en ucuz geçerli plan yazılır. "
            "Verilmezse --seed kullanılır."
        ),
    )
    parser.add_argument(
        "--set-partition-seconds",
        type=float,
        default=0.0,
        help=(
            "Koşu sonunda toplanan rotaları CP-SAT ile yeniden birleştirmeye "
            "ayrılacak saniye. Varsayılan 0: ölçüldü, kazanç üretmiyor."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/TAŞIMA PLANI.xlsx"), help="Çıktı konumu")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    return parser


def parse_seeds(raw: str | None, fallback: int) -> tuple[int, ...]:
    """Turn a comma-separated seed list into a deterministic tuple.

    Args:
        raw: CLI text such as ``"42,7,123"``; ``None`` selects ``fallback``.
        fallback: The single ``--seed`` value used when no list is supplied.

    Returns:
        Seeds in the given order, duplicates removed, order preserved so the
        winner of a tie is reproducible.

    Raises:
        ValueError: If the list is empty or holds a non-integer entry.
    """

    if raw is None:
        return (fallback,)
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            seed = int(text)
        except ValueError as exc:
            raise ValueError(f"Geçersiz tohum değeri: {text!r}") from exc
        if seed not in values:
            values.append(seed)
    if not values:
        raise ValueError("--seeds en az bir tohum içermelidir.")
    return tuple(values)


def _error_summary(error: BaselineBuildError) -> str:
    """Render a build failure's machine-readable codes, or its message."""

    return ", ".join(error.error_codes) or str(error)


def build_seed_plan(
    problem: ProblemData,
    policy: EvaluationPolicy,
) -> tuple[BaselineBuildResult, str]:
    """Return the richest greedy start plan that builds, and its layer label.

    Args:
        problem: Static network data paired with the demands being planned.
        policy: Shared costing, capacity, SLA and validation policy.

    Returns:
        A validated ``BaselineBuildResult`` and the layer that produced it:
        ``"6E"``, ``"6D"``, ``"6B"``, ``"6A"`` or ``"konservatif"``.

    Why this ladder exists:
        6D and 6E raise ``BaselineBuildError`` when they cannot *strictly beat*
        the layer below them — ``two_stop_no_strict_saving`` and
        ``multi_stop_no_strict_saving`` are refusals to improve, not signs of
        broken input.  Calling them directly made one such refusal end the whole
        run with a traceback and no workbook, even though the layer below had
        already produced a valid, costed, validator-approved plan.  ALNS only
        needs a valid start state, so degrading one layer is always better than
        failing.  Whichever layer wins is printed, because a silent drop from 6E
        to 6A would otherwise look like an unexplained cost regression.

    Raises:
        BaselineBuildError: Only when every layer fails, which means the input
            itself cannot be planned and no fallback would help.
    """

    two_stop: BaselineBuildResult | None = None
    try:
        two_stop = GreedySameOriginTwoStopBuilder(problem, policy).build()
    except BaselineBuildError as exc:
        print(f"6D kurulamadı ({_error_summary(exc)}); alt katmanlara düşülüyor.")
    if two_stop is not None:
        try:
            builder = GreedySameOriginMultiStopBuilder(problem, policy)
            return builder.build_from_two_stop(two_stop), "6E"
        except BaselineBuildError as exc:
            print(f"6E kurulamadı ({_error_summary(exc)}); 6D planı kullanılıyor.")
            return two_stop, "6D"
    # Resolved here rather than at module level so the layers stay patchable
    # and no builder class is instantiated on the successful 6E path.
    fallbacks = (
        ("6B", GreedyMandatoryRentalFilledBuilder),
        ("6A", GreedySameODConsolidatedBuilder),
        ("konservatif", ConservativeDirectBaselineBuilder),
    )
    for label, builder_class in fallbacks:
        try:
            return builder_class(problem, policy).build(), label
        except BaselineBuildError as exc:
            print(f"{label} kurulamadı ({_error_summary(exc)}).")
    raise BaselineBuildError(
        "Hiçbir greedy katmanı geçerli başlangıç planı üretemedi.",
        error_codes=("greedy_chain_exhausted",),
    )


def run_alns_plan(
    iterations: int,
    seed: int,
    time_limit_seconds: float = 600.0,
    manifest_path: str | Path | None = None,
    forecast_path: str | Path | None = None,
    output_path: str | Path | None = None,
    seeds: tuple[int, ...] | None = None,
    set_partition_seconds: float = 0.0,
) -> TransportPlanWorkbookExport:
    """Build the greedy start plan, search it with ALNS and export the best.

    Args:
        iterations: ALNS iteration target for each seed.
        seed: Single random seed, used when ``seeds`` is not supplied.
        time_limit_seconds: Wall-clock ceiling **per seed**.
        manifest_path: Alternative config manifest.
        forecast_path: Alternative demand workbook; this is how the real
            forecast replaces the development one.
        output_path: Destination workbook.
        seeds: Optional list of seeds. Every seed searches the same start
            state and the cheapest valid result is exported.

    Start plan:
        ``build_seed_plan`` normally returns 6E, but degrades to 6D, 6B, 6A or
        the conservative baseline when a layer refuses to improve.  The layer
        actually used is printed, since it changes the cost the run starts from.

    Why multiple seeds:
        The search result varies far more with the seed than with most code
        changes. Measured on this dataset at 1500 iterations, four seeds spanned
        494.760 TL with a standard deviation near 197.000 TL. Running several
        seeds and keeping the cheapest is therefore worth more than many
        algorithmic tweaks, and costs nothing but time. See

        The greedy start plan is deterministic, so it is built once and shared
        by every seed instead of being rebuilt each time.
    """

    if iterations <= 0:
        raise ValueError("Iterations must be greater than 0")
    if time_limit_seconds <= 0:
        raise ValueError("Time limit must be greater than 0")
    selected_seeds = tuple(seeds) if seeds else (seed,)

    loader = ExcelProblemLoader(manifest_path)
    static = loader.load_static_problem_data()
    selected_forecast = (
        Path(forecast_path).resolve()
        if forecast_path is not None
        else configured_project_path(
            loader.manifest,
            "development_forecast",
            project_root=loader.project_root,
        )
    )
    demands = loader.load_demands_from_workbook(
        selected_forecast,
        static_data=static,
    )
    problem = ProblemData(static, demands)
    policy = EvaluationPolicy.from_manifest(loader.manifest)

    print("Başlangıç Planı Kuruluyor...")
    seed_result, seed_layer = build_seed_plan(problem, policy)

    original_plan = seed_result.plan
    original_cost = seed_result.total_cost_tl
    print(f"Başlangıç Planı Katmanı: {seed_layer}")
    print(f"Mevcut {seed_layer} Maliyeti: {original_cost:,.4f} TL")

    print(
        f"ALNS Optimizasyonu Başlıyor ({iterations} hedef iterasyon, "
        f"tohum başına max {time_limit_seconds}s, tohumlar: "
        f"{', '.join(str(item) for item in selected_seeds)})..."
    )
    materializer = ALNSMaterializer(problem, policy)
    initial_state = materializer.extract_state(original_plan)

    # Optionally enable logging to see operator weights
    logging.getLogger("src.alns.engine").setLevel(logging.INFO)

    import gc
    import time

    # The greedy start plan is always a valid fallback, so an ALNS run that
    # finds nothing cheaper simply leaves the incumbent untouched.
    final_plan = original_plan
    best_cost = original_cost
    winning_seed: int | None = None
    seed_results: list[tuple[int, float, float]] = []

    for current_seed in selected_seeds:
        # Each engine owns a proxy route cache holding up to 100.000 entries
        # plus an audited-signature set.  Without an explicit release between
        # seeds the retained memory slowed later seeds from 0,51 to 1,10
        # seconds per iteration, so they hit the time limit and returned worse
        # plans than the same seeds reach when run on their own.
        gc.collect()
        start_time = time.time()
        engine = ALNSEngine(
            problem,
            policy,
            initial_state,
            original_plan,
            original_cost,
            seed=current_seed,
        )
        plan, cost = engine.run(
            max_iterations=iterations,
            time_limit_seconds=time_limit_seconds,
            start_temperature=10000.0,
            end_temperature=1.0,
            set_partition_seconds=set_partition_seconds,
        )
        elapsed_seconds = time.time() - start_time
        completed_iterations = engine.completed_iterations
        if completed_iterations >= iterations:
            stop_reason = "completed"
        elif elapsed_seconds >= time_limit_seconds:
            stop_reason = "time_limit"
        else:
            stop_reason = "early_stop"
        seed_results.append((current_seed, cost, elapsed_seconds))

        print(f"--- Tohum {current_seed} ---")
        print(f"Maliyet: {cost:,.4f} TL")
        print(f"Tamamlanan İterasyon: {completed_iterations} / {iterations} ({stop_reason})")
        print(f"Geçen Süre: {elapsed_seconds:.2f} saniye")
        print(engine.render_diagnostics())

        # Strict improvement keeps the earliest seed on a tie, so a rerun with
        # the same seed list always exports the same plan.
        if cost < best_cost:
            final_plan = plan
            best_cost = cost
            winning_seed = current_seed

        # Drop the search memory before the next seed builds its own.
        engine.release()
        del engine, plan

    print("=== TOHUM KARŞILAŞTIRMASI ===")
    for current_seed, cost, elapsed_seconds in seed_results:
        mark = " <-- seçilen" if current_seed == winning_seed else ""
        print(f"  tohum {current_seed:>5d}: {cost:>16,.2f} TL  ({elapsed_seconds:.0f}s){mark}")
    if len(seed_results) > 1:
        costs = [cost for _, cost, _ in seed_results]
        print(f"  en iyi / en kötü fark : {max(costs) - min(costs):,.2f} TL")
    if winning_seed is None:
        print(
            f"  Hiçbir tohum {seed_layer} planını geçemedi; "
            f"{seed_layer} planı yazılıyor."
        )

    improvement_tl = original_cost - best_cost
    print(f"Seçilen Maliyet: {best_cost:,.4f} TL")
    print(
        f"Kazanç: {improvement_tl:,.4f} TL "
        f"(%{improvement_tl / original_cost * 100.0:.4f})"
    )

    print("Nihai Plan için Kesin Maliyet Hesaplanıyor...")
    from src.evaluation.costs import CostCalculator
    import dataclasses
    cost_calc = CostCalculator(problem, policy)
    cost_breakdown = cost_calc.calculate(final_plan)
    final_plan = dataclasses.replace(final_plan, reported_cost=cost_breakdown)

    selected_output = Path(output_path).resolve() if output_path is not None else Path("outputs/TAŞIMA PLANI.xlsx").resolve()
    
    print("Excel'e Yazdırılıyor ve Doğrulanıyor...")
    return TransportPlanWorkbookExporter(
        problem,
        policy,
        loader.manifest,
    ).write(final_plan, selected_output)


def render_export_summary(result: TransportPlanWorkbookExport) -> str:
    """Render path, leg/vehicle counts and reconciled TL audit totals."""
    return "\n".join(
        (
            f"Taşıma planı: {result.path}",
            f"Satır / fiziksel araç: {result.audit.row_count} / {result.audit.vehicle_count}",
            f"Toplam maliyet: {result.audit.total_cost_tl:,.2f} TL",
            f"SLA cezası: {result.audit.sla_penalty_tl:,.2f} TL",
            "Round-trip doğrulama: GEÇTİ",
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    result = run_alns_plan(
        iterations=args.iterations,
        seed=args.seed,
        time_limit_seconds=args.time_limit,
        manifest_path=args.manifest,
        forecast_path=args.forecast,
        output_path=args.output,
        seeds=parse_seeds(args.seeds, args.seed),
        set_partition_seconds=args.set_partition_seconds,
    )
    print("\n================ FINAL RAPOR ================")
    print(render_export_summary(result))
    print("=============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
