"""Command-line smoke inspection for source data (no optimization)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from .excel_loader import ExcelProblemLoader


def build_parser() -> argparse.ArgumentParser:
    """Create the read-only inspection command's argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Excel kaynaklarını yükler, doğrular ve kısa bir özet yazar. "
            "Tahmin veya optimizasyon çalıştırmaz."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Varsayılan: config/rules.yaml",
    )
    parser.add_argument(
        "--demand",
        default="historical_demand",
        help="Manifestteki mantıksal talep kaydı (varsayılan: historical_demand)",
    )
    return parser


def render_summary(loader: ExcelProblemLoader, demand_name: str) -> str:
    """Load validated data and render a short non-optimization summary."""

    problem = loader.load_problem_data(demand_name)
    static = problem.static
    demands = problem.demands
    date_min = min((demand.demand_date for demand in demands), default=None)
    date_max = max((demand.demand_date for demand in demands), default=None)
    slot_counts = Counter(demand.time_slot.strftime("%H:%M") for demand in demands)
    split_count = sum(demand.split_required for demand in demands)
    zero_truck_centers = sorted(
        item.center
        for item in static.daily_truck_capacities
        if item.max_daily_truck_visits == 0
    )
    mandatory_vehicle_count = sum(
        route.vehicle_count for route in static.mandatory_rental_routes
    )

    lines = [
        "Linehaul veri incelemesi: doğrulama başarılı",
        f"Manifest: {loader.manifest_path.resolve()}",
        f"Ham veri klasörü: {loader.raw_directory}",
        f"Merkez / yönlü ağ yayı: {len(static.centers)} / {len(static.travel_arcs)}",
        f"Araç tipi: {len(static.vehicle_types)}",
        (
            "Zorunlu kiralık rota / günlük araç: "
            f"{len(static.mandatory_rental_routes)} / {mandatory_vehicle_count}"
        ),
        f"Talep kaynağı / satır: {demand_name} / {len(demands)}",
        f"Talep tarih aralığı: {date_min or '-'} - {date_max or '-'}",
        "Saat dilimleri: "
        + (
            ", ".join(f"{slot}={count}" for slot, count in sorted(slot_counts.items()))
            if slot_counts
            else "-"
        ),
        f"Bölme gereken talep: {split_count}",
        (
            f"Günlük Tır kapasitesi sıfır merkez: {len(zero_truck_centers)}"
            + (f" ({', '.join(zero_truck_centers)})" if zero_truck_centers else "")
        ),
        "Optimizasyon çalıştırılmadı.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run inspection and convert loader failures into a non-zero exit code."""

    args = build_parser().parse_args(argv)
    try:
        loader = ExcelProblemLoader(args.manifest)
        print(render_summary(loader, args.demand))
    except Exception as exc:  # CLI boundary: concise error, non-zero exit.
        print(f"Veri incelemesi başarısız: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
