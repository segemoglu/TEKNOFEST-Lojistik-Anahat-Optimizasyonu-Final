"""Config-backed policies shared by costing and transport-plan validation.

Keeping unresolved competition interpretations in this object prevents hidden
constants from spreading through the calculator and validator.  The defaults
match ``config/rules.yaml``, which is the single place an ambiguous
competition rule may be interpreted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.domain.solution_models import CostGranularity
from src.io.manifest import load_manifest


def _require_fixed_rule(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    path: str,
) -> None:
    """Reject config that contradicts an implemented, non-optional rule."""

    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{path}.{key} kesin kuralla çelişiyor; "
            f"beklenen={expected!r}, bulunan={actual!r}."
        )


def _validate_fixed_rules(rules: Mapping[str, Any]) -> None:
    """Validate config fields that may document but may not alter fixed rules."""

    costs = rules.get("costs", {})
    handling = rules.get("handling", {})
    truck_counting = rules.get("truck_daily_quota", {}).get("counting", {})
    truck_quota = rules.get("truck_daily_quota", {})
    mandatory = rules.get("mandatory_rentals", {})
    minimum_fill = rules.get("minimum_fill", {})
    horizon = rules.get("planning_horizon", {})
    fixed_values = (
        (costs, "hourly_usage_rounding", "ceiling_to_next_whole_hour", "rules.costs"),
        (costs, "fractional_positive_hour_rounds_up", True, "rules.costs"),
        (costs, "sla_delay_rounding", "ceiling_to_next_whole_hour", "rules.costs"),
        (handling, "midnight_capacity_allocation", "proportional_by_elapsed_time", "rules.handling"),
        (handling, "same_vehicle_intermediate_sequence", "unload_then_load", "rules.handling"),
        (handling, "minimum_wait_between_unload_and_reload_minutes", 0.0, "rules.handling"),
        (handling, "simultaneous_dock_capacity_enabled", False, "rules.handling"),
        (handling, "parallel_vehicle_handling_allowed", True, "rules.handling"),
        (handling, "parallel_handling_requires_available_handling_capacity", True, "rules.handling"),
        (handling, "no_overlap_constraint_enabled", False, "rules.handling"),
        (handling, "vehicle_queue_constraint_enabled", False, "rules.handling"),
        (truck_counting, "departure_consumes_origin_quota_on_departure_day", 1, "rules.truck_daily_quota.counting"),
        (truck_counting, "arrival_consumes_destination_quota_on_arrival_day", 1, "rules.truck_daily_quota.counting"),
        (truck_counting, "arrivals_and_departures_share_one_daily_pool", True, "rules.truck_daily_quota.counting"),
        (truck_counting, "continuous_unload_reload_visit_counts_once", True, "rules.truck_daily_quota.counting"),
        (truck_counting, "leave_and_return_same_day_starts_new_visit", True, "rules.truck_daily_quota.counting"),
        (truck_counting, "handling_cross_midnight_splits_visit_quota", False, "rules.truck_daily_quota.counting"),
        (truck_quota, "applies_to_rented_trucks", True, "rules.truck_daily_quota"),
        (truck_quota, "applies_to_spot_trucks", True, "rules.truck_daily_quota"),
        (truck_quota, "period", "local_calendar_day", "rules.truck_daily_quota"),
        (truck_quota, "reset_time", "00:00", "rules.truck_daily_quota"),
        (mandatory, "cadence", "daily", "rules.mandatory_rentals"),
        (mandatory, "operate_when_empty", True, "rules.mandatory_rentals"),
        (mandatory, "empty_operation_costed_normally", True, "rules.mandatory_rentals"),
        (mandatory, "allow_intermediate_stops", False, "rules.mandatory_rentals"),
        (mandatory, "allow_return_trip", False, "rules.mandatory_rentals"),
        (mandatory, "unique_vehicle_id_per_daily_vehicle", True, "rules.mandatory_rentals"),
        (mandatory, "continue_after_forecast_end_while_demands_remain", True, "rules.mandatory_rentals"),
        (mandatory, "continuation_uses_same_counts", True, "rules.mandatory_rentals"),
        (mandatory, "continuation_uses_same_routes", True, "rules.mandatory_rentals"),
        (mandatory, "allow_additional_mandatory_vehicles_after_forecast_end", False, "rules.mandatory_rentals"),
        (mandatory, "allow_fewer_mandatory_vehicles_after_forecast_end", False, "rules.mandatory_rentals"),
        (mandatory, "create_new_departures_after_all_demands_delivered", False, "rules.mandatory_rentals"),
        (mandatory, "complete_started_vehicle_movements", True, "rules.mandatory_rentals"),
        (mandatory, "include_cost_of_started_vehicle_movements", True, "rules.mandatory_rentals"),
        (minimum_fill, "first_departure_minimum_ratio", 0.0, "rules.minimum_fill"),
        (minimum_fill, "empty_first_departure_allowed", True, "rules.minimum_fill"),
        (minimum_fill, "empty_vehicle_costed_normally", True, "rules.minimum_fill"),
        (horizon, "end_condition", "all_forecast_demands_delivered", "rules.planning_horizon"),
        (horizon, "allow_activity_after_forecast_window", True, "rules.planning_horizon"),
        (horizon, "include_post_window_travel_cost", True, "rules.planning_horizon"),
        (horizon, "include_post_window_handling_cost", True, "rules.planning_horizon"),
        (horizon, "include_post_window_vehicle_cost", True, "rules.planning_horizon"),
        (horizon, "include_post_window_delay_cost", True, "rules.planning_horizon"),
        (horizon, "allow_cost_avoidance_by_deferring_outside_window", False, "rules.planning_horizon"),
    )
    for mapping, key, expected, path in fixed_values:
        _require_fixed_rule(mapping, key, expected, path)


def _policy_kwargs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract typed constructor values after fixed-rule validation."""

    rules = manifest.get("rules", {})
    costs = rules.get("costs", {})
    handling = rules.get("handling", {})
    truck_quota = rules.get("truck_daily_quota", {})
    truck_counting = truck_quota.get("counting", {})
    mandatory = rules.get("mandatory_rentals", {})
    forecast = rules.get("forecast_window", {})
    solution = manifest.get("validation", {}).get("solution", {})
    _validate_fixed_rules(rules)
    return {
        "hourly_cost_granularity": str(costs.get("hourly_cost_granularity", "vehicle")),
        "usage_time_boundary_policy": str(costs.get("usage_time_boundary_policy", "activity_envelope")),
        "include_idle_between_activities": bool(costs.get("include_idle_between_activities", True)),
        "sla_penalty_per_desi_hour_tl": float(costs.get("sla_penalty_per_desi_hour_tl", 0.4)),
        "sla_rounding_scope": str(costs.get("sla_rounding_scope", "demand_part_final_delivery")),
        "handling_minutes_per_desi": float(handling.get("minutes_per_desi", 0.01)),
        "desi_tolerance": float(solution.get("desi_tolerance", 1e-6)),
        "travel_time_tolerance_minutes": float(solution.get("travel_time_tolerance_minutes", 0.01)),
        "handling_duration_tolerance_seconds": float(solution.get("handling_duration_tolerance_seconds", 1.0)),
        "cost_tolerance_tl": float(solution.get("cost_tolerance_tl", 0.01)),
        "capacity_evaluation_scope": str(solution.get("capacity_evaluation_scope", "vehicle_leg_carried_load")),
        "reported_cost_breakdown_required": bool(solution.get("reported_cost_breakdown_required", False)),
        "truck_vehicle_type": str(truck_quota.get("vehicle_type", "Tır")),
        "continuous_visit_midnight_policy": str(truck_counting.get("continuous_visit_midnight_policy", "arrival_day")),
        "mandatory_schedule_start_date": date.fromisoformat(str(forecast.get("start_date", "2026-06-29"))),
        "mandatory_forecast_end_date": date.fromisoformat(str(forecast.get("end_date", "2026-07-05"))),
        "mandatory_schedule_end_policy": str(mandatory.get("schedule_end_policy", "through_demand_completion_date")),
        "mandatory_completion_date_is_included": bool(mandatory.get("completion_date_is_included", True)),
    }


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """Hold config-backed choices shared by costing and validation.

    ``CostCalculator``, ``cost_audit`` and ``TransportPlanValidator`` receive
    this immutable object rather than reading files themselves.  Fields cover
    TL/desi-hour SLA rates, minute/second tolerances, cost granularity, Tır
    visit-day policy and mandatory schedule dates.  Unsupported or negative
    values raise ``ValueError`` during construction so no evaluator uses a
    silent fallback.
    """

    hourly_cost_granularity: CostGranularity = "vehicle"
    usage_time_boundary_policy: str = "activity_envelope"
    include_idle_between_activities: bool = True
    sla_penalty_per_desi_hour_tl: float = 0.4
    sla_rounding_scope: str = "demand_part_final_delivery"
    handling_minutes_per_desi: float = 0.01
    desi_tolerance: float = 1e-6
    travel_time_tolerance_minutes: float = 0.01
    handling_duration_tolerance_seconds: float = 1.0
    cost_tolerance_tl: float = 0.01
    capacity_evaluation_scope: str = "vehicle_leg_carried_load"
    reported_cost_breakdown_required: bool = False
    truck_vehicle_type: str = "Tır"
    continuous_visit_midnight_policy: str = "arrival_day"
    mandatory_schedule_start_date: date = date(2026, 6, 29)
    mandatory_forecast_end_date: date = date(2026, 7, 5)
    mandatory_schedule_end_policy: str = "through_demand_completion_date"
    mandatory_completion_date_is_included: bool = True

    def __post_init__(self) -> None:
        """Reject unsupported config choices before plan evaluation starts."""

        if self.hourly_cost_granularity not in {"vehicle", "trip", "leg"}:
            raise ValueError(
                "hourly_cost_granularity vehicle, trip veya leg olmalıdır."
            )
        if self.usage_time_boundary_policy != "activity_envelope":
            raise ValueError(
                "Bu sürüm yalnız usage_time_boundary_policy=activity_envelope "
                "seçeneğini destekler."
            )
        if self.sla_rounding_scope != "demand_part_final_delivery":
            raise ValueError(
                "Bu sürüm SLA yuvarlamasını demand_part_final_delivery "
                "kapsamında destekler."
            )
        if self.capacity_evaluation_scope != "vehicle_leg_carried_load":
            raise ValueError(
                "Bu sürüm kapasiteyi vehicle_leg_carried_load kapsamında denetler."
            )
        if self.continuous_visit_midnight_policy not in {
            "arrival_day",
            "departure_day",
            "both_days",
        }:
            raise ValueError(
                "continuous_visit_midnight_policy arrival_day, departure_day "
                "veya both_days olmalıdır."
            )
        if self.mandatory_schedule_end_policy != "through_demand_completion_date":
            raise ValueError(
                "Bu sürüm mandatory schedule bitişini "
                "through_demand_completion_date olarak destekler."
            )

        # Tolerances and rates are configuration values rather than plan data;
        # negative values would invalidate every comparison and are rejected.
        non_negative_values = {
            "sla_penalty_per_desi_hour_tl": self.sla_penalty_per_desi_hour_tl,
            "handling_minutes_per_desi": self.handling_minutes_per_desi,
            "desi_tolerance": self.desi_tolerance,
            "travel_time_tolerance_minutes": self.travel_time_tolerance_minutes,
            "handling_duration_tolerance_seconds": (
                self.handling_duration_tolerance_seconds
            ),
            "cost_tolerance_tl": self.cost_tolerance_tl,
        }
        for field_name, value in non_negative_values.items():
            if value < 0:
                raise ValueError(f"{field_name} negatif olamaz.")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "EvaluationPolicy":
        """Build evaluator policy from the central JSON-compatible manifest.

        The input is the mapping returned by ``src.io.manifest.load_manifest``;
        the output is an immutable policy consumed by cost and validator
        modules.  Fixed rules such as parallel handling, minute-based ``ceil``
        and unique mandatory identities are checked for exact config agreement.
        Missing/invalid dates or contradictory values raise ``ValueError``.
        """

        return cls(**_policy_kwargs(manifest))


def load_evaluation_policy(
    manifest_path: str | Path | None = None,
) -> EvaluationPolicy:
    """Load and validate the policy used by cost and validator entry points.

    ``manifest_path`` may select another JSON-compatible YAML file; ``None``
    uses the repository config.  The returned ``EvaluationPolicy`` is normally
    passed to ``CostCalculator`` or ``TransportPlanValidator``.  Manifest I/O,
    date parsing and fixed-rule contradictions propagate as explicit errors.
    """

    return EvaluationPolicy.from_manifest(load_manifest(manifest_path))


__all__ = ["EvaluationPolicy", "load_evaluation_policy"]
