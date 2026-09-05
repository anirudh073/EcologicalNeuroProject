"""Small, dataset-independent analysis utilities."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


__all__ = [
    "BrakingMetrics",
    "BrakingPeriod",
    "calculate_braking_metrics",
    "calculate_closure_speed",
    "detect_braking_period",
    "differentiate_time_series",
    "find_distance_matched_pairs",
    "smooth_time_series",
]


@dataclass(frozen=True)
class BrakingPeriod:
    """Indices and times delimiting a sustained final braking period."""

    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    negative_fraction: float


@dataclass(frozen=True)
class BrakingMetrics:
    """Point and period metrics using the spatial units of the input arrays."""

    onset_time_s: float
    arrival_time_s: float
    time_to_arrival_s: float
    remaining_gap_at_onset: float
    closure_speed_at_onset: float
    tau_at_onset_s: float
    speed_at_onset: float
    acceleration_at_onset: float
    braking_duration_s: float
    speed_at_arrival: float
    speed_reduction: float
    distance_travelled: float
    mean_deceleration: float
    max_deceleration: float
    negative_acceleration_fraction: float


def _prepare_time_series(
    timestamps_s: Sequence[float],
    values: Sequence[float],
    max_gap_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    """Validate a time series and identify continuous finite runs."""
    time = np.asarray(timestamps_s, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if time.ndim != 1 or value_array.ndim != 1:
        raise ValueError("timestamps_s and values must be 1D.")
    if len(time) != len(value_array) or len(time) == 0:
        raise ValueError("Input arrays must have equal, non-zero lengths.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing.")
    if max_gap_multiplier <= 0:
        raise ValueError("max_gap_multiplier must be positive.")

    finite_indices = np.flatnonzero(np.isfinite(value_array))
    if finite_indices.size < 2:
        return time, value_array, [], np.nan
    median_dt_s = float(np.median(np.diff(time[finite_indices])))
    split_after = np.where(
        (np.diff(finite_indices) != 1)
        | (np.diff(time[finite_indices]) > max_gap_multiplier * median_dt_s)
    )[0] + 1
    runs = [run for run in np.split(finite_indices, split_after) if run.size >= 3]
    return time, value_array, runs, median_dt_s


def smooth_time_series(
    timestamps_s: Sequence[float],
    values: Sequence[float],
    *,
    smoothing_sigma_s: float,
    max_gap_multiplier: float = 3.0,
) -> np.ndarray:
    """Gaussian-smooth continuous finite runs using a sigma in seconds."""
    if smoothing_sigma_s < 0:
        raise ValueError("smoothing_sigma_s must be non-negative.")
    time, value_array, runs, median_dt_s = _prepare_time_series(
        timestamps_s, values, max_gap_multiplier
    )
    smoothed = np.full(value_array.shape, np.nan, dtype=float)
    for run_indices in runs:
        if smoothing_sigma_s == 0:
            smoothed[run_indices] = value_array[run_indices]
        else:
            smoothed[run_indices] = gaussian_filter1d(
                value_array[run_indices],
                sigma=smoothing_sigma_s / median_dt_s,
                mode="nearest",
            )
    return smoothed


def differentiate_time_series(
    timestamps_s: Sequence[float],
    values: Sequence[float],
    *,
    max_gap_multiplier: float = 3.0,
) -> np.ndarray:
    """Differentiate continuous finite runs with respect to irregular time."""
    time, value_array, runs, _ = _prepare_time_series(
        timestamps_s, values, max_gap_multiplier
    )
    derivative = np.full(value_array.shape, np.nan, dtype=float)
    for run_indices in runs:
        derivative[run_indices] = np.gradient(
            value_array[run_indices], time[run_indices]
        )
    return derivative


def calculate_closure_speed(
    timestamps_s: Sequence[float],
    remaining_gap: Sequence[float],
    *,
    smoothing_sigma_s: float = 0.0,
    max_gap_multiplier: float = 3.0,
) -> np.ndarray:
    """Smooth a remaining gap and calculate its signed closure speed."""
    smoothed_gap = smooth_time_series(
        timestamps_s,
        remaining_gap,
        smoothing_sigma_s=smoothing_sigma_s,
        max_gap_multiplier=max_gap_multiplier,
    )
    return -differentiate_time_series(
        timestamps_s, smoothed_gap, max_gap_multiplier=max_gap_multiplier
    )


def find_distance_matched_pairs(
    ids: Sequence[object],
    distances: Sequence[float],
    speeds: Sequence[float],
    *,
    max_distance_difference: float,
    min_speed_ratio: float,
) -> pd.DataFrame:
    """Find similar-distance pairs with different speeds and flag non-reused pairs."""
    id_array = np.asarray(ids, dtype=object)
    distance_array = np.asarray(distances, dtype=float)
    speed_array = np.asarray(speeds, dtype=float)
    if any(array.ndim != 1 for array in (id_array, distance_array, speed_array)):
        raise ValueError("ids, distances, and speeds must be 1D.")
    if len({len(id_array), len(distance_array), len(speed_array)}) != 1:
        raise ValueError("ids, distances, and speeds must have equal lengths.")
    if max_distance_difference < 0:
        raise ValueError("max_distance_difference must be non-negative.")
    if min_speed_ratio < 1:
        raise ValueError("min_speed_ratio must be at least 1.")

    candidates = []
    valid = np.isfinite(distance_array) & np.isfinite(speed_array) & (speed_array > 0)
    valid_indices = np.flatnonzero(valid)
    for offset, first_index in enumerate(valid_indices):
        for second_index in valid_indices[offset + 1 :]:
            distance_difference = abs(
                distance_array[first_index] - distance_array[second_index]
            )
            speed_ratio = max(
                speed_array[first_index], speed_array[second_index]
            ) / min(speed_array[first_index], speed_array[second_index])
            if (
                distance_difference <= max_distance_difference
                and speed_ratio >= min_speed_ratio
            ):
                candidates.append(
                    {
                        "first_index": int(first_index),
                        "second_index": int(second_index),
                        "first_id": id_array[first_index],
                        "second_id": id_array[second_index],
                        "distance_difference": float(distance_difference),
                        "speed_ratio": float(speed_ratio),
                    }
                )

    columns = [
        "first_index",
        "second_index",
        "first_id",
        "second_id",
        "distance_difference",
        "speed_ratio",
        "selected",
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)

    pairs = pd.DataFrame(candidates).sort_values(
        ["distance_difference", "speed_ratio"], ascending=[True, False]
    ).reset_index(drop=True)
    pairs["selected"] = False
    used_indices: set[int] = set()
    for pair_index, pair in pairs.iterrows():
        indices = {int(pair["first_index"]), int(pair["second_index"])}
        if used_indices.isdisjoint(indices):
            pairs.at[pair_index, "selected"] = True
            used_indices.update(indices)
    return pairs[columns]


def calculate_braking_metrics(
    timestamps_s: Sequence[float],
    remaining_gap: Sequence[float],
    speed: Sequence[float],
    acceleration: Sequence[float],
    onset_index: int,
    arrival_index: int,
    *,
    closure_speed_floor: float = 0.0,
    smoothing_sigma_s: float = 0.0,
    max_gap_multiplier: float = 3.0,
) -> BrakingMetrics:
    """Calculate onset and period metrics for one detected braking event."""
    time = np.asarray(timestamps_s, dtype=float)
    gap = np.asarray(remaining_gap, dtype=float)
    speed_array = np.asarray(speed, dtype=float)
    accel = np.asarray(acceleration, dtype=float)
    if any(array.ndim != 1 for array in (time, gap, speed_array, accel)):
        raise ValueError("All metric inputs must be 1D.")
    if len({len(time), len(gap), len(speed_array), len(accel)}) != 1:
        raise ValueError("All metric inputs must have equal lengths.")
    if not 0 <= onset_index <= arrival_index < len(time):
        raise IndexError("Event indices must satisfy 0 <= onset <= arrival < length.")
    if closure_speed_floor < 0:
        raise ValueError("closure_speed_floor must be non-negative.")

    smoothed_gap = smooth_time_series(
        time,
        gap,
        smoothing_sigma_s=smoothing_sigma_s,
        max_gap_multiplier=max_gap_multiplier,
    )
    closure_speed = -differentiate_time_series(
        time, smoothed_gap, max_gap_multiplier=max_gap_multiplier
    )
    onset_gap = float(smoothed_gap[onset_index])
    onset_closure = float(closure_speed[onset_index])
    tau_s = (
        onset_gap / onset_closure
        if np.isfinite(onset_gap)
        and onset_gap >= 0
        and np.isfinite(onset_closure)
        and onset_closure > closure_speed_floor
        else np.nan
    )

    event_slice = slice(onset_index, arrival_index + 1)
    event_gap = smoothed_gap[event_slice]
    event_accel = accel[event_slice]
    finite_accel = event_accel[np.isfinite(event_accel)]
    deceleration = np.maximum(-finite_accel, 0.0)
    distance_travelled = (
        float(np.sum(np.abs(np.diff(event_gap))))
        if np.isfinite(event_gap).all()
        else np.nan
    )

    return BrakingMetrics(
        onset_time_s=float(time[onset_index]),
        arrival_time_s=float(time[arrival_index]),
        time_to_arrival_s=float(time[arrival_index] - time[onset_index]),
        remaining_gap_at_onset=onset_gap,
        closure_speed_at_onset=onset_closure,
        tau_at_onset_s=float(tau_s),
        speed_at_onset=float(speed_array[onset_index]),
        acceleration_at_onset=float(accel[onset_index]),
        braking_duration_s=float(time[arrival_index] - time[onset_index]),
        speed_at_arrival=float(speed_array[arrival_index]),
        speed_reduction=float(speed_array[onset_index] - speed_array[arrival_index]),
        distance_travelled=distance_travelled,
        mean_deceleration=float(np.mean(deceleration)) if deceleration.size else np.nan,
        max_deceleration=float(np.max(deceleration)) if deceleration.size else np.nan,
        negative_acceleration_fraction=(
            float(np.mean(finite_accel < 0)) if finite_accel.size else np.nan
        ),
    )


def detect_braking_period(
    timestamps_s: Sequence[float],
    acceleration: Sequence[float],
    final_arm_mask: Sequence[bool],
    arrival_index: int,
    *,
    min_duration_s: float = 0.20,
    max_interruption_s: float = 0.05,
    min_negative_fraction: float = 0.80,
) -> BrakingPeriod | None:
    """Detect the final sustained negative-acceleration period before arrival.

    Acceleration should be derived from a smoothed speed signal. The final-arm
    mask and arrival index are supplied by dataset geometry rather than inferred
    here. Brief non-negative or missing interruptions can be bridged.
    """
    time = np.asarray(timestamps_s, dtype=float)
    accel = np.asarray(acceleration, dtype=float)
    final_arm = np.asarray(final_arm_mask, dtype=bool)

    if time.ndim != 1 or accel.ndim != 1 or final_arm.ndim != 1:
        raise ValueError("timestamps, acceleration, and final_arm_mask must be 1D.")
    if not (len(time) == len(accel) == len(final_arm)):
        raise ValueError("Input arrays must have equal lengths.")
    if len(time) == 0:
        raise ValueError("Input arrays must not be empty.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing.")
    if not 0 <= arrival_index < len(time):
        raise IndexError("arrival_index is outside the input arrays.")
    if not final_arm[arrival_index]:
        raise ValueError("arrival_index must lie on the final arm.")
    if min_duration_s < 0 or max_interruption_s < 0:
        raise ValueError("Duration parameters must be non-negative.")
    if not 0 < min_negative_fraction <= 1:
        raise ValueError("min_negative_fraction must be in (0, 1].")

    arm_start = arrival_index
    while arm_start > 0 and final_arm[arm_start - 1]:
        arm_start -= 1

    if arrival_index == arm_start:
        return None

    sample_interval_s = float(np.median(np.diff(time[arm_start : arrival_index + 1])))
    negative = np.isfinite(accel) & (accel < 0)
    candidate_start = arrival_index
    cursor = arrival_index

    while cursor >= arm_start:
        if negative[cursor]:
            candidate_start = cursor
            cursor -= 1
            continue

        interruption_end = cursor
        while cursor >= arm_start and not negative[cursor]:
            cursor -= 1
        interruption_start = cursor + 1
        interruption_duration_s = (
            time[interruption_end] - time[interruption_start] + sample_interval_s
        )
        if cursor < arm_start or interruption_duration_s > max_interruption_s:
            break
        candidate_start = interruption_start

    selected = slice(candidate_start, arrival_index + 1)
    duration_s = float(time[arrival_index] - time[candidate_start])
    negative_fraction = float(np.mean(negative[selected]))
    if duration_s < min_duration_s or negative_fraction < min_negative_fraction:
        return None

    return BrakingPeriod(
        start_index=int(candidate_start),
        end_index=int(arrival_index),
        start_time_s=float(time[candidate_start]),
        end_time_s=float(time[arrival_index]),
        duration_s=duration_s,
        negative_fraction=negative_fraction,
    )
