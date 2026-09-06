"""Small, dataset-independent analysis utilities."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


__all__ = [
    "BrakingMetrics",
    "BrakingPeriod",
    "TerminalCorrection",
    "calculate_braking_metrics",
    "calculate_closure_speed",
    "discrete_hazard_event_distribution",
    "detect_braking_period",
    "detect_terminal_correction",
    "differentiate_time_series",
    "evaluate_repeated_hazard_holdout",
    "find_distance_matched_pairs",
    "fit_winsorized_spline_state",
    "hazard_calibration_table",
    "make_group_holdout_splits",
    "mean_discrete_hazard_event_distribution",
    "score_discrete_hazard_predictions",
    "smooth_time_series",
    "smoothed_event_mass_mode",
    "summarize_paired_hazard_scores",
    "transform_winsorized_natural_cubic_spline",
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


@dataclass(frozen=True)
class TerminalCorrection:
    """A spatially supported turn near the end of a movement path."""

    onset_index: int
    turn_index: int
    terminal_start_index: int
    movement_end_index: int
    onset_time_s: float
    turn_time_s: float
    signed_heading_change_deg: float
    path_fraction: float


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


def _signed_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the signed planar angle from one vector to another."""
    cross = first[0] * second[1] - first[1] * second[0]
    return float(np.degrees(np.arctan2(cross, np.dot(first, second))))


def detect_terminal_correction(
    timestamps_s: Sequence[float],
    position: Sequence[Sequence[float]],
    speed: Sequence[float],
    movement_onset_index: int,
    *,
    terminal_path_fraction: float = 0.10,
    movement_speed_floor: float = 25.0,
    smoothing_sigma_s: float = 0.05,
    incoming_arc_length: float = 8.0,
    outgoing_arc_length: float = 4.0,
    onset_tangent_arc_length: float = 3.0,
    resample_step: float = 0.25,
    min_heading_change_deg: float = 45.0,
    onset_heading_change_deg: float = 10.0,
    max_gap_multiplier: float = 3.0,
) -> TerminalCorrection | None:
    """Detect a sharp, spatially supported turn near movement-path completion.

    The terminal window is a fraction of the realized movement path, ending at
    the last sample above ``movement_speed_floor``. This prevents post-arrival
    dwell and unstable headings at near-zero speed from creating false turns.
    """
    time = np.asarray(timestamps_s, dtype=float)
    coordinates = np.asarray(position, dtype=float)
    speed_array = np.asarray(speed, dtype=float)
    if time.ndim != 1 or speed_array.ndim != 1:
        raise ValueError("timestamps_s and speed must be 1D.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("position must have shape (samples, 2).")
    if not (len(time) == len(coordinates) == len(speed_array)) or len(time) == 0:
        raise ValueError("Input arrays must have equal, non-zero lengths.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing.")
    if not 0 <= movement_onset_index < len(time):
        raise IndexError("movement_onset_index is outside the input arrays.")
    if not 0 < terminal_path_fraction <= 1:
        raise ValueError("terminal_path_fraction must be in (0, 1].")
    positive_parameters = (
        movement_speed_floor,
        incoming_arc_length,
        outgoing_arc_length,
        onset_tangent_arc_length,
        resample_step,
        min_heading_change_deg,
        onset_heading_change_deg,
    )
    if any(value <= 0 for value in positive_parameters):
        raise ValueError("Speed, distance, and angle parameters must be positive.")
    if onset_tangent_arc_length > incoming_arc_length:
        raise ValueError("onset_tangent_arc_length cannot exceed incoming_arc_length.")

    moving = np.flatnonzero(
        (np.arange(len(time)) >= movement_onset_index)
        & np.isfinite(speed_array)
        & (speed_array >= movement_speed_floor)
    )
    if moving.size == 0:
        return None
    movement_end_index = min(int(moving[-1]) + 1, len(time) - 1)

    smoothed = np.column_stack(
        [
            smooth_time_series(
                time,
                coordinates[:, axis],
                smoothing_sigma_s=smoothing_sigma_s,
                max_gap_multiplier=max_gap_multiplier,
            )
            for axis in range(2)
        ]
    )
    movement_slice = slice(movement_onset_index, movement_end_index + 1)
    movement_position = smoothed[movement_slice]
    movement_time = time[movement_slice]
    finite = np.all(np.isfinite(movement_position), axis=1)
    if len(movement_time) < 3 or not finite.all():
        return None

    cumulative_distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(movement_position, axis=0), axis=1)))
    )
    distinct = np.concatenate(([True], np.diff(cumulative_distance) > 1e-9))
    cumulative_distance = cumulative_distance[distinct]
    movement_position = movement_position[distinct]
    movement_time = movement_time[distinct]
    required_distance = incoming_arc_length + outgoing_arc_length
    if cumulative_distance[-1] < required_distance:
        return None

    path_distance = np.arange(
        0.0, cumulative_distance[-1] + 0.5 * resample_step, resample_step
    )
    path_distance = path_distance[path_distance <= cumulative_distance[-1]]
    path_position = np.column_stack(
        [
            np.interp(path_distance, cumulative_distance, movement_position[:, axis])
            for axis in range(2)
        ]
    )
    path_time = np.interp(path_distance, cumulative_distance, movement_time)
    incoming_steps = int(np.ceil(incoming_arc_length / resample_step))
    outgoing_steps = int(np.ceil(outgoing_arc_length / resample_step))
    tangent_steps = int(np.ceil(onset_tangent_arc_length / resample_step))
    terminal_start_distance = (1.0 - terminal_path_fraction) * path_distance[-1]

    candidates: list[tuple[float, float, int]] = []
    for index in range(incoming_steps, len(path_distance) - outgoing_steps):
        if path_distance[index] < terminal_start_distance:
            continue
        incoming = path_position[index] - path_position[index - incoming_steps]
        outgoing = path_position[index + outgoing_steps] - path_position[index]
        heading_change = _signed_angle_degrees(incoming, outgoing)
        candidates.append((abs(heading_change), heading_change, index))
    if not candidates:
        return None

    _, signed_heading_change, turn_index = max(candidates)
    if abs(signed_heading_change) < min_heading_change_deg:
        return None

    search_start = turn_index - incoming_steps
    reference_heading = (
        path_position[search_start + tangent_steps] - path_position[search_start]
    )
    turn_direction = np.sign(signed_heading_change)
    onset_index = turn_index
    for index in range(search_start, turn_index + 1):
        local_end = min(index + tangent_steps, len(path_position) - 1)
        local_heading = path_position[local_end] - path_position[index]
        deviation = _signed_angle_degrees(reference_heading, local_heading)
        if turn_direction * deviation >= onset_heading_change_deg:
            onset_index = index
            break

    terminal_path_index = int(np.searchsorted(path_distance, terminal_start_distance))

    def original_index(resampled_index: int) -> int:
        return int(np.searchsorted(time, path_time[resampled_index]))

    original_onset = original_index(onset_index)
    original_turn = original_index(turn_index)
    original_terminal_start = original_index(terminal_path_index)
    return TerminalCorrection(
        onset_index=original_onset,
        turn_index=original_turn,
        terminal_start_index=original_terminal_start,
        movement_end_index=movement_end_index,
        onset_time_s=float(time[original_onset]),
        turn_time_s=float(time[original_turn]),
        signed_heading_change_deg=float(signed_heading_change),
        path_fraction=float(path_distance[turn_index] / path_distance[-1]),
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


def make_group_holdout_splits(
    groups: Sequence[object],
    *,
    n_splits: int,
    test_fraction: float,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create repeated random train/test splits that keep complete groups together."""
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups must be a non-empty 1D sequence.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1).")

    unique_groups = pd.unique(group_array)
    if unique_groups.size < 2:
        raise ValueError("At least two unique groups are required for a holdout split.")
    n_test_groups = int(np.ceil(test_fraction * unique_groups.size))
    n_test_groups = min(max(n_test_groups, 1), unique_groups.size - 1)

    generator = np.random.default_rng(random_seed)
    splits = []
    for _ in range(n_splits):
        test_groups = generator.choice(
            unique_groups, size=n_test_groups, replace=False
        )
        test_mask = np.isin(group_array, test_groups)
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return splits


def fit_winsorized_spline_state(
    values: Sequence[float],
    *,
    df: int,
    quantiles: tuple[float, float] = (0.005, 0.995),
) -> dict[str, float | tuple[float, ...]]:
    """Fit winsorization bounds and natural-spline knots from training values."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence.")
    if not np.isfinite(value_array).all():
        raise ValueError("values must be finite.")
    if df < 3:
        raise ValueError("df must be at least 3 for a natural cubic spline.")
    lower_quantile, upper_quantile = quantiles
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")

    lower_bound, upper_bound = np.quantile(value_array, quantiles)
    if not upper_bound > lower_bound:
        raise ValueError("Training values need non-zero range for a spline basis.")
    clipped = np.clip(value_array, lower_bound, upper_bound)
    knot_quantiles = np.linspace(0.0, 1.0, df)[1:-1]
    knots = tuple(float(value) for value in np.quantile(clipped, knot_quantiles))
    if len(set(knots)) != len(knots) or knots[0] <= lower_bound or knots[-1] >= upper_bound:
        raise ValueError("Training values do not support distinct interior spline knots.")
    return {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "knots": knots,
    }


def transform_winsorized_natural_cubic_spline(
    values: pd.Series,
    state: dict[str, float | tuple[float, ...]],
    *,
    name: str,
) -> pd.DataFrame:
    """Apply a training-fitted winsorized natural-cubic-spline transformation."""
    try:
        from patsy import dmatrix
    except ImportError as error:
        raise ImportError("Spline transforms require patsy.") from error

    lower_bound = float(state["lower_bound"])
    upper_bound = float(state["upper_bound"])
    knots = tuple(state["knots"])
    if values.ndim != 1:
        raise ValueError("values must be a pandas Series.")
    numeric_values = values.astype(float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("values must be finite.")
    clipped = numeric_values.clip(lower_bound, upper_bound)
    formula = (
        "cr(value, knots="
        f"{knots!r}, lower_bound={lower_bound!r}, upper_bound={upper_bound!r}) - 1"
    )
    basis = dmatrix(formula, {"value": clipped}, return_type="dataframe")
    basis.index = values.index
    basis.columns = [f"{name}_s{index}" for index in range(basis.shape[1])]
    return basis


def score_discrete_hazard_predictions(
    data: pd.DataFrame,
    hazard: Sequence[float],
    *,
    group_column: str,
    time_column: str,
    event_column: str,
    split_id: int,
    model_name: str,
    probability_floor: float = 1e-12,
    event_time_smoothing_sigma_s: float = 0.01,
) -> pd.DataFrame:
    """Score one discrete-time hazard prediction per complete held-out group.

    The event probability is the cumulative incidence predicted before the last
    observed at-risk bin. Predicted onset time is the mode of a lightly
    smoothed unconditional event-mass curve. Each group may contain zero or
    one observed event.
    """
    required_columns = {group_column, time_column, event_column}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"data is missing columns: {sorted(missing_columns)}")
    if not 0 < probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5).")
    if event_time_smoothing_sigma_s < 0:
        raise ValueError("event_time_smoothing_sigma_s must be non-negative.")

    hazard_array = np.asarray(hazard, dtype=float)
    if hazard_array.ndim != 1 or len(hazard_array) != len(data):
        raise ValueError("hazard must be a 1D sequence with one value per data row.")
    if not np.isfinite(hazard_array).all():
        raise ValueError("hazard must contain only finite values.")
    if not data[time_column].map(np.isfinite).all():
        raise ValueError("time_column must contain only finite values.")

    scored = data.loc[:, [group_column, time_column, event_column]].copy()
    scored["hazard"] = np.clip(hazard_array, probability_floor, 1 - probability_floor)
    scored["event"] = scored[event_column].astype(bool)
    scored = scored.sort_values([group_column, time_column], kind="stable")

    rows = []
    for group_id, group in scored.groupby(group_column, sort=False):
        event = group["event"].to_numpy(dtype=bool)
        if event.sum() > 1:
            raise ValueError(f"Group {group_id!r} contains more than one event.")
        probabilities = group["hazard"].to_numpy()
        time_s = group[time_column].to_numpy(dtype=float)
        event_mass, cumulative_event_probability = discrete_hazard_event_distribution(
            probabilities
        )
        event_probability = float(cumulative_event_probability[-1])
        observed_event = float(event.any())
        observed_time = float(time_s[event][0]) if event.any() else np.nan
        predicted_time, _ = smoothed_event_mass_mode(
            event_mass,
            time_s,
            smoothing_sigma_s=event_time_smoothing_sigma_s,
        )
        negative_log_likelihood = -float(
            np.sum(
                event * np.log(probabilities)
                + (~event) * np.log1p(-probabilities)
            )
        )
        rows.append(
            {
                "split_id": split_id,
                "model": model_name,
                "group_id": group_id,
                "n_at_risk_bins": len(group),
                "negative_log_likelihood": negative_log_likelihood,
                "predicted_event_probability": event_probability,
                "observed_event": observed_event,
                "calibration_error": abs(observed_event - event_probability),
                "predicted_time_s": predicted_time,
                "observed_time_s": observed_time,
                "absolute_time_error_s": (
                    abs(predicted_time - observed_time)
                    if event.any() and np.isfinite(predicted_time)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def discrete_hazard_event_distribution(
    hazard: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a discrete hazard into event mass and cumulative incidence."""
    hazard_array = np.asarray(hazard, dtype=float)
    if hazard_array.ndim != 1 or hazard_array.size == 0:
        raise ValueError("hazard must be a non-empty 1D sequence.")
    if not np.isfinite(hazard_array).all() or np.any(
        (hazard_array < 0) | (hazard_array > 1)
    ):
        raise ValueError("hazard must contain finite probabilities in [0, 1].")
    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - hazard_array[:-1])))
    event_mass = survival_before * hazard_array
    return event_mass, np.cumsum(event_mass)


def mean_discrete_hazard_event_distribution(
    hazards: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average event distributions from several discrete-hazard predictions.

    Each model's hazard is first converted to its unconditional event-mass
    distribution. Averaging raw hazards would be incorrect because the
    hazard-to-event-mass transformation includes survival from earlier bins.
    """
    hazard_array = np.asarray(hazards, dtype=float)
    if hazard_array.ndim != 2 or hazard_array.shape[0] == 0 or hazard_array.shape[1] == 0:
        raise ValueError("hazards must be a non-empty 2D array of probabilities.")
    if not np.isfinite(hazard_array).all() or np.any(
        (hazard_array < 0) | (hazard_array > 1)
    ):
        raise ValueError("hazards must contain finite probabilities in [0, 1].")
    survival_before = np.concatenate(
        (
            np.ones((hazard_array.shape[0], 1)),
            np.cumprod(1.0 - hazard_array[:, :-1], axis=1),
        ),
        axis=1,
    )
    mean_event_mass = np.mean(survival_before * hazard_array, axis=0)
    return mean_event_mass, np.cumsum(mean_event_mass)


def smoothed_event_mass_mode(
    event_mass: Sequence[float],
    time_s: Sequence[float],
    *,
    smoothing_sigma_s: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Return the MAP event time from a lightly smoothed event-mass curve.

    ``event_mass`` is the unconditional probability of an event in each
    discrete time bin, rather than the conditional hazard. Gaussian smoothing
    is expressed in seconds and uses the median bin width, so this function is
    intended for regularly sampled discrete-time hazard tables.
    """
    mass = np.asarray(event_mass, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if mass.ndim != 1 or mass.size == 0:
        raise ValueError("event_mass must be a non-empty 1D sequence.")
    if time.shape != mass.shape or not np.isfinite(time).all():
        raise ValueError("time_s must be finite and have one value per event-mass bin.")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("event_mass must contain finite, non-negative values.")
    if smoothing_sigma_s < 0:
        raise ValueError("smoothing_sigma_s must be non-negative.")
    if mass.size == 1 or smoothing_sigma_s == 0:
        smoothed_mass = mass.copy()
    else:
        intervals_s = np.diff(time)
        if np.any(intervals_s <= 0):
            raise ValueError("time_s must be strictly increasing within a group.")
        median_interval_s = float(np.median(intervals_s))
        smoothed_mass = gaussian_filter1d(
            mass,
            sigma=smoothing_sigma_s / median_interval_s,
            mode="nearest",
        )
    mode_index = int(np.argmax(smoothed_mass))
    return float(time[mode_index]), smoothed_mass


def hazard_calibration_table(
    event: Sequence[bool],
    hazard: Sequence[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed event rates in equal-count hazard bins."""
    event_array = np.asarray(event, dtype=bool)
    hazard_array = np.asarray(hazard, dtype=float)
    if event_array.ndim != 1 or hazard_array.ndim != 1:
        raise ValueError("event and hazard must be 1D.")
    if len(event_array) != len(hazard_array) or len(event_array) == 0:
        raise ValueError("event and hazard must have equal, non-zero lengths.")
    if not np.isfinite(hazard_array).all() or np.any((hazard_array < 0) | (hazard_array > 1)):
        raise ValueError("hazard must contain finite probabilities in [0, 1].")
    if not 1 <= n_bins <= len(event_array):
        raise ValueError("n_bins must be between 1 and the number of rows.")

    sorted_indices = np.argsort(hazard_array, kind="stable")
    bins = np.array_split(sorted_indices, n_bins)
    rows = []
    for bin_id, indices in enumerate(bins):
        predicted_rate = float(np.mean(hazard_array[indices]))
        observed_rate = float(np.mean(event_array[indices]))
        rows.append(
            {
                "bin_id": bin_id,
                "n_bins": len(indices),
                "mean_predicted_hazard": predicted_rate,
                "observed_event_rate": observed_rate,
                "absolute_calibration_gap": abs(observed_rate - predicted_rate),
            }
        )
    return pd.DataFrame(rows)


def summarize_paired_hazard_scores(
    scores: pd.DataFrame,
    *,
    baseline_model: str,
) -> pd.DataFrame:
    """Return per-group held-out score differences versus a common baseline.

    Negative differences mean lower error than the baseline for all returned
    metrics. Every compared model must have predictions for exactly the same
    split/group rows as the baseline.
    """
    required_columns = {
        "split_id",
        "model",
        "group_id",
        "negative_log_likelihood",
        "calibration_error",
        "absolute_time_error_s",
    }
    missing_columns = required_columns.difference(scores.columns)
    if missing_columns:
        raise ValueError(f"scores is missing columns: {sorted(missing_columns)}")
    if baseline_model not in set(scores["model"]):
        raise ValueError("baseline_model is not present in scores.")

    metric_columns = [
        "negative_log_likelihood",
        "calibration_error",
        "absolute_time_error_s",
    ]
    index_columns = ["split_id", "group_id"]
    baseline = scores.loc[scores["model"] == baseline_model, index_columns + metric_columns]
    if baseline.duplicated(index_columns).any():
        raise ValueError("Baseline scores must have one row per split and group.")
    baseline = baseline.set_index(index_columns)

    paired_rows = []
    for model_name, model_scores in scores.groupby("model", sort=False):
        if model_name == baseline_model:
            continue
        candidate = model_scores.loc[:, index_columns + metric_columns]
        if candidate.duplicated(index_columns).any():
            raise ValueError("Candidate scores must have one row per split and group.")
        candidate = candidate.set_index(index_columns)
        if not candidate.index.equals(baseline.index):
            raise ValueError("Candidate and baseline must use identical held-out groups.")
        differences = candidate - baseline
        differences = differences.reset_index()
        differences.insert(0, "baseline_model", baseline_model)
        differences.insert(0, "model", model_name)
        paired_rows.append(differences)
    return pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()


def evaluate_repeated_hazard_holdout(
    data: pd.DataFrame,
    *,
    group_column: str,
    time_column: str,
    event_column: str,
    model_names: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    fit_and_predict: Callable[[str, pd.DataFrame, pd.DataFrame], Sequence[float]],
    event_time_smoothing_sigma_s: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate hazard models on repeated, group-disjoint held-out samples.

    ``fit_and_predict`` receives one model name plus a training and held-out
    table, and must return one held-out hazard probability per test row. It is
    responsible for fitting all model preprocessing from the training table.
    The event-time summary uses the specified Gaussian smoothing scale.
    """
    if group_column not in data.columns:
        raise ValueError(f"data is missing group column {group_column!r}.")
    if not model_names:
        raise ValueError("model_names must not be empty.")
    if not splits:
        raise ValueError("splits must not be empty.")

    score_tables = []
    prediction_tables = []
    for split_id, (train_index, test_index) in enumerate(splits):
        train_data = data.iloc[np.asarray(train_index, dtype=int)].copy()
        test_data = data.iloc[np.asarray(test_index, dtype=int)].copy()
        if train_data.empty or test_data.empty:
            raise ValueError("Every holdout split must have non-empty train and test data.")
        train_groups = set(train_data[group_column])
        test_groups = set(test_data[group_column])
        if not train_groups.isdisjoint(test_groups):
            raise ValueError("Train and test data must contain disjoint groups.")

        for model_name in model_names:
            probabilities = np.asarray(
                fit_and_predict(model_name, train_data, test_data), dtype=float
            )
            if probabilities.shape != (len(test_data),):
                raise ValueError(
                    "fit_and_predict must return one probability for every test row."
                )
            score_tables.append(
                score_discrete_hazard_predictions(
                    test_data,
                    probabilities,
                    group_column=group_column,
                    time_column=time_column,
                    event_column=event_column,
                    split_id=split_id,
                    model_name=model_name,
                    event_time_smoothing_sigma_s=event_time_smoothing_sigma_s,
                )
            )
            prediction = test_data.loc[:, [group_column, time_column, event_column]].copy()
            prediction["split_id"] = split_id
            prediction["model"] = model_name
            prediction["hazard"] = probabilities
            prediction_tables.append(prediction)

    return (
        pd.concat(score_tables, ignore_index=True),
        pd.concat(prediction_tables, ignore_index=True),
    )
