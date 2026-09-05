import numpy as np

from src.ecological_neuro.utils import (
    calculate_braking_metrics,
    calculate_closure_speed,
    detect_braking_period,
    differentiate_time_series,
    find_distance_matched_pairs,
    smooth_time_series,
)


def test_detects_final_sustained_deceleration() -> None:
    time = np.arange(0.0, 1.01, 0.01)
    acceleration = np.where(time < 0.60, 2.0, -3.0)
    final_arm = time >= 0.20

    event = detect_braking_period(time, acceleration, final_arm, arrival_index=100)

    assert event is not None
    assert event.start_time_s == 0.60
    assert event.end_time_s == 1.00


def test_bridges_only_brief_nonnegative_interruptions() -> None:
    time = np.arange(0.0, 1.01, 0.01)
    acceleration = np.where(time < 0.50, 1.0, -2.0)
    acceleration[70:73] = 0.5
    final_arm = np.ones(time.shape, dtype=bool)

    event = detect_braking_period(time, acceleration, final_arm, arrival_index=100)

    assert event is not None
    assert event.start_time_s == 0.50
    assert event.negative_fraction > 0.90


def test_rejects_a_short_final_deceleration() -> None:
    time = np.arange(0.0, 1.01, 0.01)
    acceleration = np.where(time < 0.90, 1.0, -2.0)
    final_arm = np.ones(time.shape, dtype=bool)

    event = detect_braking_period(time, acceleration, final_arm, arrival_index=100)

    assert event is None


def test_calculates_signed_closure_speed() -> None:
    time = np.linspace(0.0, 2.0, 21)
    remaining_gap = 10.0 - 5.0 * time

    closure_speed = calculate_closure_speed(time, remaining_gap)

    np.testing.assert_allclose(closure_speed, 5.0)


def test_closure_speed_uses_the_common_smoothing_pipeline() -> None:
    time = np.linspace(0.0, 2.0, 101)
    remaining_gap = 10.0 - 5.0 * time + 0.2 * np.sin(40.0 * time)
    smoothed_gap = smooth_time_series(
        time, remaining_gap, smoothing_sigma_s=0.10
    )

    closure_speed = calculate_closure_speed(
        time, remaining_gap, smoothing_sigma_s=0.10
    )

    expected = -differentiate_time_series(time, smoothed_gap)
    np.testing.assert_allclose(closure_speed, expected, equal_nan=True)


def test_calculates_braking_point_and_period_metrics() -> None:
    time = np.linspace(0.0, 2.0, 21)
    remaining_gap = 10.0 - 5.0 * time
    speed = 10.0 - 3.0 * time
    acceleration = np.full(time.shape, -3.0)

    metrics = calculate_braking_metrics(
        time,
        remaining_gap,
        speed,
        acceleration,
        onset_index=5,
        arrival_index=20,
        closure_speed_floor=1.0,
    )

    assert np.isclose(metrics.time_to_arrival_s, 1.5)
    assert np.isclose(metrics.remaining_gap_at_onset, 7.5)
    assert np.isclose(metrics.closure_speed_at_onset, 5.0)
    assert np.isclose(metrics.tau_at_onset_s, 1.5)
    assert np.isclose(metrics.speed_reduction, 4.5)
    assert np.isclose(metrics.distance_travelled, 7.5)
    assert np.isclose(metrics.mean_deceleration, 3.0)
    assert np.isclose(metrics.max_deceleration, 3.0)
    assert np.isclose(metrics.negative_acceleration_fraction, 1.0)


def test_finds_distance_matched_speed_contrasts_without_reusing_samples() -> None:
    pairs = find_distance_matched_pairs(
        ids=["a", "b", "c", "d"],
        distances=[10.0, 10.5, 11.0, 20.0],
        speeds=[100.0, 90.0, 160.0, 200.0],
        max_distance_difference=1.0,
        min_speed_ratio=1.5,
    )

    assert len(pairs) == 2
    assert pairs["selected"].sum() == 1
    selected = pairs[pairs["selected"]].iloc[0]
    assert {selected["first_id"], selected["second_id"]} == {"b", "c"}
