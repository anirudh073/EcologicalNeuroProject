import numpy as np

from src.ecological_neuro.utils import (
    calculate_braking_metrics,
    calculate_closure_speed,
    discrete_hazard_event_distribution,
    detect_braking_period,
    detect_terminal_correction,
    differentiate_time_series,
    find_distance_matched_pairs,
    hazard_calibration_table,
    make_group_holdout_splits,
    mean_discrete_hazard_event_distribution,
    score_discrete_hazard_predictions,
    smooth_time_series,
    smoothed_event_mass_mode,
    summarize_paired_hazard_scores,
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


def test_detects_spatially_supported_terminal_correction() -> None:
    path_distance = np.linspace(0.0, 100.0, 401)
    time = path_distance / 100.0
    position = np.column_stack(
        [np.minimum(path_distance, 90.0), np.maximum(path_distance - 90.0, 0.0)]
    )
    speed = np.full(time.shape, 100.0)

    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    for transformed in (position, position @ rotation.T + [12.0, -7.0]):
        event = detect_terminal_correction(
            time, transformed, speed, movement_onset_index=0, smoothing_sigma_s=0.0
        )

        assert event is not None
        assert 0.85 <= event.path_fraction <= 0.95
        assert event.signed_heading_change_deg > 80.0
        assert event.onset_time_s < 0.95


def test_terminal_correction_rejects_straight_path_and_stationary_jitter() -> None:
    time = np.linspace(0.0, 1.0, 201)
    position = np.column_stack([100.0 * time, np.zeros_like(time)])
    speed = np.full(time.shape, 100.0)
    speed[160:] = 0.0
    position[160:, 1] = 0.5 * np.sin(np.arange(len(time) - 160))

    event = detect_terminal_correction(
        time, position, speed, movement_onset_index=0, smoothing_sigma_s=0.0
    )

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


def test_group_holdout_splits_preserve_complete_groups() -> None:
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    splits = make_group_holdout_splits(
        groups, n_splits=3, test_fraction=0.5, random_seed=4
    )

    assert len(splits) == 3
    for train_index, test_index in splits:
        train_groups = {groups[index] for index in train_index}
        test_groups = {groups[index] for index in test_index}
        assert train_groups.isdisjoint(test_groups)
        assert train_groups | test_groups == {"a", "b", "c", "d"}


def test_scores_discrete_hazards_and_pairs_candidates_with_baseline() -> None:
    import pandas as pd

    data = pd.DataFrame(
        {
            "reach": [1, 1, 1, 2, 2, 2],
            "time_s": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "event": [False, False, True, False, True, False],
        }
    )
    baseline_scores = score_discrete_hazard_predictions(
        data,
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        group_column="reach",
        time_column="time_s",
        event_column="event",
        split_id=0,
        model_name="time",
    )
    candidate_scores = score_discrete_hazard_predictions(
        data,
        [0.05, 0.05, 0.8, 0.05, 0.8, 0.05],
        group_column="reach",
        time_column="time_s",
        event_column="event",
        split_id=0,
        model_name="tau",
    )

    assert np.isclose(candidate_scores.loc[0, "predicted_time_s"], 0.2)
    assert (candidate_scores["negative_log_likelihood"] < baseline_scores["negative_log_likelihood"]).all()
    assert (candidate_scores["absolute_time_error_s"] < baseline_scores["absolute_time_error_s"]).all()

    paired = summarize_paired_hazard_scores(
        pd.concat([baseline_scores, candidate_scores], ignore_index=True),
        baseline_model="time",
    )
    assert (paired["negative_log_likelihood"] < 0).all()
    assert (paired["absolute_time_error_s"] < 0).all()


def test_hazard_calibration_table_uses_equal_count_bins() -> None:
    table = hazard_calibration_table(
        event=[False, False, True, True],
        hazard=[0.05, 0.10, 0.80, 0.90],
        n_bins=2,
    )

    assert table["n_bins"].tolist() == [2, 2]
    assert np.isclose(table.loc[0, "observed_event_rate"], 0.0)
    assert np.isclose(table.loc[1, "observed_event_rate"], 1.0)


def test_discrete_hazard_event_distribution_respects_survival() -> None:
    event_mass, cumulative_probability = discrete_hazard_event_distribution(
        [0.2, 0.5, 0.5]
    )

    np.testing.assert_allclose(event_mass, [0.2, 0.4, 0.2])
    np.testing.assert_allclose(cumulative_probability, [0.2, 0.6, 0.8])


def test_mean_event_distribution_converts_each_hazard_before_averaging() -> None:
    event_mass, cumulative_probability = mean_discrete_hazard_event_distribution(
        [[0.2, 0.5, 0.5], [0.4, 0.0, 0.0]]
    )

    np.testing.assert_allclose(event_mass, [0.3, 0.2, 0.1])
    np.testing.assert_allclose(cumulative_probability, [0.3, 0.5, 0.6])


def test_smoothed_event_mass_mode_uses_event_mass_not_conditional_hazard() -> None:
    time_s = np.arange(5) * 0.001
    event_mass = np.array([0.01, 0.02, 0.08, 0.06, 0.01])

    predicted_time_s, smoothed_mass = smoothed_event_mass_mode(
        event_mass, time_s, smoothing_sigma_s=0.001
    )

    assert np.isclose(predicted_time_s, time_s[np.argmax(smoothed_mass)])
    assert np.isclose(predicted_time_s, 0.002)
