"""Synthetic checks for causality, horizon labels and group-isolated forecasts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.ecological_neuro.forecasting_utils import (
    ForecastConfig, causal_reach_features, build_forecast_table,
    score_forecasts, evaluate_forecast_holdout, paired_forecast_scores,
    fit_forecast_ridge, predict_forecast_ridge,
)


def record(group_id=0, dt=0.01, origin=0.0, speed=40.0, event=0.7):
    time = np.arange(0, 1 + dt/2, dt)
    return dict(group_id=group_id, time_s=time+origin,
                position=np.column_stack((speed*time, np.zeros(len(time)))),
                target=np.array([100., 0.]), event_time_s=event+origin)


def test_causal_features_are_invariant_to_future_samples_and_prefix_length():
    r = record(dt=0.005)
    config = ForecastConfig()
    full = causal_reach_features(r['time_s'], r['position'], r['target'], config=config)
    stop = 85
    prefix = causal_reach_features(r['time_s'][:stop], r['position'][:stop], r['target'], config=config)
    pd.testing.assert_frame_equal(full.iloc[:stop], prefix)
    changed = r['position'].copy()
    changed[stop:] += 1000
    altered = causal_reach_features(r['time_s'], changed, r['target'], config=config)
    pd.testing.assert_frame_equal(full.iloc[:stop], altered.iloc[:stop])
    first = full.loc[full.decision_sample].iloc[0]
    assert first.time_s >= config.warmup_s + config.movement_confirmation_s - 1e-12
    assert first.elapsed_time_s == 0


def test_horizon_labels_boundaries_and_censoring():
    config = ForecastConfig()
    r = record()
    data, manifest = build_forecast_table([r], config=config)
    assert manifest.included.all()
    assert (data.time_s < r['event_time_s']).all()
    np.testing.assert_array_equal(data.event_within_horizon, data.time_s >= 0.6-1e-12)
    r['event_time_s'] = None
    censored, _ = build_forecast_table([r], config=config)
    assert not censored.event_within_horizon.any()
    assert (censored.time_s + config.horizon_s <= 1+1e-12).all()


def test_whole_reach_tau_exclusion_and_sampling_gap_report():
    config = ForecastConfig()
    r = record()
    r['position'][:, 1] = np.arange(len(r['time_s'])) * 3  # moving rapidly away
    data, manifest = build_forecast_table([r], config=config)
    assert data.empty
    assert manifest.reason.iloc[0] == 'undefined_tau_in_at_risk_interval'
    r = record()
    r['time_s'][40:] += .05
    r['event_time_s'] += .05
    data, manifest = build_forecast_table([r], config=config)
    assert data.empty
    assert manifest.reason.iloc[0] == 'sampling_gap_before_event'


def test_coordinates_clock_units_and_irregular_sampling():
    config = ForecastConfig()
    r = record(dt=.005)
    base = causal_reach_features(r['time_s'], r['position'], r['target'], config=config)
    rotation = np.array([[0., -1.], [1., 0.]])
    rotated = causal_reach_features(r['time_s'], r['position'] @ rotation + 7,
                                     r['target'] @ rotation + 7, config=config)
    np.testing.assert_allclose(base.distance, rotated.distance)
    np.testing.assert_allclose(base.tau_s, rotated.tau_s, equal_nan=True)
    shifted = causal_reach_features(r['time_s']+100, r['position'], r['target'], config=config)
    np.testing.assert_allclose(base.closure_speed, shifted.closure_speed, equal_nan=True, atol=1e-9)
    meters = causal_reach_features(r['time_s'], r['position']/1000, r['target']/1000,
        config=replace(config, spatial_unit='m', movement_speed_floor=.025, closure_speed_floor=.010))
    np.testing.assert_allclose(base.tau_s, meters.tau_s, equal_nan=True)
    irregular = np.cumsum(np.resize([.003, .007], 100))
    features = causal_reach_features(irregular, np.column_stack((40*irregular, irregular*0)),
                                     r['target'], config=config)
    np.testing.assert_allclose(features.closure_speed.iloc[1:], 40, atol=1e-9)


def test_scores_are_forecast_scores_not_truncated_event_times():
    data = pd.DataFrame({'group_id': [1, 1, 2], 'event_within_horizon': [False, True, True]})
    scores = score_forecasts(data, [.2, .8, .5])
    assert np.isclose(scores.log_loss.iloc[0], -np.log(.8))
    assert np.isclose(scores.brier_score.iloc[0], .04)
    assert 'predicted_time_s' not in scores
    with pytest.raises(ValueError):
        score_forecasts(data, [-1, .8, .5])


def test_nested_forecasting_and_paired_comparisons_use_same_test_reaches():
    records = [record(i, speed=30+i*2, event=.55+i*.025) for i in range(12)]
    data, _ = build_forecast_table(records, config=ForecastConfig())
    terms = {'time': [('elapsed_time_s',)], 'distance': [('elapsed_time_s',), ('distance',)]}
    scores, predictions, tuning = evaluate_forecast_holdout(
        data, terms, outer_splits=2, inner_splits=1, alpha_grid=(.01,), random_seed=3)
    paired = paired_forecast_scores(scores, baseline='time')
    assert len(paired) == len(scores)//2
    assert np.isfinite(scores[['log_loss', 'brier_score']]).all().all()
    assert (predictions.time_s < predictions.event_time_s).all()
    assert tuning.alpha.eq(.01).all()
    for _, group in predictions.groupby('split_id'):
        assert set(group.loc[group.model.eq('time'), 'group_id']) == set(group.loc[group.model.eq('distance'), 'group_id'])
    # Test values cannot alter training knots/standardization.
    train = data.loc[data.group_id < 8]
    fitted = fit_forecast_ridge(train, terms['distance'], alpha=.01)
    original = predict_forecast_ridge(fitted, data.iloc[:3])
    extreme = data.iloc[:3].copy()
    extreme['distance'] *= 1000
    predict_forecast_ridge(fitted, extreme)
    np.testing.assert_allclose(original, predict_forecast_ridge(fitted, data.iloc[:3]))
