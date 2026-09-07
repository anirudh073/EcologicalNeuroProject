"""Past-only kinematics and grouped, fixed-horizon event forecasting.

Canonical reach records contain group_id, time_s (strictly increasing seconds),
position (samples x dimensions), target (dimensions), and event_time_s (same
clock, or None for no observed event). Spatial values and speed floors share
the caller's configured physical unit. Core code never reads source files.
"""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import qr

from .utils import (
    fit_winsorized_spline_state,
    transform_winsorized_natural_cubic_spline,
    make_group_holdout_splits,
)


@dataclass(frozen=True)
class ForecastConfig:
    """Frozen acquisition-independent forecast settings; spatial units supplied."""

    horizon_s: float = 0.1
    decision_interval_s: float = 0.01
    smoothing_time_constant_s: float = 0.1
    warmup_s: float = 0.01
    movement_speed_floor: float = 25.0
    movement_confirmation_s: float = 0.05
    closure_speed_floor: float = 10.0
    max_sample_gap_s: float = 0.02
    spatial_unit: str = "mm"

    def __post_init__(self) -> None:
        for name in (
            "horizon_s", "decision_interval_s", "smoothing_time_constant_s",
            "warmup_s", "movement_speed_floor", "movement_confirmation_s",
            "closure_speed_floor", "max_sample_gap_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not self.spatial_unit:
            raise ValueError("spatial_unit must be explicit.")


def causal_reach_features(
    time_s: Sequence[float], position: np.ndarray, target: Sequence[float],
    *, config: ForecastConfig,
) -> pd.DataFrame:
    """Compute prefix-invariant features with backward differences and EMA.

    A derivative at t uses position at t and its previous sample. Exponential
    averaging uses the actual interval, never a full-trace sampling estimate.
    Movement is confirmed only AFTER a sustained above-floor speed interval.
    Progress is referenced to the gap at this confirmation, available then.
    Missing samples and gaps restart warmup and movement confirmation.
    """
    time = np.asarray(time_s, dtype=float)
    coordinates = np.asarray(position, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("time_s must contain at least two finite timestamps.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    if coordinates.ndim != 2 or coordinates.shape[0] != len(time):
        raise ValueError("position must have shape (samples, dimensions).")
    if target_array.shape != (coordinates.shape[1],) or not np.isfinite(target_array).all():
        raise ValueError("target must be a finite vector matching position dimensions.")
    distance = np.linalg.norm(coordinates - target_array, axis=1)
    speed = np.full(len(time), np.nan)
    closure = np.full(len(time), np.nan)
    elapsed = np.full(len(time), np.nan)
    progress = np.full(len(time), np.nan)
    decision = np.zeros(len(time), dtype=bool)
    run_start = time[0]
    sustained_start = None
    confirmed_time = None
    start_gap = None
    next_decision = None
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        if dt > config.max_sample_gap_s or not np.isfinite(coordinates[i - 1:i + 1]).all():
            run_start = time[i]
            sustained_start = confirmed_time = start_gap = next_decision = None
            continue
        raw_speed = np.linalg.norm(coordinates[i] - coordinates[i - 1]) / dt
        raw_closure = (distance[i - 1] - distance[i]) / dt
        weight = -np.expm1(-dt / config.smoothing_time_constant_s)
        if np.isfinite(speed[i - 1]):
            speed[i] = speed[i - 1] + weight * (raw_speed - speed[i - 1])
            closure[i] = closure[i - 1] + weight * (raw_closure - closure[i - 1])
        else:
            speed[i], closure[i] = raw_speed, raw_closure
        if confirmed_time is None:
            if time[i] - run_start < config.warmup_s or speed[i] < config.movement_speed_floor:
                sustained_start = None
                continue
            if sustained_start is None:
                sustained_start = time[i]
            if time[i] - sustained_start + 1e-12 < config.movement_confirmation_s:
                continue
            if distance[i] <= 0:
                continue
            confirmed_time, start_gap = time[i], distance[i]
            next_decision = time[i]
        elapsed[i] = time[i] - confirmed_time
        progress[i] = 1 - distance[i] / start_gap
        if time[i] + 1e-12 >= next_decision:
            decision[i] = True
            next_decision += config.decision_interval_s
    valid_tau = (distance > 0) & (closure > config.closure_speed_floor)
    tau = np.divide(distance, closure, out=np.full(len(time), np.nan), where=valid_tau)
    required = np.divide(closure ** 2, 2 * distance,
                         out=np.full(len(time), np.nan), where=valid_tau)
    return pd.DataFrame({
        "time_s": time, "distance": distance, "speed": speed,
        "closure_speed": closure, "elapsed_time_s": elapsed,
        "progress": progress, "tau_s": tau, "required_deceleration": required,
        "decision_sample": decision,
    })


def build_forecast_table(
    records: Sequence[Mapping], *, config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label decisions with event in (t, t+h]; return whole-reach exclusions.

    Known event times are used exclusively for labels and at-risk eligibility.
    For censored records, discard windows whose outcome is not observable.
    Whole-reach tau eligibility is checked at every native at-risk sample
    after causal movement confirmation, before selecting decision samples.
    This remains a retrospectively selected cohort, not a deployment filter.
    """
    tables, manifest = [], []
    seen = set()
    for record in records:
        group_id = record["group_id"]
        if group_id in seen:
            raise ValueError("group_id must be unique per reach record.")
        seen.add(group_id)
        features = causal_reach_features(record["time_s"], record["position"],
                                         record["target"], config=config)
        event_time = record.get("event_time_s")
        observed = event_time is not None and np.isfinite(event_time)
        if event_time is not None and not observed:
            raise ValueError("Use None for censoring, or a finite event_time_s.")
        if observed and not features.time_s.iloc[0] <= event_time <= features.time_s.iloc[-1]:
            raise ValueError("event_time_s must be within the observed reach.")
        at_risk = features.time_s < event_time if observed else pd.Series(True, index=features.index)
        active = at_risk & features.elapsed_time_s.notna()
        reason = "included"
        if not active.any():
            reason = "no_confirmed_movement_before_event"
        elif not np.isfinite(features.loc[active, "tau_s"]).all():
            reason = "undefined_tau_in_at_risk_interval"
        elif not np.isfinite(np.asarray(record["position"])[np.asarray(at_risk)]).all():
            reason = "missing_position_before_event"
        elif np.any(np.diff(features.loc[at_risk, "time_s"]) > config.max_sample_gap_s):
            reason = "sampling_gap_before_event"
        decisions = features.loc[active & features.decision_sample].copy()
        if not observed:
            decisions = decisions.loc[
                decisions.time_s + config.horizon_s <= features.time_s.iloc[-1] + 1e-12
            ]
        if reason == "included" and decisions.empty:
            reason = "no_complete_forecast_windows"
        if reason == "included":
            decisions["event_within_horizon"] = (
                (event_time - decisions.time_s <= config.horizon_s + 1e-12)
                if observed else False
            )
            decisions["group_id"] = group_id
            decisions["event_time_s"] = event_time if observed else np.nan
            decisions["horizon_s"] = config.horizon_s
            tables.append(decisions)
        manifest.append({
            "group_id": group_id, "included": reason == "included", "reason": reason,
            "input_samples": len(features),
            "decision_samples": len(decisions) if reason == "included" else 0,
            "positive_windows": int(decisions.event_within_horizon.sum()) if reason == "included" else 0,
        })
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return table, pd.DataFrame(manifest)


def score_forecasts(data: pd.DataFrame, probability: Sequence[float]) -> pd.DataFrame:
    """Mean binary log loss and Brier score within each reach, not event NLL."""
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(data),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Supply one finite probability in [0, 1] per decision.")
    y = data.event_within_horizon.to_numpy(dtype=float)
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    scores = pd.DataFrame({
        "group_id": data.group_id.to_numpy(),
        "log_loss": -(y * np.log(safe) + (1-y) * np.log1p(-safe)),
        "brier_score": (p-y) ** 2,
    })
    return scores.groupby("group_id", sort=False).agg(
        log_loss=("log_loss", "mean"), brier_score=("brier_score", "mean"),
        n_decisions=("log_loss", "size"),
    ).reset_index()


def _forecast_design(data: pd.DataFrame, terms: Sequence[Sequence[str]], states: dict) -> np.ndarray:
    columns = [np.ones((len(data), 1))]
    for term in terms:
        basis = [transform_winsorized_natural_cubic_spline(data[name], states[name], name=name).to_numpy()
                 for name in term]
        values = basis[0]
        for other in basis[1:]:
            values = (values[:, :, None] * other[:, None, :]).reshape(len(data), -1)
        columns.append(values)
    return np.column_stack(columns)


def fit_forecast_ridge(
    train: pd.DataFrame, terms: Sequence[Sequence[str]], *, alpha: float,
    spline_df: int = 4,
) -> dict:
    """Train-only spline knots, rank selection and scaling; equal reach weight."""
    import statsmodels.api as sm

    names = dict.fromkeys(name for term in terms for name in term)
    states = {name: fit_winsorized_spline_state(train[name], df=spline_df) for name in names}
    design = _forecast_design(train, terms, states)
    center = design.mean(axis=0)
    center[0] = 0
    _, r, pivot = qr(design - center, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    keep = np.sort(pivot[:np.sum(diagonal > 1e-8 * diagonal.max())])
    scale = design.std(axis=0)
    scale[scale == 0] = 1
    exog = ((design - center) / scale)[:, keep]
    penalty = np.full(len(keep), alpha)
    penalty[keep == 0] = 0
    weights = 1 / train.groupby("group_id").group_id.transform("size").to_numpy()
    weights /= weights.mean()
    result = sm.GLM(train.event_within_horizon.astype(float).to_numpy(), exog,
                    family=sm.families.Binomial(), freq_weights=weights).fit_regularized(
        alpha=penalty, L1_wt=0.0, maxiter=1000, cnvrg_tol=1e-10,
    )
    return dict(result=result, terms=terms, states=states, center=center, scale=scale, keep=keep)


def predict_forecast_ridge(fitted: dict, test: pd.DataFrame) -> np.ndarray:
    """Apply a fixed training transform to new decisions."""
    design = _forecast_design(test, fitted["terms"], fitted["states"])
    exog = ((design - fitted["center"]) / fitted["scale"])[:, fitted["keep"]]
    return np.asarray(fitted["result"].predict(exog))


def evaluate_forecast_holdout(
    data: pd.DataFrame, model_terms: Mapping[str, Sequence[Sequence[str]]], *,
    outer_splits: int = 10, inner_splits: int = 3, test_fraction: float = 0.2,
    alpha_grid: Sequence[float] = (0.001, 0.01, 0.1), random_seed: int = 20260906,
    spline_df: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested whole-reach validation of fixed-horizon ridge forecasts.

    All candidates share outer and inner groups. Hyperparameters minimize
    mean reach log loss. Repeated windows/splits are not independent trials.
    """
    splits = make_group_holdout_splits(data.group_id, n_splits=outer_splits,
                                       test_fraction=test_fraction, random_seed=random_seed)
    score_tables, prediction_tables, tuning = [], [], []
    for split_id, (train_idx, test_idx) in enumerate(splits):
        train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
        inner = make_group_holdout_splits(train.group_id, n_splits=inner_splits,
                                          test_fraction=test_fraction, random_seed=random_seed + split_id + 1)
        for model, terms in model_terms.items():
            losses = {}
            for alpha in alpha_grid:
                fold_losses = []
                for train_inner, test_inner in inner:
                    fitted = fit_forecast_ridge(train.iloc[train_inner], terms, alpha=alpha, spline_df=spline_df)
                    p = predict_forecast_ridge(fitted, train.iloc[test_inner])
                    fold_losses.append(score_forecasts(train.iloc[test_inner], p).log_loss.mean())
                losses[alpha] = float(np.mean(fold_losses))
            alpha = min(losses, key=losses.get)
            fitted = fit_forecast_ridge(train, terms, alpha=alpha, spline_df=spline_df)
            probability = predict_forecast_ridge(fitted, test)
            score_tables.append(score_forecasts(test, probability).assign(model=model, split_id=split_id))
            prediction_tables.append(test.assign(probability=probability, model=model, split_id=split_id))
            tuning.append(dict(model=model, split_id=split_id, alpha=alpha,
                               inner_log_loss=losses[alpha]))
    return (pd.concat(score_tables, ignore_index=True),
            pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(tuning))


def paired_forecast_scores(scores: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    """Candidate minus baseline on exactly the same reach and outer split."""
    keys, metrics = ["split_id", "group_id"], ["log_loss", "brier_score"]
    reference = scores.loc[scores.model.eq(baseline)].set_index(keys)[metrics].sort_index()
    if reference.empty or reference.index.has_duplicates:
        raise ValueError("Baseline needs unique reach/split scores.")
    tables = []
    for model, values in scores.loc[~scores.model.eq(baseline)].groupby("model", sort=False):
        candidate = values.set_index(keys)[metrics].sort_index()
        if not candidate.index.equals(reference.index):
            raise ValueError("Candidate and baseline must share identical reach/split keys.")
        tables.append((candidate-reference).reset_index().assign(model=model))
    return pd.concat(tables, ignore_index=True)


def forecast_calibration(predictions: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Reliability bins with equal total weight per reach/split within model.

    Bins summarize next-horizon outcomes, not instantaneous event hazards.
    """
    tables = []
    for model, values in predictions.groupby("model", sort=False):
        values = values.sort_values("probability").copy()
        values["weight"] = 1 / values.groupby(["split_id", "group_id"]).probability.transform("size")
        values["bin"] = np.minimum(
            ((values.weight.cumsum() - values.weight / 2) / values.weight.sum() * n_bins).astype(int), n_bins-1)
        for bin_id, group in values.groupby("bin"):
            tables.append(dict(model=model, bin=bin_id,
                predicted=np.average(group.probability, weights=group.weight),
                observed=np.average(group.event_within_horizon, weights=group.weight)))
    return pd.DataFrame(tables)


def plot_forecast_examples(
    predictions: pd.DataFrame, records: Sequence[Mapping], group_ids: Sequence,
    *, titles: Sequence[str] | None = None, context_margin_s: float = 0.15,
):
    """Zoomed-window position and rolling horizon probabilities; no event-time MAP.

    Pass one model and optionally one split. Otherwise average probabilities
    over test appearances at the same decision time. Only evaluated times are
    drawn; unavailable post-event predictions are never replaced with zeros.

    The x-axis is cropped to the evaluated decision window plus
    ``context_margin_s`` of surrounding context on each side, not the full
    recorded reach. Pre-movement hold and post-arrival dwell are typically
    several times longer than the evaluated window and would otherwise
    dominate the plot while carrying no forecast. The full reach duration is
    reported in the title so cropping is explicit rather than silent.
    """
    import matplotlib.pyplot as plt

    lookup = {record["group_id"]: record for record in records}
    fig, axes = plt.subplots(2 * len(group_ids), 1, figsize=(8, 3.8 * len(group_ids)), squeeze=False)
    for row, group_id in enumerate(group_ids):
        top, bottom = axes[2*row:2*row+2, 0]
        record = lookup[group_id]
        time = np.asarray(record["time_s"])
        gap = np.linalg.norm(np.asarray(record["position"]) - np.asarray(record["target"]), axis=1)
        event = record["event_time_s"]
        values = predictions.loc[predictions.group_id.eq(group_id)]
        if values.model.nunique() != 1:
            raise ValueError("Select exactly one model for example plots.")
        curve = values.groupby("time_s", sort=True).agg(probability=("probability", "mean"),
                                                        outcome=("event_within_horizon", "first"))
        horizon = float(values.horizon_s.iloc[0])
        reach_start_s, reach_end_s = float(time[0] - time[0]), float(time[-1] - time[0])
        window_start_s = min(curve.index.min(), event - horizon) - time[0] - context_margin_s
        window_end_s = min(reach_end_s, event - time[0] + context_margin_s)
        window_start_s = max(reach_start_s, window_start_s)
        top.plot(time-time[0], gap, color="0.35", lw=1.5)
        top.scatter(event-time[0], np.interp(event, time, gap), color="crimson", marker="*", s=90,
                    label="Observed brake", zorder=3)
        bottom.plot(curve.index-time[0], curve.probability, color="#0072B2", lw=1.8,
                    label=f"Predicted brake within {horizon*1000:g} ms")
        bottom.step(curve.index-time[0], curve.outcome, where="post", color="0.5", ls=":", lw=1,
                    label="Observed next-window outcome")
        for axis in (top, bottom):
            axis.axvspan(max(time[0], event-horizon)-time[0], event-time[0], color="crimson", alpha=0.08)
            axis.axvline(event-time[0], color="crimson", lw=1.2)
            axis.set_xlim(window_start_s, window_end_s)
            axis.set_xlabel("Time in reach (s)")
            axis.legend(fontsize=7, frameon=False)
        title = titles[row] if titles is not None else f"Reach {group_id}"
        top.set(title=f"{title} ({values.split_id.nunique()} fits; {reach_end_s:.2f}s reach, zoomed)",
                ylabel=f"Remaining gap ({record.get('spatial_unit', 'configured units')})")
        bottom.set(title="Prospective braking probability", ylabel="Probability", ylim=(-0.03, 1.03))
    fig.tight_layout(h_pad=2.2)
    return fig


def plot_causal_forecast_qc(records: Sequence[Mapping], *, config: ForecastConfig):
    """Inspect past-only predictors beside optional retrospective detector traces."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(records), figsize=(5*len(records), 9), squeeze=False)
    for column, record in enumerate(records):
        f = causal_reach_features(record['time_s'], record['position'], record['target'], config=config)
        relative = f.time_s-f.time_s.iloc[0]
        for row, (name, label) in enumerate([
            ('distance', f'Gap ({config.spatial_unit})'),
            ('speed', f'Speed ({config.spatial_unit}/s)'),
            ('closure_speed', f'Closure ({config.spatial_unit}/s)'),
            ('tau_s', 'Tau (s)'),
        ]):
            ax = axes[row, column]
            ax.plot(relative, f[name], color='#0072B2', label='Past-only feature')
            if row == 1 and 'offline_speed' in record:
                ax.plot(relative, record['offline_speed'], color='0.5', ls='--', label='Retrospective detector speed')
            if record.get('event_time_s') is not None:
                ax.axvline(record['event_time_s']-f.time_s.iloc[0], color='crimson', label='Observed brake')
            confirmed = f.loc[f.decision_sample, 'time_s']
            if len(confirmed):
                ax.axvline(confirmed.iloc[0]-f.time_s.iloc[0], color='black', ls=':', label='Causal movement confirmation')
            ax.set(xlabel='Time in reach (s)', ylabel=label)
            if row == 0:
                ax.set_title(f"Reach {record['group_id']}")
            if row == 1:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig
