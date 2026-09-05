"""Reusable plotting helpers."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


STATUS_COLORS = {
    "inbound": "#0173B2",
    "outbound": "#029E73",
    "incorrect": "#D55E00",
}


def add_trial_status(fig: go.Figure, status: str) -> go.Figure:
    """Add a colored trial-status badge to a Plotly figure."""
    status = str(status).lower()
    fig.add_annotation(
        text=status.title(),
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor=STATUS_COLORS.get(status, "#6B7280"),
        borderpad=5,
        font={"color": "white"},
    )
    return fig


def add_mpl_trial_status(ax: Axes, status: str) -> Axes:
    """Add a colored trial-status badge to a Matplotlib axis."""
    status = str(status).lower()
    ax.text(
        0.02,
        0.98,
        status.title(),
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": STATUS_COLORS.get(status, "#6B7280"), "edgecolor": "none"},
    )
    return ax


def plot_direction_maps(
    samples: pd.DataFrame,
    background: pd.DataFrame,
    *,
    variable: str,
    x_column: str,
    y_column: str,
    direction_column: str,
    directions: tuple[str, str],
    units: str,
    signed: bool,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Plot spatial values for two movement directions on matched axes."""
    limit = (
        samples[variable].abs().quantile(0.99)
        if signed
        else samples[variable].quantile(0.99)
    )
    vmin, cmap = (-limit, "RdBu") if signed else (0, "viridis")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, direction in zip(axes, directions):
        selected = samples[samples[direction_column].eq(direction)]
        ax.scatter(background[x_column], background[y_column], s=1, color="lightgrey")
        points = ax.scatter(
            selected[x_column], selected[y_column], s=3,
            c=selected[variable], cmap=cmap, vmin=vmin, vmax=limit,
            rasterized=True,
        )
        ax.set(title=direction.title(), xlabel="Position x (cm)", aspect="equal")
    axes[0].set_ylabel("Position y (cm)")
    fig.colorbar(points, ax=axes, label=f"{variable.title()} ({units})")
    return fig
