"""Reusable notebook controls."""

from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
from IPython.display import display


def trial_selector(
    plot_function: Callable[[int, int], Any],
    *,
    epoch: int,
    trial_number: int,
) -> dict[str, widgets.IntText]:
    """Display editable epoch/trial controls and update a plot callback."""
    controls = {
        "epoch": widgets.IntText(value=epoch, description="Epoch", step=1),
        "trial_number": widgets.IntText(
            value=trial_number, description="Trial", step=1
        ),
    }
    output = widgets.interactive_output(plot_function, controls)
    display(widgets.HBox(list(controls.values())), output)
    return controls


def epoch_selector(
    plot_function: Callable[[int], Any], *, epoch: int
) -> dict[str, widgets.IntText]:
    """Display an editable epoch control and update a plot callback."""
    controls = {
        "epoch": widgets.IntText(value=epoch, description="Epoch", step=1)
    }
    output = widgets.interactive_output(plot_function, controls)
    display(widgets.HBox(list(controls.values())), output)
    return controls
