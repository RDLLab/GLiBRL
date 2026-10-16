"""CSV logging for GLiBRL's task-belief KL diagnostics.

The algorithm is a frozen ``struct.PyTreeNode`` that is rebuilt on every gradient
step, so the per-run state -- the output file, the update counter and the belief
the previous update ended on -- lives here, keyed by output path, instead of on
the agent.
"""

import csv
import os
from pathlib import Path
from typing import Any, Mapping


class PosteriorKLLogger:
    """Appends one CSV row per model update.

    The column set is taken from the first row written and reused afterwards, so
    callers can decide how much per-task detail to emit. Rows are stamped with the
    number of environment steps taken so far (see :meth:`set_global_step`) and with
    a per-process count of update calls.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        if str(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_posterior: Any | None = None
        """The belief the previous update ended on, or None before the first one."""
        self._fieldnames: list[str] | None = None
        self._update_index = 0
        self._global_step = 0

    def set_global_step(self, global_step: int) -> None:
        """Stamp subsequent rows with this environment step count."""
        self._global_step = global_step

    def write_update(self, row: Mapping[str, float]) -> None:
        """Append one update's row and advance the update counter."""
        stamped = {
            "global_step": self._global_step,
            "update": self._update_index,
            **row,
        }

        write_header = False
        if self._fieldnames is None:
            self._fieldnames = list(stamped.keys())
            # Resuming into an existing file keeps the header that is already there.
            write_header = not (self.path.exists() and self.path.stat().st_size > 0)

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(stamped)

        self._update_index += 1


_loggers: dict[str, PosteriorKLLogger] = {}


def get_posterior_kl_logger(path: str | os.PathLike) -> PosteriorKLLogger:
    """Get (or create) the logger writing to ``path``."""
    key = os.path.abspath(path)
    if key not in _loggers:
        _loggers[key] = PosteriorKLLogger(key)
    return _loggers[key]
