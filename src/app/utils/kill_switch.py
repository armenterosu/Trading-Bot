from __future__ import annotations
"""Drawdown kill switch: stop trading when equity drawdown exceeds limit."""
from dataclasses import dataclass


@dataclass
class DrawdownKillSwitch:
    limit: float  # e.g., 0.2 for 20%
    enabled: bool = True

    def should_stop(self, peak_equity: float, current_equity: float) -> bool:
        if not self.enabled or peak_equity <= 0:
            return False
        dd = (peak_equity - current_equity) / peak_equity
        return dd >= self.limit
