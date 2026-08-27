from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HierarchySpec:
    """Defines a cross-sectional hierarchical aggregation structure."""

    levels: list[str]
    structure: dict[str, list[str]] = field(default_factory=dict)
    # e.g., structure={"Total": ["Region_A", "Region_B"], "Region_A": ["Store_1", "Store_2"]}
    bottom_level: str | None = None

    def __post_init__(self):
        if not self.bottom_level and self.levels:
            self.bottom_level = self.levels[-1]


@dataclass
class TemporalSpec:
    """Defines temporal aggregation frequencies and forecast horizons."""

    frequencies: list[str] = field(default_factory=lambda: ["D", "W", "M"])  # e.g., Daily, Weekly, Monthly
    horizon: dict[str, int] = field(default_factory=lambda: {"D": 28, "W": 4, "M": 1})
