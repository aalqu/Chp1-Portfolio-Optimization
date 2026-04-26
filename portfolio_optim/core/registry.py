from __future__ import annotations

from typing import Callable

from portfolio_optim.core.base import BaseSolver


class SolverRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], BaseSolver]] = {}

    def register(self, name: str, builder: Callable[[], BaseSolver]) -> None:
        self._builders[name] = builder

    def build(self, name: str) -> BaseSolver:
        if name not in self._builders:
            raise KeyError(f"Unknown solver '{name}'. Available: {sorted(self._builders)}")
        return self._builders[name]()

    @property
    def names(self) -> list[str]:
        return sorted(self._builders)

