from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product

from .pauli import AXIS_NAMES


@dataclass(frozen=True)
class FeatureTerm:
    sites: tuple[int, ...]
    axes: tuple[int, ...]
    name: str


@dataclass(frozen=True)
class FeatureSpec:
    N: int
    all_two_body: bool = True
    contiguous_three_body: bool = True
    contiguous_four_body: bool = True

    @cached_property
    def _cached_terms(self) -> list[FeatureTerm]:
        terms: list[FeatureTerm] = []
        for i in range(self.N):
            for a in range(3):
                terms.append(FeatureTerm((i,), (a,), f"{AXIS_NAMES[a]}_{i}"))
        if self.all_two_body:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    for axes in product(range(3), repeat=2):
                        name = " ".join(f"{AXIS_NAMES[a]}_{s}" for s, a in zip((i, j), axes))
                        terms.append(FeatureTerm((i, j), axes, name))
        if self.contiguous_three_body:
            for i in range(self.N - 2):
                sites = (i, i + 1, i + 2)
                for axes in product(range(3), repeat=3):
                    name = " ".join(f"{AXIS_NAMES[a]}_{s}" for s, a in zip(sites, axes))
                    terms.append(FeatureTerm(sites, axes, name))
        if self.contiguous_four_body:
            for i in range(self.N - 3):
                sites = (i, i + 1, i + 2, i + 3)
                for axes in product(range(3), repeat=4):
                    name = " ".join(f"{AXIS_NAMES[a]}_{s}" for s, a in zip(sites, axes))
                    terms.append(FeatureTerm(sites, axes, name))
        return terms

    def terms(self) -> list[FeatureTerm]:
        return self._cached_terms

    @property
    def num_features(self) -> int:
        return 3 * self.N + (9 * self.N * (self.N - 1) // 2 if self.all_two_body else 0) + (
            27 * max(0, self.N - 2) if self.contiguous_three_body else 0
        ) + (81 * max(0, self.N - 3) if self.contiguous_four_body else 0)

    @cached_property
    def _cached_names(self) -> list[str]:
        return [term.name for term in self.terms()]

    def names(self) -> list[str]:
        return self._cached_names
