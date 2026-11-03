"""Fixed sub-component functions g_i that discovered models compose into f."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SubComponent:
    """A fixed, callable sub-component of the composite system.

    Attributes:
        fn: The sub-component function, called as `fn(x, *params)`.
        latex: The sub-component's equation as a LaTeX string.
        n_params: Number of parameters the sub-component takes (excluding its input).
    """

    fn: Callable
    latex: str
    n_params: int

    def __call__(self, *args, **kwargs):
        """Calls the underlying sub-component function."""
        return self.fn(*args, **kwargs)


def _g_light(L, K_L):
    return L / (K_L + L)


def _g_nitrogen(N, K_N):
    return N / (K_N + N)


g_light = SubComponent(
    fn=_g_light,
    latex=r"g_{\mathrm{light}}(L; K_L) = \frac{L}{K_L + L}",
    n_params=1,
)

g_nitrogen = SubComponent(
    fn=_g_nitrogen,
    latex=r"g_{\mathrm{nitrogen}}(N; K_N) = \frac{N}{K_N + N}",
    n_params=1,
)
