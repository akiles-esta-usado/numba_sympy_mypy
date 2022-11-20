from numpy.typing import NDArray
import numpy as np

from numba import njit


def inc(a : NDArray) -> NDArray:
  return a + 1

@njit
def inc_jitter(a: NDArray):
  return a + 1



a = np.array([2, 2])

print(inc_jitter(a))
