from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from mmfsim.grid import Grid
from mmfsim.plots import complex_image


class Phase(ABC):

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.phase = np.zeros(shape=(grid.pixel_numbers), dtype=np.float64)
        self.centers = None
        self.angle = 0

    def _add_2D_offsets(self, offsets: list[int] = None) -> None:
        self.centers = self.grid.offsets + np.array(offsets) if offsets else self.grid.offsets

    @abstractmethod
    def compute(self):
        return

    def plot(self, cmap: str = 'hsv', extent_coeff: float = 1e6):
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * extent_coeff

        fig = plt.figure()
        ax = plt.gca()
        pl = plt.imshow(self.phase, cmap=cmap, extent=extent)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_title(f"Phase")
        plt.colorbar(pl, ax=ax)
        return (fig, ax, pl)
    
    def add(self, phase):
        self.phase = np.angle(np.exp(1j * (self.phase + phase)))

    def sub(self, phase):
        self.phase = np.angle(np.exp(1j * (self.phase - phase)))
    
    def __add__(self, other):
        self.phase = np.angle(np.exp(1j * (self.phase + other.phase)))
        return self

    def __sub__(self, other):
        self.phase = np.angle(np.exp(1j * (self.phase - other.phase)))
        return self
    

class FlatPhase(Phase):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.offset = None

    def compute(self, offset: float = 0, centers: list[int] = [0, 0]):
        self._add_2D_offsets(centers)
        self.offset = offset
        self.phase = np.angle(np.exp(1j * (self.phase + offset)))
        
    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Offset: {self.offset}\n"
            f"  - Centers: {self.centers}\n"
        )


class PowerPhase(Phase):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.offsets = None
        self.order = None

    def compute(self, coeffs: list[float], order: float, centers: list[int] = [0, 0]):
        self._add_2D_offsets(centers)
        self.order = order
        self.add(self._compute_gradient(coeffs))

    def _compute_gradient(self, coeffs):
        return (
            2 * np.pi * np.power(self.grid.X / coeffs[0], self.order)
            + 2 * np.pi * np.power(self.grid.Y / coeffs[1], self.order)
        )

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Centers: {self.centers}\n"
        )


class LinearPhase(PowerPhase):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.order = 1
        self.offsets = None

    def compute(self, coeffs: list[float], centers: list[int] = [0, 0]):
        self._add_2D_offsets(centers)
        self.add(self._compute_gradient(coeffs))

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Centers: {self.centers}\n"
        )
    

class QuadraticPhase(PowerPhase):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.order = 2
        self.offsets = None

    def compute(self, coeffs: list[float], centers: list[int] = [0, 0]):
        self._add_2D_offsets(centers)
        self.add(self._compute_gradient(coeffs))

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Centers: {self.centers}\n"
        )
    

class VortexPhase(Phase):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.offset = None
        self.order = None

    def compute(self, offset: float = 0, order: float = 1, centers: list[int] = [0, 0]):
        self._add_2D_offsets(centers)
        self.offset = offset
        self.order = order
        vortex = np.angle(np.exp(1j * (self.order * self.grid.A + self.offset)))
        self.add(vortex)

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Value: {self.value}\n"
            f"  - Centers: {self.centers}\n"
        )


if __name__ == "__main__":
    pass
