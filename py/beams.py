from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from grid import Grid


class Beam(ABC):

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.field = np.zeros(shape=(grid.pixel_numbers), dtype=float)
        self.centers = None
        self.amplitude = None

    def _add_offsets(self, offsets: list[int]= None) -> None:
        self.centers = self.grid.offsets + np.array(offsets) if offsets else self.grid.offsets

    @abstractmethod
    def compute(self):
        return

    @property
    def intensity(self):
        return np.power(self.field,2)
    
    @property
    def energy(self):
        return np.sum(self.intensity)
    
    def plot(self, cmap: str = 'hot', extent_coeff: float = 1e6):
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * extent_coeff
        fig = plt.figure()
        ax = plt.gca()
        pl = plt.imshow(self.intensity, cmap=cmap, extent=extent)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_title(f"Beam intensity")
        plt.colorbar(pl, ax=ax)
        return (fig, ax, pl)
    

class GaussianBeam(Beam):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.width = None

    def compute(self, amplitude: float = 1, width: float = 10e-6, centers: list[int] = None):
        self._add_offsets(centers)
        self.amplitude = amplitude
        self.width = width

        self.field = amplitude * np.exp(
            -((np.square(self.grid.X - self.centers[0]) + np.square(self.grid.Y - self.centers[1]))
            / (2 * np.square(width)))
            )
        
    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Amplitude: {self.amplitude}\n"
            f"  - Width: {self.width}\n"
            f"  - Centers: {self.centers}\n"
        )
        

class BesselBeam(Beam):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.width = None
        self.order = None

    def compute(self, amplitude: float = 1, order: int = 1, width: float = 10e-6, centers: list[int] = None):
        self._add_offsets(centers)
        self.amplitude = amplitude
        self.width = width
        self.order = order

        arg = np.sqrt(np.power(self.grid.X - self.centers[0], 2) + np.power(self.grid.Y - self.centers[1], 2))
        self.field = sp.jn(order, arg / width)
        self.field = amplitude * self.field / np.max(self.field)

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Amplitude: {self.amplitude}\n"
            f"  - Width: {self.width}\n"
            f"  - Order: {self.order}\n"
            f"  - Centers: {self.centers}\n"
        )
        
    
        



if __name__ == "__main__":
    grid = Grid(pixel_size=2e-6)
    beam = GaussianBeam(grid)
    beam.compute( amplitude=1, width=5e-6, centers=[50e-6,0])
    print(beam)

    beam.plot()
    plt.show()
