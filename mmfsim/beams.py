from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from mmfsim.grid import Grid
from mmfsim.plots import complex_image


class Beam(ABC):

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.field = np.zeros(shape=(grid.pixel_numbers), dtype=np.complex128)
        self.centers = None
        self.amp = None

    def _add_offsets(self, offsets: list[int]= None) -> None:
        self.centers = self.grid.offsets + np.array(offsets) if offsets else self.grid.offsets

    def add_phase(self, phase: np.ndarray) -> None:
        self.field = self.field * np.exp(1j * phase)

    def normalize_by_energy(self):
        self.field /= np.sqrt(self.energy)

    @abstractmethod
    def compute(self):
        return
    
    @property
    def phase(self):
        return np.angle(self.field)
    
    @property
    def amplitude(self):
        return np.abs(self.field)

    @property
    def intensity(self):
        return np.square(np.abs(self.field))
    
    @property
    def energy(self):
        return np.sum(self.intensity)
    
    def plot(self, cmap: str = 'gray', extent_coeff: float = 1e6, complex: bool = False, complex_hsv: bool = False):
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * extent_coeff

        if complex:
            if complex_hsv:
                fig = plt.figure()
                ax = plt.gca()
                pl = plt.imshow(complex_image(self.field), extent=extent)
                ax.set_xlabel("x [um]")
                ax.set_ylabel("y [um]")
                ax.set_title(f"Beam field")
                return (fig, ax, pl)
            else:
                fig, axs = plt.subplots(1, 2, figsize=(13,4))
                pl0 = axs[0].imshow(self.intensity, extent=extent, cmap=cmap)
                pl1 = axs[1].imshow(self.phase, extent=extent, cmap="twilight")
                axs[0].set_xlabel("x [um]")
                axs[1].set_xlabel("x [um]")
                axs[0].set_ylabel("y [um]")
                axs[1].set_ylabel("y [um]")
                axs[0].set_title(f"Beam intensity")
                axs[1].set_title(f"Beam phase")
                plt.colorbar(pl0, ax=axs[0])
                plt.colorbar(pl1, ax=axs[1])
                return (fig, axs, [pl0, pl1])
        else:
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
        self.amp = amplitude
        self.width = width

        self.field = self.amp * np.exp(
            -((np.square(self.grid.X - self.centers[0]) + np.square(self.grid.Y - self.centers[1]))
            / (np.square(width)))
            )
        self.field
        
    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Amplitude: {self.amp}\n"
            f"  - Width: {self.width}\n"
            f"  - Centers: {self.centers}\n"
            f"  - Energy: {self.energy}\n"
        )
        

class BesselBeam(Beam):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        self.width = None
        self.order = None

    def compute(self, amplitude: float = 1, order: int = 1, width: float = 10e-6, centers: list[int] = None):
        self._add_offsets(centers)
        self.amp = amplitude
        self.width = width
        self.order = order

        arg = np.sqrt(np.power(self.grid.X - self.centers[0], 2) + np.power(self.grid.Y - self.centers[1], 2))
        self.field = sp.jn(order, arg / width)
        self.field = self.amp * self.field / np.max(self.field)

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Amplitude: {self.amp}\n"
            f"  - Width: {self.width}\n"
            f"  - Order: {self.order}\n"
            f"  - Centers: {self.centers}\n"
            f"  - Energy: {self.energy}\n"
        )
        

class BesselGaussianBeam(Beam):

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)
        
        self.bessel_width = None
        self.gaussian_width = None
        self.order = None

    def compute(self, amplitude: float = 1, order: int = 1, bessel_width: float = 10e-6, gaussian_width: float = 20e-6, centers: list[int] = None):
        self._add_offsets(centers)
        self.amp = amplitude
        self.bessel_width = bessel_width
        self.gaussian_width = gaussian_width
        
        gauss = GaussianBeam(self.grid)
        gauss.compute(amplitude=amplitude, width=gaussian_width, centers=centers)
        bessel = BesselBeam(self.grid)
        bessel.compute(amplitude=amplitude, order=order, width=bessel_width, centers=centers)

        self.field = gauss.field * bessel.field
        self.field = self.amp * self.field / np.max(self.field)

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Amplitude: {self.amp}\n"
            f"  - Bessel width: {self.bessel_width}\n"
            f"  - Gaussian width: {self.gaussian_width}\n"
            f"  - Order: {self.order}\n"
            f"  - Centers: {self.centers}\n"
            f"  - Energy: {self.energy}\n"
        )


if __name__ == "__main__":
    grid = Grid(pixel_size=2e-6)
    beam = GaussianBeam(grid)
    beam.compute( amplitude=1, width=30e-6, centers=[0,0])
    print(beam)

    beam.plot(complex=True)
    plt.show()
