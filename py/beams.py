import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from grid import Grid
        

class Beam():

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.field = np.zeros(shape=(grid.pixel_numbers), dtype=float)

    def _add_offsets(self, offsets: list[int]= None) -> list[int]:
        if offsets:
            centers = self.grid.offsets + np.array(offsets)
        else:
            centers = self.grid.offsets
        return centers

    def gen_gaussian(self, amplitude: float=1, width: float=10, centers: list[int]=None):
        centers = self._add_offsets(centers)
        self.field = amplitude * np.exp(
            -((np.square(self.grid.X - centers[0]) + np.square(self.grid.Y - centers[1]))
            / (2 * np.square(width)))
            )

    def gen_bessel(self, amplitude: float=1, order=1, width: float=10, centers: list[int]=None):
        centers = self._add_offsets(centers)
        arg = np.sqrt(np.power(self.grid.X - centers[0], 2) + np.power(self.grid.Y - centers[1], 2))
        self.field = sp.jn(order, arg / width)
        self.field = amplitude * self.field / np.max(self.field)

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


if __name__ == "__main__":
    grid = Grid(pixel_size=2e-6)
    beam = Beam(grid)
    # print(beam.field.shape)
    # beam.gen_bessel(order=1, amplitude=1, width=10e-6)
    beam.gen_bessel(order=2, amplitude=1, width=20e-6, centers=[0,0])

    beam.plot()
    plt.show()
