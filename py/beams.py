import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as sp


class Grid():

    def __init__(self, size: int=128) -> None:
        self.size = np.abs(size)
        self.x_vec = np.linspace(start=0, stop=self.size, num=self.size)
        self.y_vec = np.linspace(start=0, stop=self.size, num=self.size)
        self.x_grid, self.y_grid = np.meshgrid(self.x_vec, self.y_vec)

    @property
    def centers(self):
        return np.array([self.size//2, self.size//2])

        

class Beam():

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.field = np.zeros_like(self.grid.x_grid, dtype=float)

    def _add_offsets(self, offsets: list[int]= None) -> list[int]:
        if offsets:
            centers = self.grid.centers + np.array(offsets)
        else:
            centers = self.grid.centers
        return centers

    def gen_gaussian(self, amplitude: float=1, width: float=10, centers: list[int]=None):
        centers = self._add_offsets(centers)
        self.field = amplitude*np.exp(-((np.power(self.grid.x_grid-centers[0],2) + np.power(self.grid.y_grid-centers[1],2))/np.power(width,2)))

    def gen_bessel(self, amplitude: float=1, order=1, width: float=10, centers: list[int]=None):
        centers = self._add_offsets(centers)
        arg = np.sqrt(np.power(self.grid.x_grid - centers[0],2) + np.power(self.grid.y_grid - centers[1],2))
        self.field = sp.jn(order, arg/width)
        self.field = amplitude*self.field/np.max(self.field)

    @property
    def intensity(self):
        return np.power(self.field,2)



if __name__ == "__main__":
    grid = Grid(122)
    beam = Beam(grid)
    beam.gen_bessel(order=1, amplitude=1, width=10, centers=[0,0])

    fig = plt.figure()
    sns.heatmap(beam.intensity, square=True)
    plt.show()