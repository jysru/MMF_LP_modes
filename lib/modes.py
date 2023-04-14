from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt

from lib.grid import Grid
from lib.fiber import GrinFiber

lru_cache_default_size = 1024

@lru_cache(maxsize=lru_cache_default_size)
class GrinLPMode():

    def __init__(self, n: int, m: int, theta0: float = 0) -> None:
        self.n = int(n)
        self.m = int(m)
        self.theta0 = theta0 * np.pi / 180
        self._radius = None
        self._x = None
        self._y = None
        self._fields = None

    @lru_cache(maxsize=lru_cache_default_size)
    def compute(self, fiber: GrinFiber, grid: Grid):
        self._radius = fiber.radius
        self._x = grid.x
        self._y = grid.y
        self._centers = grid.offsets

        fac_n = np.math.factorial(self._fn)
        fac_m_plus_n = np.math.factorial(self._fm + self._fn)

        delta0m = 1 if self._fm==0 else 0
        epsilon_mn = np.pi * np.square(fiber.radius) * fac_m_plus_n * (1 + delta0m) / (2 * fiber._V * fac_n)
        ro = np.array(grid.R / fiber.radius * np.sqrt(fiber._V)).astype(float)

        Lmn = np.zeros(shape=(grid.pixel_numbers[0], grid.pixel_numbers[1], self._fn + 1))
        for s in range(self._fn + 1):
            num = fac_m_plus_n * np.power(-1, s) * np.power(ro, 2 * s)
            denom = np.math.factorial(self._fm + s) * np.math.factorial(self._fn - s) * np.math.factorial(s)
            Lmn[:, :, s] = num / denom
        Lmn = np.sum(Lmn, axis=2)

        fac1 = 1 / np.sqrt(epsilon_mn)
        fac2 = np.power(ro, self._fm)
        fac3 = np.exp(-np.square(ro) / 2)

        self._fields = np.zeros(shape=(len(grid.x), len(grid.y), 2))
        self._fields[:,:,0] = fac1 * fac2 * fac3 * Lmn * np.cos(self._fm * grid.A + self.theta0)
        self._fields[:,:,1] = fac1 * fac2 * fac3 * Lmn * np.cos(self._fm * grid.A + self.theta0 + np.pi / 2 )
        self._fields = self._fields / np.sqrt(self.energies)

    @property
    def _fn(self):
        return self.m - 1
    
    @property
    def _fm(self):
        return self.n
    
    @property
    def intensities(self):
        return np.square(np.abs(self._fields))
    
    @property
    def energies(self):
        return np.sum(self.intensities, axis=(0,1))
    
    def plot(self):
        r = self._radius * 1e6
        extent = np.array([np.min(self._x), np.max(self._x), np.min(self._y), np.max(self._y)]) * 1e6
        str_mode = f"{self.n,self.m}"

        circle1 = plt.Circle((-self._centers[0], -self._centers[1]), r, fill=False, edgecolor='k', linestyle='--')
        circle2 = plt.Circle((-self._centers[0], -self._centers[1]), r, fill=False, edgecolor='k', linestyle='--')

        fig, axs = plt.subplots(1, 2, figsize=(12,4))
        pl0 = axs[0].imshow(self._fields[:,:,0], extent=extent, cmap="bwr")
        pl1 = axs[1].imshow(self._fields[:,:,1], extent=extent, cmap="bwr")
        axs[0].add_patch(circle1)
        axs[1].add_patch(circle2)
        axs[0].set_xlabel("x [um]")
        axs[1].set_xlabel("x [um]")
        axs[0].set_ylabel("y [um]")
        axs[1].set_ylabel("y [um]")
        axs[0].set_title(f"LP{str_mode}")
        axs[1].set_title(f"LP{str_mode}")
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])


if __name__ == "__main__":
    grid = Grid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    mode = GrinLPMode(1,1)
    mode.compute(fiber, grid)
    mode.plot()
    plt.show()
