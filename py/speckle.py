import numpy as np
import matplotlib.pyplot as plt

from grid import CameraGrid
from fiber import GrinFiber
from modes import GrinLPMode


class GrinSpeckle():

    def __init__(self, fiber: GrinFiber, grid: CameraGrid, N_modes: int = 10, noise_std: float = 0.0) -> None:
        self.N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self.fiber = fiber
        self.grid = grid
        self.noise_std = noise_std
        self.modes_coeffs = None
        self.orient_coeffs = None
        self.field = None
        self.compose()

    def compose(self):
        fields1 = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        fields2 = np.zeros_like(fields1)
        self._modes_random_coeffs()

        for i in range(self.N_modes):
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(self.fiber, self.grid)
            fields1[:,:,i], fields2[:,:,i] = mode._fields[:,:,0], mode._fields[:,:,1]

        field = 0
        for i in range(self.N_modes):
            n = self.fiber._neff_hnm[i, 2]
            Cp = self.modes_coeffs[i]
            if n == 0: # Centro-symmetric mode
                field += fields1[:,:,i] * Cp
            else:
                Cor = self.orient_coeffs[i]
                tmp = fields1[:,:,i] * np.sqrt(Cor) + fields2[:,:,i] * np.sqrt(1 - Cor)
                field += tmp * Cp
        self.field = field / np.max(np.abs(field))

    def _modes_random_coeffs(self):
        # Generate vector that sums up to one (intensity coefficients)
        Ip = np.random.rand(self.N_modes)
        Ip = Ip / np.sum(Ip)
    
        # Generate random phases
        Phip = 2 * np.pi* np.random.rand(self.N_modes)
    
        # Get the complex coefficients
        self.modes_coeffs = np.sqrt(Ip) * np.exp(1j * Phip)
        self.orient_coeffs = np.random.rand(self.N_modes)

    @property
    def intensity(self):
        val = np.square(np.abs(self.field))
        val = val / np.max(val)
        return np.abs(val + self.noise_std * np.random.randn(*val.shape))
    
    def plot(self, cmap: str = 'hot'):
        r = self.fiber.radius * 1e6
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * 1e6

        fig = plt.figure()
        ax = plt.gca()
        pl = plt.imshow(self.intensity, cmap=cmap, extent=extent)
        ax.set_xlabel("x [um]")
        ax.set_xlabel("x [um]")
        ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
        plt.colorbar(pl, ax=ax)


if __name__ == "__main__":
    grid = CameraGrid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    speckle = GrinSpeckle(fiber, grid, N_modes=50)
    speckle.plot()
    plt.show()
