import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from fiber import GrinFiber
from modes import GrinLPMode


class GrinSpeckle():

    def __init__(self, fiber: GrinFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0.0) -> None:
        self.N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self.fiber = fiber
        self.grid = grid
        self.noise_std = noise_std
        self.modes_coeffs = None
        self.orient_coeffs = None
        self.field = None
        self.compose()

    def compose(self, coeffs: tuple[np.array, np.array] = None):
        fields1 = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        fields2 = np.zeros_like(fields1)

        if coeffs is not None:
            self.modes_coeffs = coeffs[0]
            self.orient_coeffs = coeffs[1]
        else:
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
        self.field = field

    def _modes_random_coeffs(self):
        # Generate vector that sums up to one (intensity coefficients)
        Ip = np.random.rand(self.N_modes)
        Ip = Ip / np.sum(Ip)
    
        # Generate random phases
        Phip = 2 * np.pi* np.random.rand(self.N_modes)
    
        # Get the complex coefficients
        modes_coeffs = np.sqrt(Ip) * np.exp(1j * Phip)
        self.orient_coeffs = np.random.rand(self.N_modes)
        self.modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs)

    def decompose(self, N_modes: int = 10):
        N_modes = self.fiber._N_modes if N_modes > self.fiber._N_modes else N_modes
        modes_coeffs = np.zeros(shape=(N_modes), dtype=np.complex64)
        orient_coeffs = np.zeros(shape=(N_modes))

        for i in range(N_modes):
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(self.fiber, self.grid)
            mode0, mode90 = mode._fields[:,:,0], mode._fields[:,:,1]

            if n == 0: # Centro-symmetric mode
                Cp = GrinSpeckle.power_overlap_integral(self.field, mode0)
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode0)
                modes_coeffs[i] = np.sqrt(Cp) * np.exp(1j * phi)
            else:
                Cp1 = GrinSpeckle.power_overlap_integral(self.field, mode0)
                Cp2 = GrinSpeckle.power_overlap_integral(self.field, mode90)
                Cor = Cp1 / (Cp1 + Cp2)
                mode_orient = np.sqrt(Cor) * mode0 +  np.sqrt(1 - Cor) * mode90
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode_orient)
                modes_coeffs[i] = np.sqrt(Cp1 + Cp2) * np.exp(1j * phi)
                orient_coeffs[i] = Cor

        modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs)
        return modes_coeffs, orient_coeffs

    @staticmethod
    def power_overlap_integral(field, mode):
        return np.square(np.abs(np.sum(field * np.conj(mode)))) / (np.sum(np.square(np.abs(field))) * np.sum(np.square(np.abs(mode))))
    
    @staticmethod
    def phase_from_overlap_integral(field, mode):
        return np.angle(np.sum(field * np.conj(mode)))

    @staticmethod
    def _normalize_coeffs(coeffs):
        coeffs_abs = np.abs(coeffs)
        coeffs_angles = np.angle(coeffs)
        coeffs_angles = np.angle(np.exp(1j * (coeffs_angles - coeffs_angles[0])))
        return coeffs_abs * np.exp(1j * coeffs_angles)
    
    @property
    def energy(self):
        return np.sum(np.square(np.abs(self.field)))

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

    def _sanity_checker(self):
        coeffs, orients = speckle.decompose(N_modes=self.N_modes)

        print(f"\t - Sum of intensity coeffs: {np.sum(np.square(np.abs(speckle.modes_coeffs)))}")
        print(f"\t - Sum of decomposition intensity coeffs: {np.sum(np.square(np.abs(coeffs)))}")

        print(f"\t - Intensity coeffs:")
        print(np.square(np.abs(speckle.modes_coeffs)))
        print(f"\t - Decomposition intensity coeffs:")
        print(np.square(np.abs(coeffs)))

        print(f"\t - Phases coeffs:")
        print(np.angle(speckle.modes_coeffs))
        print(f"\t - Decomposition phases coeffs:")
        print(np.angle(coeffs))

        print(f"\t - Orient coeffs:")
        print(speckle.orient_coeffs)
        print(f"\t - Decomposition orient coeffs:")
        print(orients)



if __name__ == "__main__":
    grid = Grid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    speckle = GrinSpeckle(fiber, grid, N_modes=15)
    speckle._sanity_checker()
    speckle.plot()
    plt.show()