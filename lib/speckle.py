import numpy as np
import matplotlib.pyplot as plt

from lib.grid import Grid
from lib.fiber import GrinFiber
from lib.modes import GrinLPMode
from lib.plots import complex_image


class GrinSpeckle():

    def __init__(self, fiber: GrinFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0.0) -> None:
        self.N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self.fiber = fiber
        self.grid = grid
        self.noise_std = noise_std
        self.modes_coeffs = None
        self.orient_coeffs = None
        self.field = None

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
    def phase(self):
        return np.angle(self.field)
    
    @property
    def energy(self):
        return np.sum(np.square(np.abs(self.field)))

    @property
    def intensity(self):
        val = np.square(np.abs(self.field))
        val = val / np.max(val)
        return np.abs(val + self.noise_std * np.random.randn(*val.shape))
    
    @property
    def coeffs_intensity(self):
        return np.square(np.abs(self.modes_coeffs))
    
    @property
    def total_coeffs_intensity(self):
        return np.sum(np.square(np.abs(self.modes_coeffs)))
    
    @property
    def coeffs_phases(self):
        return np.angle(self.modes_coeffs)
    
    def plot(self, cmap: str = 'gray', complex: bool = False, complex_hsv: bool = False):
        r = self.fiber.radius * 1e6
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * 1e6
        circle1 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')
        circle2 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')

        if complex:
            if complex_hsv:
                fig = plt.figure()
                ax = plt.gca()
                pl = plt.imshow(complex_image(self.field), extent=extent)
                ax.add_patch(circle1)
                ax.set_xlabel("x [um]")
                ax.set_ylabel("x [um]")
                ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
                return (fig, ax, pl)
            else:
                fig, axs = plt.subplots(1, 2, figsize=(13,4))
                pl0 = axs[0].imshow(self.intensity, extent=extent, cmap=cmap)
                pl1 = axs[1].imshow(self.phase, extent=extent, cmap="twilight")
                axs[0].add_patch(circle1)
                axs[1].add_patch(circle2)
                axs[0].set_xlabel("x [um]")
                axs[1].set_xlabel("x [um]")
                axs[0].set_ylabel("y [um]")
                axs[1].set_ylabel("y [um]")
                axs[0].set_title(f"GRIN speckle intensity ({self.N_modes} modes)")
                axs[1].set_title(f"GRIN speckle phase ({self.N_modes} modes)")
                plt.colorbar(pl0, ax=axs[0])
                plt.colorbar(pl1, ax=axs[1])
                return (fig, axs, [pl0, pl1])
        else:
            fig = plt.figure()
            ax = plt.gca()
            pl = plt.imshow(self.intensity, cmap=cmap, extent=extent)
            ax.add_patch(circle1)
            ax.set_xlabel("x [um]")
            ax.set_ylabel("x [um]")
            ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
            plt.colorbar(pl, ax=ax)
            return (fig, ax, pl)
        
    # def plot_coefficients(self):
    #     fig = plt.figure()
    #     ax = plt.gca()
    #     pl = plt.plot(np.square(np), extent=extent)
    #     ax.add_patch(circle1)
    #     ax.set_xlabel("x [um]")
    #     ax.set_ylabel("x [um]")
    #     ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
    #     return (fig, ax, pl)


    def _sanity_checker(self):
        coeffs, orients = self.decompose(N_modes=self.N_modes)

        print(
            f"\n\t Speckle sanity checker ({self.N_modes} modes):\n\n"
            f"\t - Sum of intensity coefficients: {self.total_coeffs_intensity}\n"
            f"\t - Sum of decomposition intensity coefficients: {np.sum(np.square(np.abs(coeffs)))}\n"
            f"\t - Intensity coefficients:\n{self.coeffs_intensity}\n"
            f"\t - Decomposition intensity coefficients:\n{np.square(np.abs(coeffs))}\n"
            f"\t - Phases coefficients:\n{self.coeffs_phases}\n"
            f"\t - Decomposition phases coefficients:\n{np.angle(coeffs)}\n"
            f"\t - Orientation coefficients:\n{self.orient_coeffs}\n"
            f"\t - Decomposition orientation ccoefficients:\n{orients}\n"
            f"\n\t End\n\n"
        )

    def __str__(self) -> str:
        return (
            f"\t {__class__.__name__} instance ({self.N_modes} modes) with:\n"
            f"\t - Sum of intensity coefficients: {self.total_coeffs_intensity}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coefficients:\n{self.coeffs_intensity}\n"
            f"\t - Phase coefficients:\n{self.coeffs_phases}\n"
            f"\t - Orientation coefficients:\n{self.orient_coeffs}\n"
        )


if __name__ == "__main__":
    grid = Grid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    speckle = GrinSpeckle(fiber, grid, N_modes=15)
    speckle.compose()
    speckle._sanity_checker()
    speckle.plot(complex=True)
    print(speckle)
    plt.show()
