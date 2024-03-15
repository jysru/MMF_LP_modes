import numpy as np
import matplotlib.pyplot as plt

from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber, StepIndexFiber
from mmfsim.modes import GrinLPMode, StepIndexLPMode
from mmfsim.plots import complex_image


class GrinSpeckle():

    def __init__(self, fiber: GrinFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0.0) -> None:
        self.N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self.fiber = fiber
        self.grid = grid
        self.noise_std = noise_std
        self.modes_coeffs = None
        self.orient_coeffs = None
        self.field = None

    def compose(self, coeffs: tuple[np.array, np.array] = None, oriented: bool = False):
        fields1 = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        fields2 = np.zeros_like(fields1)

        if coeffs is not None:
            self.modes_coeffs = coeffs[0]
            self.orient_coeffs = coeffs[1]
        else:
            self._modes_random_coeffs(oriented=oriented)

        for i in range(self.N_modes):
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
                fields1[:,:,i], fields2[:,:,i] = mode_fields[:,:,0], mode_fields[:,:,1]
            else:
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

    def _modes_random_coeffs(self, oriented: bool = False):
        # Generate vector that sums up to one (intensity coefficients)
        Ip = np.random.rand(self.N_modes)
        Ip = Ip / np.sum(Ip)
    
        # Generate random phases
        Phip = 2 * np.pi* np.random.rand(self.N_modes)
    
        # Get the complex coefficients
        modes_coeffs = np.sqrt(Ip) * np.exp(1j * Phip)
        self.orient_coeffs = np.zeros(self.N_modes) if oriented else np.random.rand(self.N_modes)
        self.modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs)

    def decompose(self, N_modes: int = 10, normalize_coeffs: bool = False):
        N_modes = self.fiber._N_modes if N_modes > self.fiber._N_modes else N_modes
        modes_coeffs = np.zeros(shape=(N_modes), dtype=np.complex64)
        orient_coeffs = np.zeros(shape=(N_modes))

        for i in range(N_modes):
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
                mode0, mode90 = mode_fields[:, :, 0], mode_fields[:, :, 1]
            else:
                n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
                mode = GrinLPMode(n, m)
                mode.compute(self.fiber, self.grid)
                mode0, mode90 = mode._fields[:,:,0], mode._fields[:,:,1]

            if n == 0:
            # if mode.is_centrosymmetric: # Centro-symmetric mode
                Cp = GrinSpeckle.power_overlap_integral(self.field, mode0)
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode0)
                modes_coeffs[i] = Cp * np.exp(1j * phi)
            else: # Non centro-symmetric mode
                Cp1 = GrinSpeckle.power_overlap_integral(self.field, mode0)
                Cp2 = GrinSpeckle.power_overlap_integral(self.field, mode90)
                Cor = Cp1 / (Cp1 + Cp2)
                mode_orient = np.sqrt(Cor) * mode0 +  np.sqrt(1 - Cor) * mode90
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode_orient)
                modes_coeffs[i] = np.sqrt(Cp1 + Cp2) * np.exp(1j * phi)
                orient_coeffs[i] = Cor

        modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs) if normalize_coeffs else modes_coeffs
        return modes_coeffs, orient_coeffs

    @staticmethod
    def power_overlap_integral(field, mode):
        return np.square(np.abs(np.sum(field * np.conj(mode)))) / (np.sum(np.square(np.abs(field))) * np.sum(np.square(np.abs(mode))))
    
    @staticmethod
    def phase_from_overlap_integral(field, mode):
        return np.angle(np.sum(field * np.conj(mode)))
    
    @staticmethod
    def complex_overlap_integral(field, mode):
        return np.sum(field * np.conj(mode)) / np.sqrt(np.sum(np.square(np.abs(field))) * np.sum(np.square(np.abs(mode))))

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
    
    def plot(self, cmap: str = 'gray', complex: bool = False, complex_hsv: bool = False, display_core: bool = True):
        r = self.fiber.radius * 1e6
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * 1e6
        if display_core:
            circle1 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')
            circle2 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')

        if complex:
            if complex_hsv:
                fig = plt.figure()
                ax = plt.gca()
                pl = plt.imshow(complex_image(self.field), extent=extent)
                if display_core:
                    ax.add_patch(circle1)
                ax.set_xlabel("x [um]")
                ax.set_ylabel("x [um]")
                ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
                return (fig, ax, pl)
            else:
                fig, axs = plt.subplots(1, 2, figsize=(13,4))
                pl0 = axs[0].imshow(self.intensity, extent=extent, cmap=cmap)
                pl1 = axs[1].imshow(self.phase, extent=extent, cmap="twilight")
                if display_core:
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
            if display_core:
                ax.add_patch(circle1)
            ax.set_xlabel("x [um]")
            ax.set_ylabel("x [um]")
            ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
            plt.colorbar(pl, ax=ax)
            return (fig, ax, pl)
        
    def plot_coefficients(self):
        x = np.arange(self.N_modes)
        nm = self.fiber._neff_hnm[:self.N_modes, 2:].astype(int)
        nm_strings = [f"{nm[i,0]:d},{nm[i,1]:d}" for i in range(nm.shape[0])]

        fig = plt.figure(figsize=(15,7))
        ax = plt.gca()
        pl = plt.bar(x, self.coeffs_intensity * 100)
        ax_t = ax.secondary_xaxis('top')
        ax_t.tick_params(axis='x', direction='in')
        ax_t.set_xlabel("LP mode linear index")

        ax.set_xlabel(r"LP$_{n,m}$ mode")
        ax.set_xticks(x, nm_strings, rotation='vertical')
        ax.set_ylabel("Energy percentage [%]")
        ax.set_title(
            f"Energy percentage on LP modes "
            f"({self.N_modes} modes, total energy: {self.total_coeffs_intensity * 100:2.1f}%)"
        )
        return (fig, ax, pl)

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
            f"\t - Energy: {self.energy}\n"
            f"\t - Sum of intensity coefficients: {self.total_coeffs_intensity}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coefficients:\n{self.coeffs_intensity}\n"
            f"\t - Phase coefficients:\n{self.coeffs_phases}\n"
            f"\t - Orientation coefficients:\n{self.orient_coeffs}\n"
        )


class DegenGrinSpeckle(GrinSpeckle):

    def __init__(self, fiber: GrinFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.N_modes = fiber._N_modes_degen if N_modes > fiber._N_modes_degen else N_modes

    def compose(self, coeffs: np.array = None):
        fields = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        if coeffs is not None:
            self.modes_coeffs = coeffs
        else:
            self._modes_random_coeffs()
        
        k, i = 0, 0
        while k < self.N_modes:
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
            else:
                mode = GrinLPMode(n, m)
                mode.compute(self.fiber, self.grid)
                mode_fields = mode._fields

            if n > 0: # Mode is degenerated
                try:
                    fields[:, :, k] = mode_fields[:, :, 0]
                except IndexError:
                    break
                try:
                    fields[:, :, k + 1] = mode_fields[:, :, 1]
                except IndexError:
                    break
                k += 2
            else:
                try:
                    fields[:, :, k] = mode_fields[:, :, 0]
                except IndexError:
                    break
                k += 1
            i += 1

        field = 0
        for i in range(self.N_modes):
            Cp = self.modes_coeffs[i]
            field += fields[:, :, i] * Cp
        self.field = field

    def _modes_random_coeffs(self):
        # Generate vector that sums up to one (intensity coefficients)
        Ip = np.random.rand(self.N_modes)
        Ip = Ip / np.sum(Ip)
    
        # Generate random phases
        Phip = 2 * np.pi* np.random.rand(self.N_modes)
    
        # Get the complex coefficients
        modes_coeffs = np.sqrt(Ip) * np.exp(1j * Phip)
        self.modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs)

    def decompose(self, N_modes: int = 10, normalize_coeffs: bool = False):
        self.N_modes = self.fiber._N_modes if N_modes > self.fiber._N_modes else N_modes
        modes_coeffs = np.zeros(shape=(self.N_modes), dtype=np.complex64)
        k, i = 0, 0

        while k < N_modes:
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
            else:
                mode = GrinLPMode(n, m)
                mode.compute(self.fiber, self.grid)
                mode_fields = mode._fields

            if n > 0: # Mode is degenerated
                try:
                    modes_coeffs[k] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 0])
                except IndexError:
                    break
                try:
                    modes_coeffs[k + 1] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 1])
                except IndexError:
                    break
                k += 2
            else:
                try:
                    modes_coeffs[k] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 0])
                except IndexError:
                    break
                k += 1
            i += 1
        return GrinSpeckle._normalize_coeffs(modes_coeffs) if normalize_coeffs else modes_coeffs
    
    def _sanity_checker(self, normalize_coeffs: bool = False):
        coeffs = self.decompose(N_modes=self.N_modes)
        coeffs = GrinSpeckle._normalize_coeffs(coeffs) if normalize_coeffs else coeffs
        print(
            f"\n\t Speckle sanity checker ({self.N_modes} modes):\n\n"
            f"\t - Sum of intensity coefficients: {self.total_coeffs_intensity}\n"
            f"\t - Sum of decomposition intensity coefficients: {np.sum(np.square(np.abs(coeffs)))}\n"
            f"\t - Intensity coefficients:\n{self.coeffs_intensity}\n"
            f"\t - Decomposition intensity coefficients:\n{np.square(np.abs(coeffs))}\n"
            f"\t - Phases coefficients:\n{self.coeffs_phases}\n"
            f"\t - Decomposition phases coefficients:\n{np.angle(coeffs)}\n"
            f"\n\t End\n\n"
        )

    def plot_coefficients(self):
        nm = self.fiber._neff_hnm[:self.N_modes, 2:].astype(int)
        nm_strings = []
        k, i = 0, 0

        while k < self.N_modes:
            if nm[i, 0] != 0:
                nm_strings.append(f"{nm[i,0]:d},{nm[i,1]:d}a")
                nm_strings.append(f"{nm[i,0]:d},{nm[i,1]:d}b")
                k += 2
            else:
                nm_strings.append(f"{nm[i,0]:d},{nm[i,1]:d}")
                k += 1
            i += 1                
        
        x = np.arange(len(nm_strings[:self.N_modes]))
        fig = plt.figure(figsize=(15,7))
        ax = plt.gca()
        pl = plt.bar(x[:self.coeffs_intensity.shape[0]], self.coeffs_intensity * 100)
        ax_t = ax.secondary_xaxis('top')
        ax_t.tick_params(axis='x', direction='in')
        ax_t.set_xlabel("LP degenerated mode linear index")

        ax.set_xlabel(r"LP$_{n,m}$ mode")
        ax.set_xticks(x, nm_strings[:self.N_modes], rotation='vertical')
        ax.set_ylabel("Energy percentage [%]")
        ax.set_title(
            f"Energy percentage on LP modes "
            f"({self.N_modes} modes, total energy: {self.total_coeffs_intensity * 100:2.1f}%)"
        )
        return (fig, ax, pl)


class StepIndexSpeckle(GrinSpeckle):

    def __init__(self, fiber: StepIndexFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0.0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)

    def compose(self, coeffs: tuple[np.array, np.array] = None, oriented: bool = False):
        fields1 = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        fields2 = np.zeros_like(fields1)

        if coeffs is not None:
            self.modes_coeffs = coeffs[0]
            self.orient_coeffs = coeffs[1]
        else:
            self._modes_random_coeffs(oriented=oriented)

        for i in range(self.N_modes):
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
                fields1[:,:,i], fields2[:,:,i] = mode_fields[:,:,0], mode_fields[:,:,1]
            else:
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

    def __str__(self) -> str:
        return (
            f"\t {__class__.__name__} instance ({self.N_modes} modes) with:\n"
            f"\t - Energy: {self.energy}\n"
            f"\t - Sum of intensity coefficients: {self.total_coeffs_intensity}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coefficients:\n{self.coeffs_intensity}\n"
            f"\t - Phase coefficients:\n{self.coeffs_phases}\n"
            f"\t - Orientation coefficients:\n{self.orient_coeffs}\n"
        )


class DegenStepIndexSpeckle(DegenGrinSpeckle):

    def __init__(self, fiber: StepIndexFiber, grid: Grid, N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.N_modes = fiber._N_modes_degen if N_modes > fiber._N_modes_degen else N_modes

    def compose(self, coeffs: np.array = None):
        fields = np.zeros(shape=(self.grid.pixel_numbers[0], self.grid.pixel_numbers[1], self.N_modes))
        if coeffs is not None:
            self.modes_coeffs = coeffs
        else:
            self._modes_random_coeffs()
        
        k, i = 0, 0
        while k < self.N_modes:
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
            else:
                mode = GrinLPMode(n, m)
                mode.compute(self.fiber, self.grid)
                mode_fields = mode._fields

            if n > 0: # Mode is degenerated
                try:
                    fields[:, :, k] = mode_fields[:, :, 0]
                except IndexError:
                    break
                try:
                    fields[:, :, k + 1] = mode_fields[:, :, 1]
                except IndexError:
                    break
                k += 2
            else:
                try:
                    fields[:, :, k] = mode_fields[:, :, 0]
                except IndexError:
                    break
                k += 1
            i += 1

        field = 0
        for i in range(self.N_modes):
            Cp = self.modes_coeffs[i]
            field += fields[:, :, i] * Cp
        self.field = field

    def decompose(self, N_modes: int = 10, normalize_coeffs: bool = False):
        N_modes = self.fiber._N_modes_degen if N_modes > self.fiber._N_modes_degen else N_modes
        modes_coeffs = np.zeros(shape=(self.N_modes), dtype=np.complex64)
        k, i = 0, 0

        while k < N_modes:
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            if self.fiber.stored_modes_fields:
                mode_fields = self.fiber._modes[:, :, :, i]
            else:
                mode = StepIndexLPMode(n, m)
                mode.compute(self.fiber, self.grid)
                mode_fields = mode._fields

            if n > 0: # Mode is degenerated
                try:
                    modes_coeffs[k] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 0])
                except IndexError:
                    break
                try:
                    modes_coeffs[k + 1] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 1])
                except IndexError:
                    break
                k += 2
            else:
                try:
                    modes_coeffs[k] = GrinSpeckle.complex_overlap_integral(self.field, mode_fields[:, :, 0])
                except IndexError:
                    break
                k += 1
            i += 1
        return GrinSpeckle._normalize_coeffs(modes_coeffs) if normalize_coeffs else modes_coeffs


if __name__ == "__main__":
    grid = Grid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    speckle = DegenGrinSpeckle(fiber, grid, N_modes=fiber._N_modes_degen)
    speckle.compose()
    
    coeffs = speckle.decompose(N_modes = fiber._N_modes_degen)
    speckle._sanity_checker(normalize_coeffs=True)
    speckle.plot(complex=True)
    # speckle.plot_coefficients()
    # print(speckle)
    plt.show()
