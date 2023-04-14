import numpy as np
import matplotlib.pyplot as plt

from lib.grid import Grid
from lib.fiber import GrinFiber
from lib.modes import GrinLPMode
from lib.speckle import GrinSpeckle, DegenGrinSpeckle
from lib.devices import MockDeformableMirror
from lib.plots import complex_image
import lib.beams as beams


class GrinFiberCoupler(GrinSpeckle):

    def __init__(self, field: np.ndarray, grid: Grid, fiber: GrinFiber = GrinFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.field = field
        self.coupling_matrix = None
        # self.modes_coeffs, self.orient_coeffs = self.decompose(self.N_modes)
        self.decompose(self.N_modes)
        self.recompose()

    def decompose(self, N_modes: int = 10, normalize_coeffs: bool = False):
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

        self.modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs) if normalize_coeffs else modes_coeffs
        self.orient_coeffs = orient_coeffs

    def propagate(self, matrix: np.ndarray = None, complex: bool = True, full: bool = False):
        if matrix is None:
            self.coupling_matrix = self.fiber.modes_coupling_matrix(complex=complex, full=full)
        else:
            self.coupling_matrix = matrix
        out_coeffs = np.dot(self.coupling_matrix[:self.modes_coeffs.shape[0], :self.modes_coeffs.shape[0]], self.modes_coeffs)
        self.compose(coeffs=(out_coeffs, self.orient_coeffs))
        return self.field
        
    def recompose(self):
        self.compose(coeffs=(self.modes_coeffs, self.orient_coeffs))

    def __str__(self) -> str:
        return (
            f"\t - Coupled power: {np.sum(np.square(np.abs(self.modes_coeffs)))}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coeffs:\n"
            f"{np.square(np.abs(self.modes_coeffs))}"
        )


class GrinFiberDegenCoupler(DegenGrinSpeckle):

    def __init__(self, field: np.ndarray, grid: Grid, fiber: GrinFiber = GrinFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.field = field
        self.coupling_matrix = None
        self.modes_coeffs = self.decompose(self.N_modes)
        self.recompose()

    def propagate(self, matrix: np.ndarray = None, complex: bool = True, full: bool = True):
        if matrix is None:
            self.coupling_matrix = self.fiber.modes_coupling_matrix(complex=complex, full=full, degen=True)
        else:
            self.coupling_matrix = matrix
        out_coeffs = np.dot(self.coupling_matrix[:self.modes_coeffs.shape[0], :self.modes_coeffs.shape[0]], self.modes_coeffs)
        self.compose(coeffs=(out_coeffs))
        return self.field
        
    def recompose(self):
        self.compose(coeffs=(self.modes_coeffs))

    def __str__(self) -> str:
        return (
            f"\t - Coupled power: {np.sum(np.square(np.abs(self.modes_coeffs)))}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coeffs:\n"
            f"{np.square(np.abs(self.modes_coeffs))}"
        )
    

if __name__ == "__main__":

    phase_map = 2*np.pi*np.random.rand(12,12)
    dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
    dm.apply_phase_map(phase_map)

    grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
    beam = beams.GaussianBeam(grid)
    beam.compute(amplitude=1, width=3000e-6, centers=[0,0])
    dm.apply_amplitude_map(beam.amplitude)
    dm.reduce_by(200)
    dm.plot()

    beam.grid.reduce_by(200)
    beam.field = dm._field_matrix 
    beam.normalize_by_energy()
    # coupled = GrinFiberCoupler(beam.field, beam.grid, fiber=GrinFiber(), N_modes=55)
    # print(f"Coupled energy: {coupled.energy}")
    coupled = GrinFiberDegenCoupler(beam.field, beam.grid, fiber=GrinFiber(), N_modes=113)
    print(f"Coupled energy: {coupled.energy}")

    coupled.plot(cmap='gray')
    # coupled.plot_coefficients()
    speck = coupled.propagate(complex=True)
    print(f"Speckle energy: {np.sum(np.square(np.abs(speck)))}")
    coupled.plot(cmap='gray')
    # coupled.plot_coefficients()
    coupled.fiber.plot_coupling_matrix(coupled.coupling_matrix, complex=True)
    print(np.sum(np.square(np.abs(speck))))
    print(coupled)
    plt.show()
