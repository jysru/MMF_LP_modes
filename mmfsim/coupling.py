import numpy as np
import matplotlib.pyplot as plt

from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber, StepIndexFiber
from mmfsim.modes import GrinLPMode, StepIndexLPMode
from mmfsim.speckle import GrinSpeckle, DegenGrinSpeckle, StepIndexSpeckle, DegenStepIndexSpeckle
from mmfsim.devices import MockDeformableMirror
from mmfsim.plots import complex_image
import mmfsim.beams as beams


class GrinFiberCoupler(GrinSpeckle):

    def __init__(self, field: np.ndarray, grid: Grid, fiber: GrinFiber = GrinFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.field = field
        self.field = self.field / self.energy
        self.coupling_matrix = None
        self.modes_coeffs, self.orient_coeffs = self.decompose(self.N_modes)
        self.recompose()

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
        return self.field

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
        self.field = self.field / self.energy
        self.coupling_matrix = None
        self.modes_coeffs = self.decompose(self.N_modes)
        self.recompose()

    def propagate(self, matrix: np.ndarray = None, complex: bool = True, full: bool = False):
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
    

class StepIndexFiberCoupler(StepIndexSpeckle):

    def __init__(self, field: np.ndarray, grid: Grid, fiber: StepIndexFiber = StepIndexFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.field = field
        self.field = self.field / self.energy
        self.coupling_matrix = None
        self.modes_coeffs, self.orient_coeffs = self.decompose(self.N_modes)
        self.recompose()

    def propagate(self, matrix: np.ndarray = None, complex: bool = True, full: bool = True):
        if matrix is None:
            self.coupling_matrix = self.fiber.modes_coupling_matrix(complex=complex, full=full)
        else:
            self.coupling_matrix = matrix
        out_coeffs = np.dot(self.coupling_matrix[:self.modes_coeffs.shape[0], :self.modes_coeffs.shape[0]], self.modes_coeffs)
        self.compose(coeffs=(out_coeffs, self.orient_coeffs))
        return self.field
        
    def recompose(self):
        self.compose(coeffs=(self.modes_coeffs, self.orient_coeffs))
        return self.field

    def __str__(self) -> str:
        return (
            f"\t - Coupled power: {np.sum(np.square(np.abs(self.modes_coeffs)))}\n"
            f"\t - Number of modes: {self.N_modes}\n"
            f"\t - Intensity coeffs:\n"
            f"{np.square(np.abs(self.modes_coeffs))}"
        )


class StepIndexFiberDegenCoupler(DegenStepIndexSpeckle):

    def __init__(self, field: np.ndarray, grid: Grid, fiber: StepIndexFiber = StepIndexFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, grid, N_modes, noise_std)
        self.field = field
        self.field = self.field / self.energy
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
    beam.compute(amplitude=1, width=5100e-6, centers=[0,0])
    dm.apply_amplitude_map(beam.amplitude)
    dm.reduce_by(200)
    dm.plot()

    beam.grid.reduce_by(200)
    beam.field = dm._field_matrix 
    beam.normalize_by_energy()
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

