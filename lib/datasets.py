import os
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt


from grid import Grid
from fiber import GrinFiber
from modes import GrinLPMode
from speckle import GrinSpeckle
from beams import GaussianBeam
from devices import MockDeformableMirror
from coupling import GrinFiberCoupler


class GrinLPDataset:
    """Pure LP modes from GRIN fiber"""

    def __init__(self, fiber: GrinFiber, grid: Grid, N_modes: int = 55, noise_std: float = 0.0) -> None:
        self._N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self._grid = grid
        self._fiber = fiber
        self._noise_std = noise_std
        self._fields = None
        self.compute()

    def compute(self):
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (2*self._N_modes,)))
        for i in range(self._N_modes):
            n, m = self._fiber._neff_hnm[i, 2], self._fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(self._fiber, self._grid)
            self._fields[:, :, 2 * i] = mode._fields[:, :, 0]
            self._fields[:, :, 2 * i + 1] = mode._fields[:, :, 1]

    @property
    def length(self):
        return 2*self._N_modes

    @property
    def intensities(self):
        val = np.square(np.abs(self._fields))
        return np.abs(val + self._noise_std * np.random.randn(*val.shape))
    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]


class GrinLPSpeckleDataset:
    """Random combination of LP modes from GRIN fiber"""

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0) -> None:
        self._N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self._length = length
        self._grid = grid
        self._fiber = fiber
        self._noise_std = noise_std
        self._fields = None
        self.compute()

    def compute(self):
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        for i in range(self.length):
            speckle = GrinSpeckle(self._fiber, self._grid, N_modes=self._N_modes, noise_std = self._noise_std)
            speckle.compose()
            self._fields[:, :, i] = speckle.field

    @property
    def length(self):
        return self._length

    @property
    def intensities(self):
        val = np.square(np.abs(self._fields))
        return np.abs(val + self._noise_std * np.random.randn(*val.shape))
    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]

    
class SimulatedGrinSpeckleOutputDataset:
    """Coupling from modal decomposition on GRIN fiber LP modes, then propagation using a random mode coupling matrix"""

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0) -> None:
        self._N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self._length = length
        self._grid = grid
        self._fiber = fiber
        self._noise_std = noise_std
        self._fields = None
        self._phase_dims = None
        self._phase_maps = None
        self._coupling_matrix = self._fiber.modes_coupling_matrix(complex=complex)

    def compute(self, phases_dim: tuple[int, int] = (6,6), beam_width: float = 5100e-6, magnification: float = 200, verbose: bool = True):
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
        dm_grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
        beam = GaussianBeam(dm_grid)
        beam.compute(width=beam_width)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        for i in range(self.length):
            phase_map = 2*np.pi*np.random.rand(*phases_dim)
            dm.apply_phase_map(phase_map)
            dm.reduce_by(magnification)
            beam.grid.reduce_by(magnification)
            beam.field = dm._field_matrix

            coupled_in = GrinFiberCoupler(beam.field, beam.grid, fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)
            coupled_out = GrinFiberCoupler(propagated_field, beam.grid, N_modes=self._N_modes)

            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = coupled_out.field

            dm.magnify_by(magnification)
            beam.grid.magnify_by(magnification)
            if verbose:
                print(f"Computed couple {i+1}/{self.length}")

    @property
    def length(self):
        return self._length
    
    @property
    def phases_size(self):
        return np.prod(np.array(list(self._phase_dims)))

    @property
    def intensities(self):
        val = np.square(np.abs(self._fields))
        return np.abs(val + self._noise_std * np.random.randn(*val.shape))
    
    def export(self, path: str = '.', name: str = None, verbose: bool = True):
        if name is None:
            default_name = f"synth_dset_grin_Nmodes={self._N_modes}_len={self.length}_mirr={self.phases_size}"
            name = default_name
        savename = os.path.join(path, f"{name}.mat")

        savemat(
                file_name = savename,
                mdict = {
                    'phase_maps': self._phase_maps, 'intens': self.intensities,
                    'coupling_matrix': self._coupling_matrix,
                    'length': self.length, 'N_modes': self._N_modes,
                }
            )
        
        if verbose:
            print(f"Dataset saved: {savename}")
    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]


if __name__ == "__main__":
    grid = Grid(pixel_size=0.5e-6)
    fiber = GrinFiber()
    # dset = GrinLPDataset(fiber, grid, N_modes=10)
    # dset = GrinLPSpeckleDataset(fiber, grid, length=5, N_modes=10)


    import cProfile as profile
    import pstats

    prof = profile.Profile()
    prof.enable()
    dset = SimulatedGrinSpeckleOutputDataset(fiber, grid, length=3, N_modes=55)
    dset.compute(phases_dim=(6,6))
    dset.export()
    prof.disable()

    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10) # top 10 rows


    # plt.figure()
    # plt.imshow(dset[0], cmap='gray')
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(dset[3], cmap='gray')
    # plt.colorbar()
    # plt.show()
