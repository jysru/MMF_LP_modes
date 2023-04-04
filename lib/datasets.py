import os
import multiprocessing
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

from lib.grid import Grid
from lib.fiber import GrinFiber
from lib.modes import GrinLPMode
from lib.speckle import GrinSpeckle
from lib.beams import GaussianBeam
from lib.devices import MockDeformableMirror
from lib.coupling import GrinFiberCoupler

default_nprocs = multiprocessing.cpu_count()


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

            coupled_in = GrinFiberCoupler(beam.field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)
            coupled_out = GrinFiberCoupler(propagated_field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)

            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = coupled_out.field

            dm.magnify_by(magnification)
            beam.grid.magnify_by(magnification)
            if verbose:
                print(f"Computed couple {i+1}/{self.length}")

    def multiproc_compute(self, phases_dim: tuple[int, int] = (6,6), beam_width: float = 5100e-6, magnification: float = 200, n_procs: int = default_nprocs - 1, max_loops_per_proc: int = 100):
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
        dm_grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
        beam = GaussianBeam(dm_grid)
        beam.compute(width=beam_width)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        self._multiprocess_compute(phases_dim, dm, beam, magnification, n_procs, max_loops_per_proc)

    def _multiprocess_compute(self, phases_dim: tuple[int, int], dm: MockDeformableMirror, beam: GaussianBeam, magnification: int, n_procs: int, max_loops_per_proc: int = 100):
        if n_procs > default_nprocs:
            print(f"Coerced number of workers to {default_nprocs}")
            n_procs = default_nprocs

        divider = self.length // n_procs
        remainder = self.length % n_procs

        loops_per_proc = []
        counts = 0
        if divider > max_loops_per_proc:
            n_outer_loops = np.ceil(divider / max_loops_per_proc).astype(int)
            for loop in range(n_outer_loops):
                if loop < (n_outer_loops - 1):
                    loops = max_loops_per_proc * np.ones(shape=(n_procs,), dtype=int)
                    counts += np.sum(loops)
                else:
                    remaining = self.length - counts
                    remaining_divider = remaining // n_procs
                    remaining_remainder = remaining % n_procs
                    loops = remaining_divider * np.ones(shape=(n_procs,), dtype=int)
                    loops[-1] = loops[-1] + remaining_remainder
                loops_per_proc.append(loops)
        else:
            loops = divider * np.ones(shape=(n_procs,), dtype=int)
            loops[-1] = loops[-1] + remainder
            loops_per_proc.append(loops)

        for loop in range(len(loops_per_proc)):
            async_results = []
            with multiprocessing.Pool(processes=n_procs) as pool:
                for proc in range(n_procs):
                    async_results.append(pool.apply_async(self._compute_task, args=(phases_dim, dm, beam, magnification, loops_per_proc[loop][proc])))
                pool.close()
                pool.join()

            for i, result in enumerate(async_results):
                phase_maps, fields = result.get()
                if (loop == 0) and (i == 0):
                    self._phase_maps = phase_maps
                    self._fields = fields
                else:
                    self._phase_maps = np.concatenate((self._phase_maps, phase_maps), axis=2)
                    self._fields = np.concatenate((self._fields, fields), axis=2)

            print(f"Computed multiprocessing loop {loop+1}/{n_outer_loops} using {n_procs} workers")
    
    def _compute_task(self, phases_dim: tuple[int, int], dm: MockDeformableMirror, beam: GaussianBeam, magnification: int, loops_per_proc: int):
        _phase_maps = np.zeros(shape=(phases_dim + (loops_per_proc,)))
        _fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (loops_per_proc,)), dtype=np.complex128)

        for i in range(loops_per_proc):
            phase_map = 2*np.pi*np.random.rand(*phases_dim)
            dm.apply_phase_map(phase_map)
            dm.reduce_by(magnification)
            beam.grid.reduce_by(magnification)
            beam.field = dm._field_matrix

            coupled_in = GrinFiberCoupler(beam.field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)
            coupled_out = GrinFiberCoupler(propagated_field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)

            _phase_maps[:,:,i] = phase_map
            _fields[:,:,i] = coupled_out.field

            dm.magnify_by(magnification)
            beam.grid.magnify_by(magnification)

        return _phase_maps, _fields

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
    dset = SimulatedGrinSpeckleOutputDataset(fiber, grid, length=10000, N_modes=55)
    dset.multiproc_compute(phases_dim=(6,6))
    # dset.compute(phases_dim=(6,6))
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
