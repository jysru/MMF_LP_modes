import os
import multiprocessing
from scipy.io import savemat
from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt

from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode
from mmfsim.speckle import GrinSpeckle, DegenGrinSpeckle
from mmfsim.beams import GaussianBeam
from mmfsim.devices import MockDeformableMirror
from mmfsim.coupling import GrinFiberCoupler, GrinFiberDegenCoupler
from mmfsim.transforms import fresnel_transform, fourier_transform

default_nprocs = multiprocessing.cpu_count()


class RandomDataset():

    def __init__(self, phases_dim: int = 6, intens_dim: int = 96, length: int = 1000, noise_std: float = 0.0) -> None:
        self.phases_dim = phases_dim
        self.intens_dim = intens_dim
        self.length = length
        self.noise_std = noise_std
        self.inputs = None
        self.outputs = None
        self.matrix = None
        self.amplitude = None
        self._energy = None
        self.generate()

    def gaussian_amplitude(self):
        x = np.linspace(0, 1e-2, self.phases_dim)
        xx, yy = np.meshgrid(x, x)
        w = 4.5 * 1e-3
        s = 5.0 * 1e-3
        plane = np.exp(-((xx - s)**2 + (yy - s)**2) / w**2)
        return plane.flatten()

    def generate(self):
        amplitude = self.gaussian_amplitude()
        self.amplitude = np.reshape(amplitude, (1, np.square(self.phases_dim)))
        self._energy = np.sum(np.square(np.abs(amplitude)))
        self.amplitude = self.amplitude / np.sqrt(self._energy)

        self.inputs = np.exp(1j * 2 * np.pi * np.random.rand(self.length, np.square(self.phases_dim)))
        self.matrix = self.normalized_matrix()
        self.outputs = np.dot(self.inputs, self.matrix.T)

    def normalized_matrix(self, complex: bool = True):
        r, l = np.random.rand(np.square(self.phases_dim), 1), np.random.rand(np.square(self.intens_dim), 1)
        r = np.sqrt(r / np.sum(r))
        l = np.sqrt(l / np.sum(l))
        X = toeplitz(r, l)

        if complex:
            X = X * np.exp(1j * 2 * np.pi * np.random.rand(np.square(self.phases_dim), np.square(self.intens_dim)))
        return X.T

    @property
    def intensities(self):
        return np.square(np.abs(self.outputs))
    
    @property
    def rank(self):
        return np.linalg.rank(self.matrix)
    
    def export(self, path: str = '.', name: str = None, verbose: bool = True):
        inputs = np.reshape(self.inputs, newshape=(self.phases_dim, self.phases_dim, self.length))
        outputs = np.reshape(self.outputs, newshape=(self.intens_dim, self.intens_dim, self.length))

        if name is None:
            default_name = f"synth_random_dset_len={self.length}_in={self.phases_dim}_out={self.intens_dim}"
            name = default_name
        savename = os.path.join(path, f"{name}.mat")

        savemat(
                file_name = savename,
                mdict = {
                    'phase_maps': inputs, 'intens': np.square(np.abs(outputs)),
                    'matrix': self.matrix,
                    'length': self.length,
                }
            )
        
        if verbose:
            print(f"Dataset saved: {savename}")


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

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, coupling_matrix=None, oriented: bool = False) -> None:
        self._N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self._length = length
        self._grid = grid
        self._fiber = fiber
        self._noise_std = noise_std
        self._fields = None
        self._transf = None
        self._modes_coeffs = None
        self._coupling_matrix = coupling_matrix if coupling_matrix is not None else None
        self._modes_orients = np.random.rand(self._N_modes) 
        self.compute()

    def compute(self):
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._modes_coeffs = np.zeros(shape=(self._N_modes, self.length), dtype=np.complex128)
        for i in range(self.length):
            speckle = GrinSpeckle(self._fiber, self._grid, N_modes=self._N_modes, noise_std = self._noise_std)
            speckle._modes_random_coeffs()
            speckle.compose(coeffs=(speckle.modes_coeffs, self._modes_orients))
            self._modes_coeffs[:, i] = speckle.modes_coeffs

            if self._coupling_matrix is not None:
                modes_coeffs = np.dot(self._coupling_matrix[:speckle.modes_coeffs.shape[0], :speckle.modes_coeffs.shape[0]], speckle.modes_coeffs)
                speckle.compose(coeffs=(modes_coeffs, speckle.orient_coeffs))
            self._fields[:, :, i] = speckle.field

    def compute_fresnel_transforms(self, delta_z: float, pad: float = 2):
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fresnel_transform(self._fields[:, :, i], self._grid, delta_z=delta_z, pad=pad)
        else:
            print("Run compute method first!")

    def compute_fourier_transforms(self, pad: float = 2):
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fourier_transform(self._fields[:, :, i], pad=pad)
        else:
            print("Run compute method first!")

    @property
    def length(self):
        return self._length

    @property
    def intensities(self):
        val = np.square(np.abs(self._fields))
        return np.abs(val + self._noise_std * np.random.randn(*val.shape))
    
    def export(self, path: str = '.', name: str = None, return_fields: bool = False):
        if name is None:
            default_name = f"synth_dset_grinspeckle_Nmodes={self._N_modes}_len={self.length}"
            name = default_name
        savename = os.path.join(path, f"{name}.mat")

        matrix = [] if self._coupling_matrix is None else self._coupling_matrix

        mdict = {
            'phase_maps': self._modes_coeffs, 'intens': self.intensities,
            'length': self.length, 'N_modes': self._N_modes,
            'modes_orients': self._modes_orients,
            'coupling_matrix': matrix,
        }

        if self._transf is not None:
            mdict['transf'] = np.square(np.abs(self._transf))
        if return_fields:
            mdict['fields'] = self._fields

        savemat(file_name = savename, mdict = mdict)
        print(f"Saved dataset: {savename}")

    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]
    

class GrinLPDegenSpeckleDataset(GrinLPSpeckleDataset):
    """Random combination of LP modes from GRIN fiber, from a degenerated basis."""

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 15, noise_std: float = 0, coupling_matrix=None) -> None:
        super().__init__(fiber, grid, length, N_modes, noise_std, coupling_matrix)

    def compute(self):
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._modes_coeffs = np.zeros(shape=(self._N_modes, self.length), dtype=np.complex128)
        for i in range(self.length):
            speckle = DegenGrinSpeckle(self._fiber, self._grid, N_modes=self._N_modes, noise_std = self._noise_std)
            speckle._modes_random_coeffs()
            speckle.compose(coeffs=(speckle.modes_coeffs))
            self._modes_coeffs[:, i] = speckle.modes_coeffs

            if self._coupling_matrix is not None:
                modes_coeffs = np.dot(self._coupling_matrix[:speckle.modes_coeffs.shape[0], :speckle.modes_coeffs.shape[0]], speckle.modes_coeffs)
                speckle.compose(coeffs=(modes_coeffs))
            self._fields[:, :, i] = speckle.field

    
class SimulatedGrinSpeckleOutputDataset:
    """Coupling from modal decomposition on GRIN fiber LP modes, then propagation using a random mode coupling matrix"""

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, degen: bool = False) -> None:
        if degen:
            self._N_modes = fiber._N_modes_degen if N_modes > fiber._N_modes_degen else N_modes
        else:
            self._N_modes = fiber._N_modes if N_modes > fiber._N_modes else N_modes
        self._length = length
        self._grid = grid
        self._fiber = fiber
        self._noise_std = noise_std
        self._fields = None
        self._phase_dims = None
        self._phase_maps = None
        self._coupling_matrix = self._fiber.modes_coupling_matrix(complex=complex, full=False, degen=degen)
        self._normalized_energy_on_macropixels = None
        self._degenerated = degen
        self._transfer_matrix = None
        self._transf = None

    def compute(self, phases_dim: tuple[int, int] = (6,6), beam_width: float = 5100e-6, magnification: float = 200, verbose: bool = True):
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._input_fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=self._grid.pixel_numbers)
        dm_grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
        beam = GaussianBeam(dm_grid)
        beam.compute(width=beam_width)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        for i in range(self.length):
            phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
            dm.apply_phase_map(phase_map)
            dm.reduce_by(magnification)
            beam.grid.reduce_by(magnification)
            beam.field = dm._field_matrix

            if i==0:
                self._compute_transfer_matrix(dm, beam.grid)

            if self._degenerated:
                coupled_in = GrinFiberDegenCoupler(beam.field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)
            else:
                coupled_in = GrinFiberCoupler(beam.field, beam.grid, fiber=self._fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)

            self._input_fields[:,:,i] = dm._field_matrix
            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = propagated_field

            dm.magnify_by(magnification)
            beam.grid.magnify_by(magnification)
            if verbose:
                print(f"Computed couple {i+1}/{self.length}")
            self._normalized_energy_on_macropixels = dm.normalized_energies_on_macropixels

    def compute_from_transfer_matrix(self, phases_dim: tuple[int, int] = (6,6), beam_width: float = 5100e-6, magnification: float = 200, verbose: bool = True, ref_phi: int = None):
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._input_fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=self._grid.pixel_numbers)
        dm_grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
        beam = GaussianBeam(dm_grid)
        beam.compute(width=beam_width)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
        dm.apply_phase_map(phase_map)
        dm.reduce_by(magnification)
        beam.grid.reduce_by(magnification)
        beam.field = dm._field_matrix
        self._compute_transfer_matrix(dm, beam.grid)
        tm = self.reshaped_transfer_matrix
        self._normalized_energy_on_macropixels = dm.normalized_energies_on_macropixels

        for i in range(self.length):
            phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
            if ref_phi is not None:
                phase_map[np.unravel_index(ref_phi, phase_map.shape)] = 0

            x = np.sqrt(self._normalized_energy_on_macropixels) * np.exp(1j * phase_map)
            x = x.flatten()
            y = (tm @ x).reshape(self._grid.pixel_numbers)

            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = y

            if verbose:
                print(f"Computed couple {i+1}/{self.length}")
            

    def compute_fresnel_transforms(self, delta_z: float, pad: float = 2):
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fresnel_transform(self._fields[:, :, i], self._grid, delta_z=delta_z, pad=pad)
        else:
            print("Run compute method first!")

    def compute_fourier_transforms(self, pad: float = 2):
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fourier_transform(self._fields[:, :, i], pad=pad)
        else:
            print("Run compute method first!")

    def compute_fresnel_and_fourier_transforms(self, fresnel_delta_z: float, fourier_pad: float = 2, fresnel_pad: float = 2):
        if self._fields is not None:
            self._transf = np.zeros(shape=(self._fields.shape[0], 2*self._fields.shape[1], self._fields.shape[2]), dtype=np.complex128)
            for i in range(self.length):
                fres = fresnel_transform(self._fields[:, :, i], self._grid, delta_z=fresnel_delta_z, pad=fresnel_pad)
                four = fourier_transform(self._fields[:, :, i], pad=fourier_pad)
                self._transf[:, :, i] = np.concatenate((fres, four), axis=1)
        else:
            print("Run compute method first!")

    def _compute_transfer_matrix(self, dm: MockDeformableMirror, grid: Grid):
        dm.compute_transfer_matrix_amplitudes()
        self._transfer_matrix = np.zeros(shape=(dm._transfer_matrix_amplitudes.shape[0], grid.pixel_numbers[0], grid.pixel_numbers[1]), dtype=np.complex128)
        for i in range(dm._transfer_matrix_amplitudes.shape[0]):
            if self._degenerated:
                coupled_in = GrinFiberDegenCoupler(dm._transfer_matrix_amplitudes[i, ...], grid, fiber=self._fiber, N_modes=self._N_modes)
            else:
                coupled_in = GrinFiberCoupler(dm._transfer_matrix_amplitudes[i, ...], grid, fiber=self._fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)
            self._transfer_matrix[i, :, :] = propagated_field

    @property
    def reshaped_transfer_matrix(self):
        tm = self._transfer_matrix.copy()
        tm = np.swapaxes(self._transfer_matrix, 0, 2)
        tm = np.swapaxes(tm, 0, 1)
        tm = tm.reshape(np.prod(tm.shape[:-1]), tm.shape[-1])
        return tm

    @property
    def length(self):
        return self._length
    
    @property
    def phases_size(self):
        return np.prod(np.array(list(self._phase_dims)))

    @property
    def intensities(self):
        return np.square(np.abs(self._fields))
    
    @staticmethod
    def add_intensity_noise(intens, mu: float = None, sigma: float = None, tau_mu_exp: float = 228, tau_sigma_exp: float = 1.258e-4, stat_func: callable = np.median):
        """Intensity matrix should have size: m x m x N"""
        if mu is None:
            mu = tau_mu_exp * stat_func(np.max(intens, axis=(0,1)))
        if sigma is None:
            sigma = tau_sigma_exp * stat_func(np.max(intens, axis=(0,1)))
        return np.abs(intens + mu + sigma * np.random.randn(*intens.shape))
    
    def export(self, path: str = '.', name: str = None, verbose: bool = True, return_input_fields: bool = False, return_output_fields: bool = False, add_exp_noise: bool = False, noise_func: callable = np.median):
        if name is None:
            default_name = f"synth_dset_grin_Nmodes={self._N_modes}_degen={self._degenerated}_len={self.length}_mirr={self.phases_size}"
            if return_output_fields:
                name = default_name + '_fields'
            if self._transf is not None:
                name = default_name + '_transf'
            name = default_name
            if add_exp_noise:
                name = name + '_exp_noise'
        savename = os.path.join(path, f"{name}.mat")

        coupling_matrix = [] if self._coupling_matrix is None else self._coupling_matrix
        transfer_matrix = [] if self._transfer_matrix is None else self._transfer_matrix
        intens = SimulatedGrinSpeckleOutputDataset.add_intensity_noise(self.intensities, mu=0, stat_func=noise_func) if add_exp_noise else self.intensities

        mdict = {
                    'phase_maps': self._phase_maps, 'intens': intens,
                    'coupling_matrix': coupling_matrix,
                    'transfer_matrix': transfer_matrix,
                    'reshaped_transfer_matrix': self.reshaped_transfer_matrix,
                    'length': self.length, 'N_modes': self._N_modes,
                    'macropixels_energy': self._normalized_energy_on_macropixels,
                }

        if self._transf is not None:
            mdict['transf'] = SimulatedGrinSpeckleOutputDataset.add_intensity_noise(np.square(np.abs(self._transf)), mu=0, stat_func=noise_func) if add_exp_noise else np.square(np.abs(self._transf))
        if return_input_fields:
            mdict['fields'] = self._input_fields
        if return_output_fields:
            mdict['fields'] = self._fields

        savemat(
                file_name = savename,
                mdict = mdict,
            )
        
        if verbose:
            print(f"Dataset saved: {savename}")
    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]


if __name__ == "__main__":
    # grid = Grid(pixel_size=0.5e-6)
    # fiber = GrinFiber()
    # # dset = GrinLPDataset(fiber, grid, N_modes=10)
    # # dset = GrinLPSpeckleDataset(fiber, grid, length=5, N_modes=10)

    # import cProfile as profile
    # import pstats

    # prof = profile.Profile()
    # prof.enable()
    # dset = SimulatedGrinSpeckleOutputDataset(fiber, grid, length=10000, N_modes=55)
    # dset.multiproc_compute(phases_dim=(6,6))
    # # dset.compute(phases_dim=(6,6))
    # dset.export()
    # prof.disable()

    # stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(10) # top 10 rows

    # plt.figure()
    # plt.imshow(dset[0], cmap='gray')
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(dset[3], cmap='gray')
    # plt.colorbar()
    # plt.show()

    dset = RandomDataset(phases_dim=6, intens_dim=96, length=10000)
    dset.export()
    # plt.plot