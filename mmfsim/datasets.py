import os
import multiprocessing
from scipy.io import savemat
from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt
import h5py

from abc import ABC, abstractmethod
from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber, StepIndexFiber
from mmfsim.modes import GrinLPMode, StepIndexLPMode
from mmfsim.speckle import GrinSpeckle, DegenGrinSpeckle, StepIndexSpeckle, DegenStepIndexSpeckle
from mmfsim.beams import GaussianBeam
from mmfsim.devices import MockDeformableMirror
from mmfsim.coupling import GrinFiberCoupler, GrinFiberDegenCoupler, StepIndexFiberCoupler, StepIndexFiberDegenCoupler
from mmfsim.transforms import fresnel_transform, fourier_transform

from waveoptics.utils.utils import slice_elements_by_batch

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



class SimulatedSpeckleOutputDataset:
    """Coupling from modal decomposition on fiber LP modes, then propagation using a random mode coupling matrix
    
       Can use either degenerated modes coupling or non-degenerated mode couplings:
            - With degenerated mode-coupling (default), the system has a an existing transfer matrix since the degenerated mode basis orientation is fixed.
            - With non-degenerated mode-coupling, the system has no existing transfer matrix. The reason is that the degenerated mode basis
              orientation is uncontrolled and might change between each modal decompostion.
    """

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, degen: bool = True) -> None:
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
        self._normalized_energy_on_macropixels = None
        self._degenerated = degen
        self._transfer_matrix = None
        self._input_modes_coeffs_matrix = None
        self._transf = None
        self._low_energy_weights_indexes = None
        self._coupling_matrix = self._fiber.modes_coupling_matrix(complex=complex, full=False, degen=degen)
        self._default_name = f"synth_dset_lambda={self._fiber.wavelength*1e9:.0f}nm_Nmodes={self._N_modes}"
        self._coupling_class = GrinFiberCoupler
        self._coupling_degen_class = GrinFiberDegenCoupler

    def compute(self, phases_dim: tuple[int, int] = (6,6), verbose: bool = True):
        """Computes dataset from random phases applied to partition and their associated fiber output complex field.

           It is slow since the mode decomposition and recomposition is computed for each phase map.
           Use compute_from_transfer_matrix method for a much faster version that provides the same result.
        """
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._input_fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=self._grid.pixel_size, pixel_numbers=self._grid.pixel_numbers, diameter=2*self._fiber.radius)
        beam = GaussianBeam(self._grid)
        beam.compute(width=2*self._fiber.radius)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        for i in range(self.length):
            phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
            dm.apply_phase_map(phase_map)

            if i==0:
                self._compute_transfer_matrix(dm, beam.grid)
                self._normalized_energy_on_macropixels = dm.normalized_energies_on_macropixels

            if self._degenerated:
                coupled_in = self._coupling_degen_class(dm._field_matrix, self._grid, fiber=self._fiber, N_modes=self._N_modes)
            else:
                coupled_in = self._coupling_class(dm._field_matrix, self._grid, fiber=self._fiber, N_modes=self._N_modes)
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)

            self._input_fields[:,:,i] = dm._field_matrix
            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = propagated_field

            if verbose:
                print(f"Computed couple {i+1}/{self.length}")  

    def compute_from_transfer_matrix(self, phases_dim: tuple[int, int] = (6,6), verbose: bool = True, ref_phi: int = None):
        """Computes dataset from random phases applied to partition and their associated fiber output complex field.

           It is fast since the output field is obtained via matrix multiplication from the computed transfer matrix.
        """
        self._phase_dims = phases_dim
        self._phase_maps = np.zeros(shape=(phases_dim + (self.length,)))
        self._fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)
        self._input_fields = np.zeros(shape=(tuple(self._grid.pixel_numbers) + (self.length,)), dtype=np.complex128)

        dm = MockDeformableMirror(pixel_size=self._grid.pixel_size, pixel_numbers=self._grid.pixel_numbers, diameter=2*self._fiber.radius)
        beam = GaussianBeam(self._grid)
        beam.compute(width=2*self._fiber.radius)
        beam.normalize_by_energy()
        dm.apply_amplitude_map(beam.amplitude)

        phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
        dm.apply_phase_map(phase_map)
        self._compute_transfer_matrix(dm, beam.grid)
        tm = self.reshaped_transfer_matrix
        self._normalized_energy_on_macropixels = dm.normalized_energies_on_macropixels

        for i in range(self.length):
            phase_map = -np.pi + 2*np.pi*np.random.rand(*phases_dim)
            if ref_phi is not None:
                phase_map[np.unravel_index(ref_phi, phase_map.shape)] = 0

            x = np.sqrt(self._normalized_energy_on_macropixels) * np.exp(1j * phase_map)
            x = x.flatten()
            if len(self._low_energy_weights_indexes) > 0:
                x = np.delete(x, self._low_energy_weights_indexes)
            y = (tm @ x).reshape(self._grid.pixel_numbers)

            self._phase_maps[:,:,i] = phase_map
            self._fields[:,:,i] = y

            if verbose:
                print(f"Computed couple {i+1}/{self.length}")

    def _compute_transfer_matrix(self, dm: MockDeformableMirror, grid: Grid):
        dm.compute_transfer_matrix_amplitudes()
        self._low_energy_weights_indexes = dm._low_energy_weights_indexes

        self._input_modes_coeffs_matrix = np.zeros(shape=(dm._transfer_matrix_amplitudes.shape[0], self._N_modes), dtype=np.complex128)
        self._transfer_matrix = np.zeros(shape=(dm._transfer_matrix_amplitudes.shape[0], grid.pixel_numbers[0], grid.pixel_numbers[1]), dtype=np.complex128)
        for i in range(dm._transfer_matrix_amplitudes.shape[0]):
            if self._degenerated:
                coupled_in = self._coupling_degen_class(dm._transfer_matrix_amplitudes[i, ...], grid, fiber=self._fiber, N_modes=self._N_modes)
            else:
                coupled_in = self._coupling_class(dm._transfer_matrix_amplitudes[i, ...], grid, fiber=self._fiber, N_modes=self._N_modes)
            self._input_modes_coeffs_matrix[i, :] = coupled_in.modes_coeffs
            propagated_field = coupled_in.propagate(matrix=self._coupling_matrix)
            self._transfer_matrix[i, :, :] = propagated_field
            print(f"Computed TM row {i+1}/{dm._transfer_matrix_amplitudes.shape[0]}")

    def compute_fresnel_transforms(self, delta_z: float, pad: float = 1, verbose: bool = True):
        """Computes the Fresnel transforms of the computed fiber output complex fields."""
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fresnel_transform(self._fields[:, :, i], self._grid, delta_z=delta_z, pad=pad)
                if verbose:
                    print(f"Computed Fresnel {i+1}/{self.length}")
        else:
            raise ValueError("Run compute or compute_from_transfer_matrix method first!")

    def compute_fourier_transforms(self, pad: float = 1, verbose: bool = True):
        """Computes the Fourier transforms of the computed fiber output complex fields."""
        if self._fields is not None:
            self._transf = np.zeros_like(self._fields)
            for i in range(self.length):
                self._transf[:, :, i] = fourier_transform(self._fields[:, :, i], pad=pad)
                if verbose:
                    print(f"Computed Fourier {i+1}/{self.length}")
        else:
            raise ValueError("Run compute or compute_from_transfer_matrix method first!")

    def compute_fresnel_and_fourier_transforms(self, fresnel_delta_z: float, fourier_pad: float = 1, fresnel_pad: float = 1):
        if self._fields is not None:
            self._transf = np.zeros(shape=(self._fields.shape[0], 2*self._fields.shape[1], self._fields.shape[2]), dtype=np.complex128)
            for i in range(self.length):
                fres = fresnel_transform(self._fields[:, :, i], self._grid, delta_z=fresnel_delta_z, pad=fresnel_pad)
                four = fourier_transform(self._fields[:, :, i], pad=fourier_pad)
                self._transf[:, :, i] = np.concatenate((fres, four), axis=1)
        else:
            raise ValueError("Run compute or compute_from_transfer_matrix method first!")

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
    
    def export(self,
               path: str = '.',
               name: str = None,
               max_fields_per_file: int = None,
               verbose: bool = True,
               return_input_fields: bool = False,
               return_output_fields: bool = False,
               add_exp_noise: bool = False,
               noise_func: callable = np.median,
               file_type: str = 'matlab',
               ):
        """ Export the generated dataset to a matfile or numpy file.

            Input arguments:
                - `path`: exported data file base path (optional, str, default = current path)
                - `name`: exported data file base name (optional, str, default = appropriately generated)
                - `return_input_fields`: saves fiber-input optical fields into the data file (optional, bool, default = `False`)
                - `return_output_fields`: saves fiber-output optical fields into the data file (optional, bool, default = `True`)
                - `add_exp_noise`: adds experimental-like noise to intensities (optional, bool, default = `False`)
                - `noise_func`: defines noise function for experimental-like noise application to intensities (optional, callable, default = `np.median`)
                - `file_type`: defines data file type (optional, str, default = `'matlab'`, available = `{'matlab', 'numpy', 'hdf5'}`)

            The exported matfile has the following fields:
                - `phase_maps`: Phase maps used to generate the corresponding fiber-output optical field.
                - `intens`: Intensity of the fiber-output optical field (square modulus).
                - `degenerated_modes`: Boolean indicating if the modes decomposition has been carried on fixed degenerated modes orientations.
                - `coupling_matrix`: Fiber modes-coefficients coupling matrix, that has been used to simulated modes propagation in the fiber.
                - `transfer_matrix`: Transfer matrix in image shape. Has dimensions Nact x Nx x Ny.
                - `reshaped_transfer_matrix`: Reshaped transfer matrix for simple matrix products. Has dimensions Nact x (Nx x Ny).
                - `length`: Dataset length.
                - `wavelength`: Illumination wavelength.
                - `N_modes`: Number of non-degerated LP modes allowed to propagate in the simulated fiber.
                - `macropixels_energy`: Energy E on macropixels for the selected deformable mirror partitionning scheme. Use sqrt(E) weights on phase_maps to replicate output field.
                - `intens_transf`: Optional field. Intensity (square modulus) of the transform (Fresnel or Fourier) of the fiber-output optical field.
                - `fields`: Optional field. Fiber-output optical fields. Returned if `return_output_fields` is set to `True`.
                - `input_fields`: Optional field. Fiber-input optical fields. Returned if `return_input_fields` is set to `True`.
        """

        _allowed_file_types = {'matlab', 'numpy', 'hdf5'}
        if file_type.lower() not in _allowed_file_types:
            raise ValueError(f"Invalid file_type value. Must be in {_allowed_file_types}")
        
        if name is None:
            default_name = f"{self._default_name}_degen={self._degenerated}_len={self.length}_mirr={self.phases_size}"
            if return_output_fields:
                name = default_name + '_fields'
            if self._transf is not None:
                name = default_name + '_transf'
            name = default_name
            if add_exp_noise:
                name = name + '_exp_noise'

        coupling_matrix = [] if self._coupling_matrix is None else self._coupling_matrix
        transfer_matrix = [] if self._transfer_matrix is None else self._transfer_matrix
        intens = SimulatedSpeckleOutputDataset.add_intensity_noise(self.intensities, mu=0, stat_func=noise_func) if add_exp_noise else self.intensities

        mdict = {
                    'phase_maps': self._phase_maps,
                    'intens': intens,
                    'degenerated_modes': self._degenerated,
                    'coupling_matrix': coupling_matrix,
                    'transfer_matrix': transfer_matrix,
                    'reshaped_transfer_matrix': self.reshaped_transfer_matrix,
                    'length': self.length,
                    'N_modes': self._N_modes,
                    'macropixels_energy': self._normalized_energy_on_macropixels,
                    'wavelength': self._fiber.wavelength,
                }

        if self._transf is not None:
            intens_transf = SimulatedSpeckleOutputDataset.add_intensity_noise(np.square(np.abs(self._transf)), mu=0, stat_func=noise_func) if add_exp_noise else np.square(np.abs(self._transf))
            mdict['intens_transf'] = intens_transf
        if return_input_fields:
            mdict['input_fields'] = self._input_fields
        if return_output_fields:
            mdict['fields'] = self._fields

        if max_fields_per_file is None or max_fields_per_file < 2:
            self.__file_saver(
                data_dict=mdict,
                file_type=file_type.lower(),
                path=path,
                name=name,
                verbose=verbose,
                )
        else:
            slice_list = slice_elements_by_batch(total_elements=self.length, slice_size=max_fields_per_file)
            n_slices = len(slice_list)
            for i_slice in range(n_slices):
                mdict['phase_maps'] = self._phase_maps[:, :, slice_list[i_slice]]
                mdict['intens'] = intens[:, :, slice_list[i_slice]]
                mdict['slice_length'] = int(slice_list[i_slice].stop - slice_list[i_slice].start)
                
                if self._transf is not None:
                    mdict['intens_transf'] = intens_transf[:, :, slice_list[i_slice]]
                if return_input_fields:
                    mdict['input_fields'] = self._input_fields[:, :, slice_list[i_slice]]
                if return_output_fields:
                    mdict['fields'] = self._fields[:, :, slice_list[i_slice]]
                
                self.__file_saver(
                    data_dict=mdict,
                    file_type=file_type.lower(),
                    path=path,
                    name=f"{name}_{i_slice + 1}_of_{n_slices}",
                    verbose=verbose,
                    )    
            
    def __file_saver(self, data_dict: dict, file_type: str, path: str, name: str, verbose: bool = True):
        if file_type.lower() == 'matlab':
            savename = os.path.join(path, f"{name}.mat")
            savemat(
                    file_name = savename,
                    mdict = data_dict,
                )
        elif file_type.lower() == 'numpy':
            savename = os.path.join(path, f"{name}.npy")
            np.save(savename, data_dict)
        elif file_type.lower() == 'hdf5':
            savename = os.path.join(path, f"{name}.hdf5")
            with h5py.File(savename, 'w') as hf:
                for key_name in data_dict:
                    hf.create_dataset(name=key_name, data=data_dict[key_name])
                        
        if verbose:
            print(f"Dataset saved: {savename}")
    
    def __getitem__(self, idx):
        return self.intensities[:, :, idx]



class SimulatedGrinSpeckleOutputDataset(SimulatedSpeckleOutputDataset):
    """Coupling from modal decomposition on GRIN fiber LP modes, then propagation using a random mode coupling matrix
    
       Can use either degenerated modes coupling or non-degenerated mode couplings:
            - With degenerated mode-coupling (default), the system has a an existing transfer matrix since the degenerated mode basis orientation is fixed.
            - With non-degenerated mode-coupling, the system has no existing transfer matrix. The reason is that the degenerated mode basis
              orientation is uncontrolled and might change between each modal decompostion.
    """

    def __init__(self, fiber: GrinFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, degen: bool = True) -> None:
        super().__init__(fiber=fiber, grid=grid, length=length, N_modes=N_modes, noise_std=noise_std, degen=degen)
        self._coupling_matrix = self._fiber.modes_coupling_matrix(complex=complex, full=False, degen=degen)
        self._default_name = f"synth_dset_grin_lambda={self._fiber.wavelength*1e9:.0f}nm_Nmodes={self._N_modes}"
        self._coupling_class = GrinFiberCoupler
        self._coupling_degen_class = GrinFiberDegenCoupler



class SimulatedStepIndexSpeckleOutputDataset(SimulatedSpeckleOutputDataset):
    """Coupling from modal decomposition on step index fiber LP modes, then propagation using a random mode coupling matrix
    
       Can use either degenerated modes coupling or non-degenerated mode couplings:
            - With degenerated mode-coupling (default), the system has a an existing transfer matrix since the degenerated mode basis orientation is fixed.
            - With non-degenerated mode-coupling, the system has no existing transfer matrix. The reason is that the degenerated mode basis
              orientation is uncontrolled and might change between each modal decompostion.
    """

    def __init__(self, fiber: StepIndexFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, degen: bool = True) -> None:
        super().__init__(fiber=fiber, grid=grid, length=length, N_modes=N_modes, noise_std=noise_std, degen=degen)
        self._coupling_matrix = self._fiber.modes_coupling_matrix(complex=complex, full=True, degen=degen)
        self._default_name = f"synth_dset_step_lambda={self._fiber.wavelength*1e9:.0f}nm_Nmodes={self._N_modes}"
        self._coupling_class = StepIndexFiberCoupler
        self._coupling_degen_class = StepIndexFiberDegenCoupler



class SimulatedDynamicStepIndexSpeckleOutputDataset(SimulatedStepIndexSpeckleOutputDataset):
    """Coupling from modal decomposition on step index fiber LP modes, then propagation using a random mode coupling matrix

       Can use either degenerated modes coupling or non-degenerated mode couplings:
            - With degenerated mode-coupling (default), the system has a an existing transfer matrix since the degenerated mode basis orientation is fixed.
            - With non-degenerated mode-coupling, the system has no existing transfer matrix. The reason is that the degenerated mode basis
              orientation is uncontrolled and might change between each modal decomposition.
    """

    def __init__(self, fiber: StepIndexFiber, grid: Grid, length: int = 10, N_modes: int = 55, noise_std: float = 0.0, degen: bool = True) -> None:
        super().__init__(fiber=fiber, grid=grid, length=length, N_modes=N_modes, noise_std=noise_std, degen=degen)




    def step(self):
        """Step TM based on dynamic model, readjust properties based on new weights"""
        raise(NotImplementedError)

    def export(self, path: str = '.', name: str = None,
               verbose: bool = True,
               return_input_fields: bool = False,
               return_output_fields: bool = False,
               add_exp_noise: bool = False,
               noise_func: callable = np.median,
               file_type: str = 'matlab',
               ):
        """ Export the generated dataset to a matfile or numpy file.

            Input arguments:
                - `path`: exported data file base path (optional, str, default = current path)
                - `name`: exported data file base name (optional, str, default = appropriately generated)
                - `return_input_fields`: saves fiber-input optical fields into the data file (optional, bool, default = `False`)
                - `return_output_fields`: saves fiber-output optical fields into the data file (optional, bool, default = `True`)
                - `add_exp_noise`: adds experimental-like noise to intensities (optional, bool, default = `False`)
                - `noise_func`: defines noise function for experimental-like noise application to intensities (optional, callable, default = `np.median`)
                - `file_type`: defines data file type (optional, str, default = `'matlab'`, available = `{'matlab', 'numpy', 'hdf5'}`)

            The exported matfile has the following fields:
                - `phase_maps`: Phase maps used to generate the corresponding fiber-output optical field.
                - `intens`: Intensity of the fiber-output optical field (square modulus).
                - `degenerated_modes`: Boolean indicating if the modes decomposition has been carried on fixed degenerated modes orientations.
                - `coupling_matrix`: Fiber modes-coefficients coupling matrix, that has been used to simulated modes propagation in the fiber.
                - `transfer_matrix`: Transfer matrix in image shape. Has dimensions Nact x Nx x Ny.
                - `reshaped_transfer_matrix`: Reshaped transfer matrix for simple matrix products. Has dimensions Nact x (Nx x Ny).
                - `length`: Dataset length.
                - `wavelength`: Illumination wavelength.
                - `N_modes`: Number of non-degerated LP modes allowed to propagate in the simulated fiber.
                - `macropixels_energy`: Energy E on macropixels for the selected deformable mirror partitionning scheme. Use sqrt(E) weights on phase_maps to replicate output field.
                - `intens_transf`: Optional field. Intensity (square modulus) of the transform (Fresnel or Fourier) of the fiber-output optical field.
                - `fields`: Optional field. Fiber-output optical fields. Returned if `return_output_fields` is set to `True`.
                - `input_fields`: Optional field. Fiber-input optical fields. Returned if `return_input_fields` is set to `True`.
        """

        _allowed_file_types = {'matlab', 'numpy', 'hdf5'}
        if file_type.lower() not in _allowed_file_types:
            raise ValueError(f"Invalid file_type value. Must be in {_allowed_file_types}")

        if name is None:
            default_name = f"{self._default_name}_degen={self._degenerated}_len={self.length}_mirr={self.phases_size}"
            if return_output_fields:
                name = default_name + '_fields'
            if self._transf is not None:
                name = default_name + '_transf'
            name = default_name
            if add_exp_noise:
                name = name + '_exp_noise'

        coupling_matrix = [] if self._coupling_matrix is None else self._coupling_matrix
        transfer_matrix = [] if self._transfer_matrix is None else self._transfer_matrix
        if add_exp_noise:
            intens = SimulatedSpeckleOutputDataset.add_intensity_noise(self.intensities, mu=0, stat_func=noise_func)
        else:
            intens = self.intensities

        mdict = {
            'phase_maps': self._phase_maps, 'intens': intens,
            'degenerated_modes': self._degenerated,
            'coupling_matrix': coupling_matrix,
            'transfer_matrix': transfer_matrix,
            'reshaped_transfer_matrix': self.reshaped_transfer_matrix,
            'length': self.length, 'N_modes': self._N_modes,
            'macropixels_energy': self._normalized_energy_on_macropixels,
            'wavelength': self._fiber.wavelength,
        }

        if self._transf is not None:
            if add_exp_noise:
                mdict['intens_transf'] = SimulatedSpeckleOutputDataset.add_intensity_noise(
                    np.square(np.abs(self._transf)), mu=0, stat_func=noise_func
                    )
            else:
                mdict['intens_transf'] = np.square(np.abs(self._transf))
        if return_input_fields:
            mdict['input_fields'] = self._input_fields
        if return_output_fields:
            mdict['fields'] = self._fields

        if file_type.lower() == 'matlab':
            savename = os.path.join(path, f"{name}.mat")
            savemat(
                file_name=savename,
                mdict=mdict,
            )
        elif file_type.lower() == 'numpy':
            savename = os.path.join(path, f"{name}.npy")
            np.save(savename, mdict)
        elif file_type.lower() == 'hdf5':
            savename = os.path.join(path, f"{name}.hdf5")
            with h5py.File(savename, 'w') as hf:
                for key_name in mdict:
                    hf.create_dataset(name=key_name, data=mdict[key_name])
        else:
            raise ValueError(f"Invalid file_type value. Must be in {_allowed_file_types}")

        if verbose:
            print(f"Dataset saved: {savename}")




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
