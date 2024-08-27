import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from mmfsim.grid import Grid
import mmfsim.beams as beams
import mmfsim.matrix as matproc


deformable_mirror_diagonal_count = 34
deformable_mirror_actuator_size = 300e-6
deformable_mirror_diameter = deformable_mirror_diagonal_count * deformable_mirror_actuator_size


def load_nan_mask() -> np.array:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = 'nan_mask_34x34.npy'
    return np.load(os.path.join(path, 'imports', name))


class DeformableMirror(Grid):

    def __init__(self, pixel_size: float = 300e-6, pixel_numbers: tuple[int, int] = (34, 34), offsets: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__(pixel_size, pixel_numbers, offsets)
        self._nan_mask_matrix = load_nan_mask()
        self._field_matrix = None
        self._field_matrix = self._init_field_matrix()
        self._phase_map = None

    def _init_field_matrix(self) -> np.ndarray:
        moduli = np.ones(shape=tuple(self.pixel_numbers))
        phases = np.zeros(shape=tuple(self.pixel_numbers))
        return moduli * np.exp(1j * phases)

    def apply_nan_mask_matrix(self, matrix: np.ndarray, nan_value: float = np.nan):
        matrix[self._nan_mask_matrix] = nan_value
        return matrix

    def apply_phase_map(self, phase_map):
        if phase_map.shape != tuple(self.pixel_numbers):
            self._phase_map = phase_map
            phase_map = self._partition_to_matrix(phase_map)
        self._field_matrix = np.abs(self._field_matrix) * np.exp(1j * phase_map)

    def apply_amplitude_map(self, amplitude_map):
        if amplitude_map.shape != tuple(self.pixel_numbers):
            amplitude_map = self._partition_to_matrix(amplitude_map)
        self._field_matrix = np.abs(amplitude_map) * np.exp(1j * self.phase)

    def apply_complex_map(self, complex_map):
        if complex_map.shape != tuple(self.pixel_numbers):
            complex_map = self._partition_to_matrix(complex_map)
        self._field_matrix = complex_map

    def _partition_to_matrix(self, partition: np.ndarray):
        matrix = matproc.partition_to_matrix(partition, self.field)
        matrix = matproc.crop_center(matrix, self.pixel_numbers[0])
        return matrix
    
    def retrieve_phase_map(self, partition_size: tuple[int, int]):
        if phase_map.shape == tuple(self.pixel_numbers):
            return self.phase
        else:
            return matproc.matrix_to_partition(self.phase, partition_size)
    
    def export_to_grid(self, grid: Grid, beam_type: beams.Beam, beam_kwargs):
        beam = beam_type(grid)
        beam.compute(**beam_kwargs)
        amplitude = beam.amplitude

        phases = np.zeros_like(amplitude)
        if grid.pixel_size == self.pixel_size:
            phi = self.apply_nan_mask_matrix(self.phase, nan_value=0)
            pad_amount = int((grid.pixel_numbers[0] - self.pixel_numbers[0]) // 2)
            phases = np.pad(phi, pad_width=pad_amount)
        else:
            partition = self._phase_map
            repeater = int(np.ceil(self.pixel_size / grid.pixel_size * partition.shape[0]))
            matrix = np.repeat(partition, repeater, axis=0)
            phi = np.repeat(matrix, repeater, axis=1)
            pad_amount = int((grid.pixel_numbers[0] - phi.shape[0]) // 2)
            phases = np.pad(phi, pad_width=pad_amount)
        return amplitude * np.exp(1j * phases)
    
    def export_to_beam(self, beam: beams.Beam, keep_beam_phases: bool=False):
        phases = np.zeros_like(beam.amplitude)
        if beam.grid.pixel_size == self.pixel_size:
            phi = self.apply_nan_mask_matrix(self.phase, nan_value=0)
            pad_amount = int((beam.grid.pixel_numbers[0] - self.pixel_numbers[0]) // 2)
            phases = np.pad(phi, pad_width=pad_amount)
        else:
            partition = self._phase_map
            repeater = int(np.ceil(self.pixel_size / beam.grid.pixel_size * partition.shape[0]))
            matrix = np.repeat(partition, repeater, axis=0)
            phi = np.repeat(matrix, repeater, axis=1)
            pad_amount = int((beam.grid.pixel_numbers[0] - phi.shape[0]) // 2)
            phases = np.pad(phi, pad_width=pad_amount)
        
        if keep_beam_phases:
            beam.field = beam.field * np.exp(1j * phases)
        else:
            beam.field = beam.amplitude * np.exp(1j * phases)
        return beam
        
    @staticmethod
    def vec952_to_mat34x34(vec, nan_mask_34x34, order: str = 'F'):
        mask_flat = nan_mask_34x34.flatten(order=order)
        flat_map = np.zeros(shape=(34*34))
        flat_map[~mask_flat] = vec
        flat_map[mask_flat] = np.nan
        return np.reshape(flat_map, newshape=(34,34), order=order)
    
    @staticmethod
    def rad_to_volt(vector: np.ndarray, coeff: float = 14.6, offset: float = 50.0):
        return (vector * coeff) / (2 * np.pi) + offset

    @property
    def field(self):
        return self._field_matrix

    @property
    def full_field_vector(self):
        return matproc.matrix_to_vector(self.field)
    
    @property
    def field_vector(self):
        vector = self.full_field_vector
        return vector[~np.isnan(vector)]

    @property
    def amplitude(self):
        return np.abs(self._field_matrix)
    
    @property
    def intensity(self):
        return np.square(self.amplitude)
    
    @property
    def phase(self):
        return np.angle(self._field_matrix)
    
    def plot(self, show_extent: bool = True):
        fig, axs = plt.subplots(1, 2, figsize=(13,4))
        if show_extent:
            extent = np.array([np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)]) * 1e6
            pl0 = axs[0].imshow(self.apply_nan_mask_matrix(self.intensity), extent=extent, cmap="hot")
            pl1 = axs[1].imshow(self.apply_nan_mask_matrix(self.phase), extent=extent, cmap="twilight")
            axs[0].set_xlabel("x [um]")
            axs[1].set_xlabel("x [um]")
            axs[0].set_ylabel("y [um]")
            axs[1].set_ylabel("y [um]")
        else:
            pl0 = axs[0].imshow(self.apply_nan_mask_matrix(self.intensity), cmap="hot")
            pl1 = axs[1].imshow(self.apply_nan_mask_matrix(self.phase), cmap="twilight")
        axs[0].set_title(f"Intensity on mirror plane")
        axs[1].set_title(f"Phase on mirror plane")
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
        )
    

class MockDeformableMirror(Grid):

    def __init__(self, pixel_size: float = 1e-6, pixel_numbers: tuple[int, int] = (128, 128), offsets: tuple[float, float] = (0.0, 0.0), diameter: float = None) -> None:
        super().__init__(pixel_size, pixel_numbers, offsets)
        self._mirror_diameter = diameter if diameter is not None else deformable_mirror_diameter
        self._phase_map = None
        self._partition_size = None
        self._macropixel_size = None
        self._partitions_idxs = None
        self._masked_macropixels_counts = None
        self._energy_integrated_on_macropixels = None
        self._mask = None
        self._correcting_pad = None
        self._idxs_to_keep = None
        self._transfer_matrix_amplitudes = None
        self._low_energy_weights_indexes = None
        self._field_matrix = self._init_field_matrix()

    def _init_field_matrix(self) -> np.ndarray:
        moduli = np.ones(shape=tuple(self.pixel_numbers))
        phases = np.zeros(shape=tuple(self.pixel_numbers))
        field = moduli * np.exp(1j * phases)
        field = self.apply_mask(field)
        return field

    def apply_mask(self, matrix: np.ndarray, mask_value: float = 0):
        mask = np.zeros(shape=self.pixel_numbers, dtype=bool)
        mask[self.R > self._mirror_diameter/2] = True
        matrix[mask] = mask_value
        return matrix

    def apply_phase_map(self, phase_map):
        if phase_map.shape != tuple(self.pixel_numbers):
            recompute_idxs = True if (phase_map.shape != self._partition_size) else False
            self._partition_size = phase_map.shape
        
        self._phase_map = phase_map
        phase_map = self._partition_to_matrix(phase_map, recompute_idxs)
        self._field_matrix = np.abs(self._field_matrix) * np.exp(1j * phase_map)
        self._field_matrix = self.apply_mask(self._field_matrix)
        self._correct_coherent_zeroes()

    def apply_amplitude_map(self, amplitude_map):
        if amplitude_map.shape != tuple(self.pixel_numbers):
            amplitude_map = self._partition_to_matrix(amplitude_map)
        
        self._field_matrix = np.abs(amplitude_map) * np.exp(1j * self.phase)
        self._field_matrix = self.apply_mask(self._field_matrix)
        # self._correct_coherent_zeroes()

    def apply_complex_map(self, complex_map):
        if complex_map.shape != tuple(self.pixel_numbers):
            complex_map = self._partition_to_matrix(complex_map)
        
        self._field_matrix = self.apply_mask(complex_map)
        # self._correct_coherent_zeroes()

    def _partition_to_matrix(self, partition: np.ndarray, recompute_partitions_idxs: bool = False):
        if recompute_partitions_idxs:
            self._compute_partitions_idxs()
            self._count_partitions_pixels()
            self._macropixels_integrated_energies()
        matrix = np.repeat(partition, self._macropixel_size, axis=0)
        matrix = np.repeat(matrix, self._macropixel_size, axis=1)
        pad_amount = int((self.pixel_numbers[0] - matrix.shape[0]) // 2)
        self._correcting_pad = pad_amount
        matrix = np.pad(matrix, pad_width=pad_amount)
        if matrix.shape != tuple(self.pixel_numbers):
            matrix = np.vstack([matrix, np.zeros(shape=(1, matrix.shape[1]))])
            matrix = np.hstack([matrix, np.zeros(shape=(matrix.shape[0], 1))])
        return matrix
    
    def _correct_coherent_zeroes(self):
        pad = self._correcting_pad
        self._field_matrix[:pad, :] = 0
        self._field_matrix[:, :pad] = 0
        self._field_matrix[-pad:, :] = 0
        self._field_matrix[:, -pad:] = 0
    
    def _compute_partitions_idxs(self):
        self._macropixel_size = int(np.floor(self._mirror_diameter / self.pixel_size / self._partition_size[0]))
        offset = int((self.pixel_numbers[0] - self._partition_size[0] * self._macropixel_size) // 2)

        # Partitions idxs dimensions: partition size 0, partition size 1, macropixel size, mpx rows idxs, cols idxs
        self._partitions_idxs = np.empty(shape=(self._partition_size[0], self._partition_size[1], self._macropixel_size**2))
        self._partitions_idxs[:] = np.nan

        for mpx_row in range(self._partition_size[0]):
            rows = np.arange(0, self._macropixel_size) + mpx_row * self._macropixel_size + offset
            for mpx_col in range(self._partition_size[1]):
                cols = np.arange(0, self._macropixel_size) + mpx_col * self._macropixel_size + offset
                mpx_lin_idxs = np.ravel_multi_index(np.meshgrid(rows, cols), dims=tuple(self.pixel_numbers))
                self._partitions_idxs[mpx_row, mpx_col, :] = mpx_lin_idxs.flatten().astype(int)
        self._partitions_idxs = self._partitions_idxs.astype(int)

    def _count_partitions_pixels(self):
        mask = np.zeros(shape=self.pixel_numbers, dtype=bool)
        mask[self.R <= deformable_mirror_diameter/2] = True
        idxs_to_keep = np.argwhere(mask) # Row and col idxs
        idxs_to_keep = np.ravel_multi_index([idxs_to_keep[:,0], idxs_to_keep[:,1]], dims=tuple(self.pixel_numbers)) # Linear idxs
        
        self._masked_macropixels_counts = np.zeros(shape=self._partition_size)
        for mpx_row in range(self._partition_size[0]):
            for mpx_col in range(self._partition_size[1]):
                count = np.intersect1d(self._partitions_idxs[mpx_row, mpx_col, :], idxs_to_keep).size
                self._masked_macropixels_counts[mpx_row, mpx_col] = count
        self._idxs_to_keep = idxs_to_keep

    def _macropixels_integrated_energies(self):
        self._energy_on_macropixels = np.zeros(shape=self._partition_size)
        for mpx_row in range(self._partition_size[0]):
            for mpx_col in range(self._partition_size[1]):
                mpx_idxs = np.intersect1d(self._partitions_idxs[mpx_row, mpx_col, :], self._idxs_to_keep)
                self._energy_on_macropixels[mpx_row, mpx_col] = np.sum(np.square(np.abs(self._field_matrix.flatten()[mpx_idxs])))
        
    def compute_transfer_matrix_amplitudes(self, trsh: float = 0.001):
        self._transfer_matrix_amplitudes = np.empty(shape=(*self._partition_size, *self._field_matrix.shape))
        for mpx_row in range(self._partition_size[0]):
            for mpx_col in range(self._partition_size[1]):
                mpx_row_idxs, mpx_col_idxs = np.unravel_index(self._partitions_idxs[mpx_row, mpx_col, :], shape=self._field_matrix.shape)
                amplitudes = np.zeros(shape=self._field_matrix.shape)
                amplitudes[mpx_row_idxs, mpx_col_idxs] = np.abs(self._field_matrix[mpx_row_idxs, mpx_col_idxs])
                self._transfer_matrix_amplitudes[mpx_row, mpx_col, :, :] = amplitudes
        self._transfer_matrix_amplitudes = np.reshape(self._transfer_matrix_amplitudes, (np.prod(self._partition_size), *self._field_matrix.shape))
        self._filter_transfer_matrix_amplitudes(trsh=trsh)

    def _filter_transfer_matrix_amplitudes(self, trsh: float = 0.001):
        sums_on_mpxs = np.sum(np.square(self._transfer_matrix_amplitudes), axis=(1,2))
        self._low_energy_weights_indexes = np.argwhere(sums_on_mpxs / np.max(sums_on_mpxs) < trsh)

        if len(self._low_energy_weights_indexes) > 0:
            print(f"Found {len(self._low_energy_weights_indexes)} input variable weights below threshold {trsh} to delete.")
            self._transfer_matrix_amplitudes = np.delete(self._transfer_matrix_amplitudes, self._low_energy_weights_indexes, axis=0)
            print(f"Successfully deleted low weight input variables.")


    @property
    def field(self):
        return self._field_matrix

    @property
    def amplitude(self):
        return np.abs(self._field_matrix)
    
    @property
    def energy_integrated_on_macropixels(self):
        return self._energy_on_macropixels
    
    @property
    def masked_macropixels_counts(self):
        return self._masked_macropixels_counts
    
    @property
    def normalized_energies_on_macropixels(self):
        return self.energy_integrated_on_macropixels / np.max(self.energy_integrated_on_macropixels)

    @property
    def intensity(self):
        return np.square(self.amplitude)
    
    @property
    def phase(self):
        return np.angle(self._field_matrix)
    
    def plot(self, cmap: str = "gray", show_extent: bool = True, apply_mask: bool=True):
        intens = self.apply_mask(self.intensity) if apply_mask else intens
        phase = self.apply_mask(self.phase) if apply_mask else phase

        fig, axs = plt.subplots(1, 2, figsize=(13,4))
        if show_extent:
            extent = np.array([np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)]) * 1e6
            pl0 = axs[0].imshow(intens, extent=extent, cmap=cmap)
            pl1 = axs[1].imshow(phase, extent=extent, cmap="twilight")
            axs[0].set_xlabel("x [um]")
            axs[1].set_xlabel("x [um]")
            axs[0].set_ylabel("y [um]")
            axs[1].set_ylabel("y [um]")
        else:
            pl0 = axs[0].imshow(intens, cmap=cmap)
            pl1 = axs[1].imshow(phase, cmap="twilight")
        axs[0].set_title(f"Intensity on mirror plane")
        axs[1].set_title(f"Phase on mirror plane")
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
        )


    


if __name__ == "__main__":
    phase_map = 2*np.pi*np.random.rand(3,3)

    dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
    grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
    beam = beams.GaussianBeam(grid)
    beam.compute(amplitude=1, width=5100e-6, centers=[0,0])
    beam.normalize_by_energy()
    dm.apply_amplitude_map(beam.amplitude)
    dm.apply_phase_map(phase_map)
    print(dm.normalized_energies_on_macropixels)

    dm.compute_transfer_matrix_amplitudes() 
    dm.plot()
    plt.show()

