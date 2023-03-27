import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from grid import Grid
import beams as beams
import matrix as matproc


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

    def __init__(self, pixel_size: float = 1e-6, pixel_numbers: tuple[int, int] = (128, 128), offsets: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__(pixel_size, pixel_numbers, offsets)
        self._field_matrix = None
        self._field_matrix = self._init_field_matrix()
        self._phase_map = None

    def _init_field_matrix(self) -> np.ndarray:
        moduli = np.ones(shape=tuple(self.pixel_numbers))
        phases = np.zeros(shape=tuple(self.pixel_numbers))
        return moduli * np.exp(1j * phases)

    def apply_mask(self, matrix: np.ndarray, mask_value: float = 0):
        mask = np.zeros(shape=self.pixel_numbers, dtype=bool)
        mask[self.R > deformable_mirror_diameter/2] = True
        matrix[mask] = mask_value
        return matrix

    def apply_phase_map(self, phase_map):
        if phase_map.shape != tuple(self.pixel_numbers):
            self._phase_map = phase_map
            phase_map = self._partition_to_matrix(phase_map)
        self._field_matrix = np.abs(self._field_matrix) * np.exp(1j * phase_map)
        self._field_matrix = self.apply_mask(self._field_matrix)

    def apply_amplitude_map(self, amplitude_map):
        if amplitude_map.shape != tuple(self.pixel_numbers):
            amplitude_map = self._partition_to_matrix(amplitude_map)
        self._field_matrix = np.abs(amplitude_map) * np.exp(1j * self.phase)
        self._field_matrix = self.apply_mask(self._field_matrix)

    def apply_complex_map(self, complex_map):
        if complex_map.shape != tuple(self.pixel_numbers):
            complex_map = self._partition_to_matrix(complex_map)
        self._field_matrix = self.apply_mask(complex_map)

    def _partition_to_matrix(self, partition: np.ndarray):
        repeater_axis0 = int(np.ceil(deformable_mirror_diameter / self.pixel_size / partition.shape[0]))
        repeater_axis1 = int(np.ceil(deformable_mirror_diameter / self.pixel_size / partition.shape[1]))
        matrix = np.repeat(partition, repeater_axis0, axis=0)
        matrix = np.repeat(matrix, repeater_axis1, axis=1)
        pad_amount = int((self.pixel_numbers[0] - matrix.shape[0]) // 2)
        return np.pad(matrix, pad_width=pad_amount)

    @property
    def field(self):
        return self._field_matrix

    @property
    def amplitude(self):
        return np.abs(self._field_matrix)
    
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
    phase_map = 2*np.pi*np.random.rand(6,6)

    # dm = DeformableMirror(pixel_numbers=(128,128))
    # dm = DeformableMirror()
    # dm.apply_phase_map(phase_map)
    # grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
    # beam = beams.BesselBeam(grid)
    # beam.compute(amplitude=1, width=1500e-6, centers=[0,0], order=1)
    # dm.apply_amplitude_map(beam.amplitude)
    
    # dm.plot()
    # plt.show()



    dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
    dm.apply_phase_map(phase_map)

    grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
    beam = beams.GaussianBeam(grid)
    beam.compute(amplitude=1, width=3500e-6, centers=[0,0])
    dm.apply_amplitude_map(beam.amplitude)
    
    dm.plot()
    plt.show()

