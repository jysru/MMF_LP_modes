import os
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from grid import Grid


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

    def _init_field_matrix(self):
        moduli = np.ones(shape=tuple(self.pixel_numbers))
        phases = np.zeros(shape=tuple(self.pixel_numbers))
        field = moduli * np.exp(1j * phases)
        return self.apply_nan_mask_matrix(field)

    def apply_nan_mask_matrix(self, matrix: np.ndarray,):
        matrix[self._nan_mask_matrix] = np.nan
        return matrix

    def matrix_to_vector(self, matrix: np.ndarray, order: str = 'F') -> np.array:
        return np.reshape(matrix, newshape=(-1,), order=order)
    
    def vector_to_matrix(self, vector, order: str = 'F') -> np.array:
        return np.reshape(vector, newshape=self.pixel_numbers, order=order)

    def apply_phase_map(self, phase_map):
        if phase_map.shape != tuple(self.pixel_numbers):
            phase_map = self._partition_to_matrix(phase_map)
        field = np.abs(self._field_matrix) * np.exp(1j * phase_map)
        self._field_matrix = self.apply_nan_mask_matrix(field)

    def _partition_to_matrix(self, partition: np.ndarray):
        repeater = int(np.ceil(self.pixel_numbers[0] / partition.shape[0]))
        matrix = np.repeat(partition, repeater, axis=0)
        matrix = np.repeat(matrix, repeater, axis=1)
        matrix = DeformableMirror.crop_center(matrix, self.pixel_numbers[0])
        return matrix
    
    def retrieve_phase_map(self, partition_size: tuple[int, int]):
        if phase_map.shape == tuple(self.pixel_numbers):
            return self.phase
        else:
            return DeformableMirror._matrix_to_partition(self.phase, partition_size)
    
    @staticmethod
    def _matrix_to_partition(matrix, partition_size: tuple[int, int]):
        matrix = DeformableMirror._inverse_repeat(matrix, repeats=partition_size[0], axis=0)
        return DeformableMirror._inverse_repeat(matrix, repeats=partition_size[1], axis=1)

    @staticmethod
    def _inverse_repeat(matrix, repeats, axis):
        if isinstance(repeats, int):
            indices = np.arange(matrix.shape[axis] / repeats, dtype=int) * repeats
        else:  # assume array_like of int
            indices = np.cumsum(repeats) - 1
        return matrix.take(indices, axis)

    @staticmethod
    def crop_center(img, crop):
        y, x = img.shape
        startx = x//2 - crop//2
        starty = y//2 - crop//2
        return img[starty:starty+crop, startx:startx+crop]

    @staticmethod
    def vec952_to_mat34x34(vec, nan_mask_34x34, order: str = 'F'):
        mask_flat = nan_mask_34x34.flatten(order=order)
        flat_map = np.zeros(shape=(34*34))
        flat_map[~mask_flat] = vec
        flat_map[mask_flat] = np.nan
        return np.reshape(flat_map, newshape=(34,34), order=order)

    @staticmethod
    def vec32_to_vec36(vec):
        vec = np.ravel(vec)
        phases = np.angle(vec)
        return np.ravel(np.r_[
            np.r_[[0], phases[:4], [0]][None], 
            phases[4:-4].reshape(4, 6), 
            np.r_[[0], phases[-4:], [0]][None],
        ])[None]
    
    @staticmethod
    def rad_to_volt(vector: np.ndarray, coeff: float = 14.6, offset: float = 50.0):
        return (vector * coeff) / (2 * np.pi) + offset

    @property
    def field(self):
        return self._field_matrix

    @property
    def full_field_vector(self):
        return self.matrix_to_vector(self.field)
    
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
    
    def plot(self):
        extent = np.array([np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)]) * 1e6
        fig, axs = plt.subplots(1, 2, figsize=(15,4))
        pl0 = axs[0].imshow(self.intensity, extent=extent, cmap="hot")
        pl1 = axs[1].imshow(self.phase, extent=extent, cmap="twilight")
        axs[0].set_xlabel("x [um]")
        axs[1].set_xlabel("x [um]")
        axs[0].set_ylabel("y [um]")
        axs[1].set_ylabel("y [um]")
        axs[0].set_title(f"Intensity on mirror plane")
        axs[1].set_title(f"Phase on mirror plane")
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])

    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
        )







# def reconstruct_phases_34x34(vec, nan_mask):
#     phases = np.angle(vec)
#     phases = np.ravel(phases)
#     grid = np.array([ 0,  5, 11, 17, 23, 29, 34])

#     square_phases = np.r_[
#         np.r_[[0], phases[:4], [0]][None], 
#         phases[4:-4].reshape(4, 6), 
#         np.r_[[0], phases[-4:], [0]][None],
#     ]
#     domain_phases = np.zeros((34, 34))

#     for hi in range(len(grid) - 1):
#         for wi in range(len(grid) - 1):
#             domain_phases[grid[hi]:grid[hi + 1], grid[wi]:grid[wi + 1]] = square_phases[hi, wi]

#     domain_phases[nan_mask] = np.nan
#     return domain_phases

# def reduce_2d(domains, reduce_grid=None, agg_func=None):
#     domain = domains[0]
#     if reduce_grid is None:
#         reduce_grid = make_reduce_grid_2d(domain)
#     if agg_func is None:
#         agg_func = np.nanmean

#     h_grid, w_grid = reduce_grid

#     reduced_h, reduced_w = len(h_grid) - 1, len(w_grid) - 1
#     reduced_domains = np.zeros(
#         (len(domains), reduced_h, reduced_w),
#         dtype=domain.dtype,
#     )
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=RuntimeWarning)
#         for hi in range(reduced_h):
#             for wi in range(reduced_w):
#                 reduce_areas = domains[:, h_grid[hi]:h_grid[hi + 1], w_grid[wi]:w_grid[wi + 1]]
#                 reduced_domains[:, hi, wi] = agg_func(reduce_areas, axis=(-2, -1))
#         return reduced_domains




    


if __name__ == "__main__":
    dm = DeformableMirror()
    phase_map = 2*np.pi*np.random.rand(6,6)
    dm.apply_phase_map(phase_map)

    dm.plot()
    plt.show()