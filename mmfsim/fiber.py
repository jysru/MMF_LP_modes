import numpy as np
import matplotlib.pyplot as plt

from mmfsim import matrix as matproc
from mmfsim.plots import complex_image


class GrinFiber:

    def __init__(
        self,
        radius: float = 26e-6,
        wavelength: float = 1064e-9,
        n1: float = 1.465,
        n2: float = 1.45
        ) -> None:
        
        self.radius = radius
        self.wavelength = wavelength
        self.n1 = n1
        self.n2 = n2
        self._neff_hnm = None
        self._compute_modes_constants()

    def _compute_modes_constants(self) -> None:
        storage = np.zeros(shape=(self._N_modes**2 + 2, 4), dtype='float') # Columns with n_eff, h, n, m
        storage = []
        k = 0
        for m in range(1,self._N_modes):
            for n in range(self._N_modes):
                h = self._h_vs_nm(n, m)
                beta = self._k0 * self.n1 - h * np.sqrt(2 * self._delta) / self.radius
                n_eff = beta * self.wavelength / (2 * np.pi)
                storage.append((n_eff, h, n, m))
                k += 1

        sorted_storage = sorted(storage, reverse=True)
        sorted_storage = sorted_storage[:self._N_modes] if len(sorted_storage) > self._N_modes else sorted_storage
        self._neff_hnm = np.asarray(sorted_storage)
    
    def _h_vs_nm(self, n, m):
        return 2 * n + m - 1
    
    def modes_coupling_matrix(self, complex: bool = False, degen: bool = False, full: bool = False, decay_width: float = None):
        if full:
            n_modes = self._N_modes_degen if degen else self._N_modes
            return matproc.square_random_toeplitz(n_modes, complex=complex, decay_width=decay_width)
        else:
            return self._group_coupling_matrix(complex=complex, degen=degen)

    def _group_coupling_matrix(self, complex: bool = False, degen: bool = False):
        groups_neff, groups_indexes, groups_counts = np.unique(self._neff_hnm[:,0], return_index=True, return_counts=True)
        groups_neff = np.flip(groups_neff)
        groups_indexes = np.flip(groups_indexes)
        groups_counts = np.flip(groups_counts)

        dtype = np.complex128 if complex else np.float64
        if degen:
            matrix = np.zeros(shape=(self._N_modes_degen, self._N_modes_degen), dtype=dtype)
            degens_counter = 0
            for i, count in enumerate(groups_counts):
                if i < groups_neff.shape[0] - 1: 
                    degens_in_group = np.sum(self._neff_hnm[groups_indexes[i]:groups_indexes[i+1], 2] > 0)
                else:
                    degens_in_group = np.sum(self._neff_hnm[groups_indexes[i]:, 2] > 0)
                idx = groups_indexes[i] + degens_counter
                matrix[idx:idx+count+degens_in_group, idx:idx+count+degens_in_group] = matproc.square_random_toeplitz(count+degens_in_group, complex=complex)
                degens_counter += degens_in_group
        else:
            matrix = np.zeros(shape=(self._N_modes, self._N_modes), dtype=dtype)
            for i, count in enumerate(groups_counts):
                idx = groups_indexes[i]
                matrix[idx:idx+count, idx:idx+count] = matproc.square_random_toeplitz(count, complex=complex)
        return matrix

    @property
    def _NA(self):
        return np.sqrt(np.square(self.n1) - np.square(self.n2))

    @property
    def _V(self):
        return 2 * np.pi * self.radius * self._NA / self.wavelength
    
    @property
    def _N_modes(self):
        return np.floor(np.square(self._V) / 16).astype(int)
    
    @property
    def _N_modes_degen(self):
        centros = np.sum(self._neff_hnm[:, 2] == 0)
        non_centros = np.sum(self._neff_hnm[:, 2] > 0)
        return centros + 2 * non_centros

    @property
    def _k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def _delta(self):
        return np.square(self._NA) / (2 * np.square(self.n1))
    
    @staticmethod
    def plot_coupling_matrix(matrix, cmap: str = 'gray', complex: bool = False, complex_hsv: bool = False):
        if complex:
            if complex_hsv:
                fig = plt.figure()
                ax = plt.gca()
                pl = plt.imshow(complex_image(matrix))
                ax.set_xlabel("Input mode index")
                ax.set_ylabel("Output mode index")
                ax.set_title(f"GRIN fiber coupling matrix (complex)")
                return (fig, ax, pl)
            else:
                fig, axs = plt.subplots(1, 2, figsize=(13,4))
                pl0 = axs[0].imshow(np.abs(matrix), cmap=cmap)
                pl1 = axs[1].imshow(np.angle(matrix), cmap="twilight")
                axs[0].set_xlabel("Input mode index")
                axs[0].set_ylabel("Output mode index")
                axs[1].set_xlabel("Input mode index")
                axs[1].set_ylabel("Output mode index")
                axs[0].set_title(f"GRIN fiber coupling matrix (amplitude)")
                axs[1].set_title(f"GRIN fiber coupling matrix (phase)")
                plt.colorbar(pl0, ax=axs[0])
                plt.colorbar(pl1, ax=axs[1])
                return (fig, axs, [pl0, pl1])
        else:
            fig = plt.figure()
            ax = plt.gca()
            pl = plt.imshow(np.abs(matrix), cmap=cmap)
            ax.set_xlabel("Input mode index")
            ax.set_ylabel("Output mode index")
            ax.set_title(f"GRIN fiber coupling matrix (amplitude)")
            plt.colorbar(pl, ax=ax)
            return (fig, ax, pl)
    
    def __str__(self):
        return (
        f"{__class__.__name__} instance with:\n"
        f"  - Radius: {self.radius * 1e6} um\n"
        f"  - Core index: {self.n1}\n"
        f"  - Cladding index: {self.n2}\n"
        f"  - Wavelength: {self.wavelength * 1e9} nm\n"
        f"  - Numerical aperture: {self._NA:.3f}\n"
        f"  - Number of guided LP modes: {self._N_modes}\n"
        f"  - Number of guided LP modes (counting degenerates): {self._N_modes_degen}\n"
        f"  - First 10 LP_n,m modes characteristics:\n"
        f"  n_eff      h          n          m\n"
        f"{self._neff_hnm[:10]}"
        )


class StepIndexFiber(GrinFiber):
    
    def __init__(self, radius: float = 6.5e-6, wavelength: float = 1064e-9, n1: float = 1.465, n2: float = 1.445) -> None:
        super().__init__(radius, wavelength, n1, n2)

    @property
    def _N_modes(self):
        return np.floor(np.square(self._V) / 8).astype(int)


if __name__ == "__main__":
    fiber = GrinFiber()
    matrix = fiber.modes_coupling_matrix(complex=True, degen=True)
    GrinFiber.plot_coupling_matrix(matrix, complex=True)
    plt.show()
    