import numpy as np
import matplotlib.pyplot as plt

import matrix as matproc
from plots import complex_image


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
    
    def modes_coupling_matrix(self, complex: bool = False):
        groups_neff, groups_indexes, groups_counts = np.unique(self._neff_hnm[:,0], return_index=True, return_counts=True)
        groups_neff = np.flip(groups_neff)
        groups_indexes = np.flip(groups_indexes)
        groups_counts = np.flip(groups_counts)

        dtype = np.complex128 if complex else np.float64
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
                axs[1].set_ylabel("Output mode index")
                axs[0].set_xlabel("Input mode index")
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
        return f"""
        {self._neff_hnm[:10]}
        """


if __name__ == "__main__":
    fiber = GrinFiber()
    print(fiber._N_modes)
    matrix = fiber.modes_coupling_matrix(complex=True)
    GrinFiber.plot_coupling_matrix(matrix, complex=True)
    plt.show()
    