import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import root

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

    def _compute_modes_constants(self, froot_tol=1e-9) -> None:
        modes = self._solve_dispersion_equation(tol=froot_tol)
        modes = self._sort_solutions(modes)
        self._neff_hnm = self._compute_neff_hnm_array(modes)
        self._prop_constants = self._compute_prop_constants_array(modes)

    def _solve_dispersion_equation(self, tol=1e-9) -> dict:
        v, lbda = self._V, self.wavelength
        n, roots = 0, [0]
        modes = {'beta': [], 'u': [], 'w': [], 'n': [], 'm': [], 'number': 0}
        interval = np.arange(np.spacing(10), v - np.spacing(10), v * 1e-4)

        while len(roots):
            def root_func(u):
                w = np.sqrt(v**2 - u**2)
                return sp.jv(n, u) / (u * sp.jv(n - 1, u)) + sp.kn(n, w) / (w * sp.kn(n - 1, w))

            guesses = np.argwhere(np.abs(np.diff(np.sign(root_func(interval)))))
            froot = lambda x0: root(root_func, x0, tol=tol)
            sols = map(froot, interval[guesses])
            roots = [s.x for s in sols if s.success]

            # Remove solution outside the valid interval, round the solutions and remove duplicates
            roots = np.unique(
                [np.round(r / tol) * tol for r in roots if (r > 0 and r < v)]
            ).tolist()
            roots_num = len(roots)

            if roots_num:
                modes['beta'] = modes['beta'] + [np.sqrt(np.square(2 * np.pi / lbda * self.n1) - np.square(r / self.radius)) for r in roots]
                modes['u'] = modes['u'] + roots
                modes['w'] = modes['w'] + [np.sqrt(np.square(v) - np.square(r)) for r in roots]
                modes['number'] += roots_num
                modes['n'] = modes['n'] + [n] * roots_num
                modes['m'] = modes['m'] + [x + 1 for x in range(roots_num)]
            n += 1
        return modes

    def _sort_solutions(self, modes: dict) -> dict:
        sorted_idx = np.flip(np.argsort(modes['beta']))
        modes['beta'] = np.array(modes['beta'])[sorted_idx]
        modes['n_eff'] = modes['beta'] * self.wavelength / (2 * np.pi)
        modes['u'] = np.array(modes['u'])[sorted_idx]
        modes['w'] = np.array(modes['w'])[sorted_idx]
        modes['n'] = np.array(modes['n'])[sorted_idx]
        modes['m'] = np.array(modes['m'])[sorted_idx]
        return modes

    def _compute_neff_hnm_array(self, modes: dict) -> np.array:
        storage = np.zeros(shape=(modes['number'], 4), dtype='float') # Columns with n_eff, h, n, m
        for i in range(modes['number']):
            n, m = modes['n'][i], modes['m'][i]
            h = self._h_vs_nm(n, m)
            n_eff = modes['n_eff'][i]
            storage[i, :] = np.array([n_eff, h, n, m])
        return storage

    def _compute_prop_constants_array(self, modes: dict) -> np.array:
        storage = np.zeros(shape=(modes['number'], 6), dtype='float') # Columns with beta, n_eff, n, m, u, w
        for i in range(modes['number']):
            beta, n_eff = modes['beta'][i], modes['n_eff'][i]
            n, m = modes['n'][i], modes['m'][i]
            u, w = modes['u'][i], modes['w'][i]
            storage[i, :] = np.array([beta, n_eff, n, m, u, w])
        return storage

    def _h_vs_nm(self, n, m):
        return 2 * n + m - 1

    @property
    def _N_modes_theo(self):
        return np.floor(np.square(self._V) / 8).astype(int)

    @property
    def _N_modes(self):
        return np.floor(np.square(self._V) / 8).astype(int)


if __name__ == "__main__":
    fiber = GrinFiber()
    matrix = fiber.modes_coupling_matrix(complex=True, degen=True)
    GrinFiber.plot_coupling_matrix(matrix, complex=True)
    plt.show()
    