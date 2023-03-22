import numpy as np


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
    
    def __str__(self):
        return f"""
        {self._neff_hnm[:10]}
        """


if __name__ == "__main__":
    fiber = GrinFiber()
    print(fiber)