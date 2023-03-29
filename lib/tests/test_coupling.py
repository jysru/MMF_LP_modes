import unittest
import numpy as np

from lib.grid import Grid
from lib.fiber import GrinFiber
from lib.modes import GrinLPMode
from lib.coupling import GrinFiberCoupler


class TestCouplingGrinLPMode(unittest.TestCase):
    coup_trsh = 1e-8

    def test_couple_pure_modes(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)

        traces_sum = 0
        modes_sum = 0
        i_max = 55
        for i in range(i_max):
            n, m = fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(fiber, grid)
            coupling = GrinFiberCoupler(mode._fields[:,:,0], grid, fiber, N_modes=i_max)

            modes_sum += np.square(np.abs(coupling.modes_coeffs[i]))
            mask = np.ones(shape=(i_max,), dtype=bool)
            mask[i] = 0
            traces_sum += np.sum(np.square(np.abs(coupling.modes_coeffs[mask])))
        modes_sum /= i_max
        traces_sum /= i_max

        with self.subTest():
            self.assertAlmostEqual(modes_sum, 1, delta=1e-4)
        with self.subTest():
            self.assertAlmostEqual(traces_sum, 0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
