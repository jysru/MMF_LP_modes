import unittest
import numpy as np

from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber, StepIndexFiber
from mmfsim.modes import GrinLPMode, StepIndexLPMode
from mmfsim.beams import GaussianBeam
from mmfsim.coupling import GrinFiberCoupler, StepIndexFiberCoupler


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

    def test_couple_gaussian_flat_phases(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)
        i_max = 55

        beam = GaussianBeam(grid)
        beam.compute(amplitude=1, width=8000e-6, centers=[0,0])
        beam.normalize_by_energy()

        coupling = GrinFiberCoupler(beam.field, grid, fiber, N_modes=i_max)
        coupling_ratio = np.sum(np.square(np.abs(coupling.field)))
        n = fiber._neff_hnm[:coupling.N_modes, 2]
        sum_centro = np.sum(coupling.coeffs_intensity[n == 0])
        sum_not_centro = np.sum(coupling.coeffs_intensity[n != 0])

        with self.subTest():
            self.assertGreaterEqual(1, coupling_ratio)
        with self.subTest():
            self.assertAlmostEqual(coupling.total_coeffs_intensity, sum_centro)
        with self.subTest():
            self.assertAlmostEqual(0, sum_not_centro)


class TestCouplingStepIndexLPMode(unittest.TestCase):
    coup_trsh = 1e-8

    def test_couple_pure_modes(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=26e-6)

        traces_sum = 0
        modes_sum = 0
        i_max = 55
        for i in range(i_max):
            n, m = fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3]
            mode = StepIndexLPMode(n, m)
            mode.compute(fiber, grid)
            coupling = StepIndexFiberCoupler(mode._fields[:,:,0], grid, fiber, N_modes=i_max)

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

    def test_couple_gaussian_flat_phases(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=26e-6)
        i_max = 55

        beam = GaussianBeam(grid)
        beam.compute(amplitude=1, width=8000e-6, centers=[0,0])
        beam.normalize_by_energy()

        coupling = StepIndexFiberCoupler(beam.field, grid, fiber, N_modes=i_max)
        coupling_ratio = np.sum(np.square(np.abs(coupling.field)))
        n = fiber._neff_hnm[:coupling.N_modes, 2]
        sum_centro = np.sum(coupling.coeffs_intensity[n == 0])
        sum_not_centro = np.sum(coupling.coeffs_intensity[n != 0])

        with self.subTest():
            self.assertGreaterEqual(1, coupling_ratio)
        with self.subTest():
            self.assertAlmostEqual(coupling.total_coeffs_intensity, sum_centro)
        with self.subTest():
            self.assertAlmostEqual(0, sum_not_centro)


if __name__ == "__main__":
    unittest.main()
