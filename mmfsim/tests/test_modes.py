import unittest
import numpy as np

from mmfsim.grid import Grid
from mmfsim.fiber import GrinFiber, StepIndexFiber
from mmfsim.modes import GrinLPMode, StepIndexLPMode


class TestGrinLPMode(unittest.TestCase):
    ener_trsh = 1e-8
    orth_trsh = 1e-16

    def test_degenerate_modes_energies(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)

        diffs_sum = 0
        for i in range(15):
            n, m = fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(fiber, grid)
            diffs_sum += np.abs(mode.energies[0] - mode.energies[1])
        self.assertAlmostEqual(diffs_sum, 0)

    def test_modes_energies(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)

        modes_sum = 0
        for i in range(15):
            n, m = fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(fiber, grid)
            modes_sum += np.sum(mode.energies)
        modes_sum /= 2 * i
        self.assertAlmostEqual(modes_sum, 1, delta=0.1)

    def test_degenerate_modes_orthogonality(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)

        modes_sum = 0
        for i in range(15):
            n, m = fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(fiber, grid)
            if n != 0: # Non centro-symmetric mode
                modes_sum += self.power_overlap_integral(mode._fields[:,:,0], mode._fields[:,:,1])
        self.assertAlmostEqual(modes_sum, 0)

    def test_modes_orthogonality(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = GrinFiber(radius=26e-6)

        modes_sum = 0
        for i in range(15):
            mode_i = GrinLPMode(fiber._neff_hnm[i, 2], fiber._neff_hnm[i, 3])
            mode_i.compute(fiber, grid)
            for j in range(15):
                if i != j:
                    mode_j = GrinLPMode(fiber._neff_hnm[j, 2], fiber._neff_hnm[j, 3])
                    mode_j.compute(fiber, grid)
                    modes_sum += self.power_overlap_integral(mode_i._fields, mode_j._fields)
        self.assertAlmostEqual(modes_sum, 0)
        
    @staticmethod
    def power_overlap_integral(field, mode):
        return np.square(np.abs(np.sum(field * np.conj(mode)))) / (np.sum(np.square(np.abs(field))) * np.sum(np.square(np.abs(mode))))


class TestStepIndexLPMode(unittest.TestCase):
    ener_trsh = 1e-8
    orth_trsh = 1e-16

    def test_degenerate_modes_energies(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=10e-6)

        diffs_sum = 0
        for i in range(15):
            n, m = fiber._prop_constants[i, 2], fiber._prop_constants[i, 3]
            mode = StepIndexLPMode(n, m)
            mode.compute(fiber, grid)
            diffs_sum += np.abs(mode.energies[0] - mode.energies[1])
        self.assertAlmostEqual(diffs_sum, 0)

    def test_modes_energies(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=10e-6)

        modes_sum = 0
        for i in range(15):
            n, m = fiber._prop_constants[i, 2], fiber._prop_constants[i, 3]
            mode = StepIndexLPMode(n, m)
            mode.compute(fiber, grid)
            modes_sum += np.sum(mode.energies)
        modes_sum /= 2 * i
        self.assertAlmostEqual(modes_sum, 1, delta=0.1)

    def test_degenerate_modes_orthogonality(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=10e-6)

        modes_sum = 0
        for i in range(15):
            n, m = fiber._prop_constants[i, 2], fiber._prop_constants[i, 3]
            mode = StepIndexLPMode(n, m)
            mode.compute(fiber, grid)
            if n != 0: # Non centro-symmetric mode
                modes_sum += self.power_overlap_integral(mode._fields[:,:,0], mode._fields[:,:,1])
        self.assertAlmostEqual(modes_sum, 0)

    def test_modes_orthogonality(self):
        grid = Grid(pixel_numbers=(128,128), pixel_size=0.5e-6)
        fiber = StepIndexFiber(radius=10e-6)

        modes_sum = 0
        for i in range(15):
            mode_i = StepIndexLPMode(fiber._prop_constants[i, 2], fiber._prop_constants[i, 3])
            mode_i.compute(fiber, grid)
            for j in range(15):
                if i != j:
                    mode_j = StepIndexLPMode(fiber._prop_constants[j, 2], fiber._prop_constants[j, 3])
                    mode_j.compute(fiber, grid)
                    modes_sum += self.power_overlap_integral(mode_i._fields, mode_j._fields)
        self.assertAlmostEqual(modes_sum, 0)
        
    @staticmethod
    def power_overlap_integral(field, mode):
        return np.square(np.abs(np.sum(field * np.conj(mode)))) / (np.sum(np.square(np.abs(field))) * np.sum(np.square(np.abs(mode))))
    


if __name__ == "__main__":
    unittest.main()
