import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from fiber import GrinFiber
from modes import GrinLPMode
from speckle import GrinSpeckle
from devices import DeformableMirror
import beams


class GrinFiberBeamCoupler(GrinSpeckle):

    def __init__(self, beam: beams.Beam, fiber: GrinFiber = GrinFiber(), N_modes: int = 10, noise_std: float = 0) -> None:
        super().__init__(fiber, beam.grid, N_modes, noise_std)
        self.beam = beam
        self.field = self.beam.field
        self.decompose(self.N_modes)

    def decompose(self, N_modes: int = 10):
        N_modes = self.fiber._N_modes if N_modes > self.fiber._N_modes else N_modes
        modes_coeffs = np.zeros(shape=(N_modes), dtype=np.complex64)
        orient_coeffs = np.zeros(shape=(N_modes))

        for i in range(N_modes):
            n, m = self.fiber._neff_hnm[i, 2], self.fiber._neff_hnm[i, 3]
            mode = GrinLPMode(n, m)
            mode.compute(self.fiber, self.grid)
            mode0, mode90 = mode._fields[:,:,0], mode._fields[:,:,1]

            if n == 0: # Centro-symmetric mode
                Cp = GrinSpeckle.power_overlap_integral(self.field, mode0)
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode0)
                modes_coeffs[i] = np.sqrt(Cp) * np.exp(1j * phi)
            else:
                Cp1 = GrinSpeckle.power_overlap_integral(self.field, mode0)
                Cp2 = GrinSpeckle.power_overlap_integral(self.field, mode90)
                Cor = Cp1 / (Cp1 + Cp2)
                mode_orient = np.sqrt(Cor) * mode0 +  np.sqrt(1 - Cor) * mode90
                phi = GrinSpeckle.phase_from_overlap_integral(self.field, mode_orient)
                modes_coeffs[i] = np.sqrt(Cp1 + Cp2) * np.exp(1j * phi)
                orient_coeffs[i] = Cor

        self.modes_coeffs = GrinSpeckle._normalize_coeffs(modes_coeffs)
        self.orient_coeffs = orient_coeffs
        
    @property
    def speckle(self):
        self.compose(coeffs=(self.modes_coeffs, self.orient_coeffs))
        return self.field
    
    def plot(self, cmap: str = 'hot'):
        r = self.fiber.radius * 1e6
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * 1e6
        circle = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')

        fig = plt.figure()
        ax = plt.gca()
        pl = plt.imshow(np.abs(self.speckle), cmap=cmap)
        ax.add_patch(circle)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("x [um]")
        ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
        plt.colorbar(pl, ax=ax)


if __name__ == "__main__":

    dm = DeformableMirror()
    dm.reduce_by(500)
    phase_map = 2*np.pi*np.random.rand(6,6)
    dm.apply_phase_map(phase_map)

    grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=(128,128))
    beam = beams.GaussianBeam(grid)
    beam.compute(amplitude=1, width=20e-6, centers=[0,0])
    dm.apply_amplitude_map(beam.amplitude)
    
    dm.plot()
    plt.show()

    newgrid = Grid(pixel_size=dm.pixel_size, pixel_numbers=(128,128))
    beam = beams.GaussianBeam(newgrid)
    beam.compute(amplitude=1, width=3e-6, centers=[0,0])
    beam = dm.export_to_beam(beam, keep_beam_phases=False)
    # beam.plot(complex=True)
    # plt.show()

    
    coupled = GrinFiberBeamCoupler(beam=beam, N_modes=30)
    coupled.plot()
    plt.show()