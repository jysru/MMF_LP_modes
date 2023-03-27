import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from fiber import GrinFiber
from modes import GrinLPMode
from speckle import GrinSpeckle
from devices import MockDeformableMirror
from plots import complex_image
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
    
    def plot(self, cmap: str = 'hot', complex: bool = False, complex_hsv: bool = False):
        r = self.fiber.radius * 1e6
        extent = np.array([np.min(self.grid.x), np.max(self.grid.x), np.min(self.grid.y), np.max(self.grid.y)]) * 1e6
        circle1 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')
        circle2 = plt.Circle((-self.grid.offsets[0], -self.grid.offsets[1]), r, fill=False, edgecolor='w', linestyle='--')

        if complex:
            if complex_hsv:
                fig = plt.figure()
                ax = plt.gca()
                pl = plt.imshow(complex_image(self.speckle), extent=extent)
                ax.add_patch(circle1)
                ax.set_xlabel("x [um]")
                ax.set_ylabel("x [um]")
                ax.set_title(f"GRIN fiber speckle ({self.N_modes} modes)")
                return (fig, ax, pl)
            else:
                fig, axs = plt.subplots(1, 2, figsize=(13,4))
                pl0 = axs[0].imshow(np.square(np.abs(self.speckle)), extent=extent, cmap=cmap)
                pl1 = axs[1].imshow(np.angle(self.speckle), extent=extent, cmap="twilight")
                axs[0].add_patch(circle1)
                axs[1].add_patch(circle2)
                axs[0].set_xlabel("x [um]")
                axs[1].set_xlabel("x [um]")
                axs[0].set_ylabel("y [um]")
                axs[1].set_ylabel("y [um]")
                axs[0].set_title(f"GRIN speckle intensity ({self.N_modes} modes)")
                axs[1].set_title(f"GRIN speckle phase ({self.N_modes} modes)")
                plt.colorbar(pl0, ax=axs[0])
                plt.colorbar(pl1, ax=axs[1])
                return (fig, axs, [pl0, pl1])
        else:
            fig = plt.figure()
            ax = plt.gca()
            pl = plt.imshow(np.square(np.abs(self.speckle)), cmap=cmap, extent=extent)
            ax.add_patch(circle1)
            ax.set_xlabel("x [um]")
            ax.set_ylabel("x [um]")
            ax.set_title(f"GRIN speckle intensity ({self.N_modes} modes)")
            plt.colorbar(pl, ax=ax)

    def __str__(self) -> str:
        return (
            f"\t - Coupled power: {np.sum(np.square(np.abs(self.modes_coeffs)))}\n"
            f"\t - Number of modes: {self.N_modes}"
        )


if __name__ == "__main__":

    phase_map = 2*np.pi*np.random.rand(6,6)
    dm = MockDeformableMirror(pixel_size=100e-6, pixel_numbers=(128,128))
    dm.apply_phase_map(phase_map)

    grid = Grid(pixel_size=dm.pixel_size, pixel_numbers=dm.pixel_numbers)
    beam = beams.GaussianBeam(grid)
    beam.compute(amplitude=1, width=3500e-6, centers=[0,0])
    dm.apply_amplitude_map(beam.amplitude)
    dm.reduce_by(200)
    dm.plot()

    beam.grid.reduce_by(200)
    beam.field = dm._field_matrix
    coupled = GrinFiberBeamCoupler(beam=beam, N_modes=25)
    coupled.plot(cmap='gray')
    print(coupled)
    plt.show()