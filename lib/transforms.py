import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from beams import GaussianBeam


def fourier_transform(field, pad: float = 2):
    if pad is not None:
        init_shape = field.shape
        field = np.pad(field, pad_width=[int(init_shape[0] * pad/2), int(init_shape[1] * pad/2)], mode='constant')
    
    ft = np.fft.fftshift(np.fft.fft2(field))
    ft = ft / np.sqrt(ft.size)
    return ft


def fresnel_transform(field, grid: Grid, delta_z: float, wavelength: float = 1064e-9):
    dNxy = 1/grid.extents # Size of a sample in the Fourier plane [1/m]
    limNxy = (grid.pixel_numbers / 2) * dNxy ; # Interval boundaries [1/m]
    kx = 2 * np.pi * np.arange(start= -limNxy[0], stop=limNxy[0], step=dNxy[0]) # Angular frequencies vector (x-axis) [rad/m]
    ky = 2 * np.pi * np.arange(start= -limNxy[1], stop=limNxy[1], step=dNxy[1]) # Angular frequencies vector (y-axis) [rad/m]
    KX, KY = np.meshgrid(kx, ky)

    ft = np.fft.fftshift(np.fft.fft2(field))
    ft = ft * np.exp(1j * delta_z * np.sqrt(4 * np.square(np.pi / wavelength) - np.square(KX) - np.square(KY) ))
    ift = np.fft.ifft2(np.fft.ifftshift(ft))
    return ift


def unitary_fourier_transform(field, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = np.pad(field, pad_width=[int(init_shape[0] * pad/4), int(init_shape[1] * pad/4)], mode='constant')
    
    ft = np.fft.fftshift(np.fft.fft2(field))
    ift = np.fft.ifft2(np.fft.ifftshift(ft))

    if pad is not None:
        row_diff = ift.shape[0] - init_shape[0]
        col_diff = ift.shape[1] - init_shape[1]
        ift = ift[row_diff//2:-row_diff//2, col_diff//2:-col_diff//2]
    return ift



if __name__ == "__main__":
    grid = Grid(pixel_size=2e-6)
    beam = GaussianBeam(grid)
    beam.compute(amplitude=1, width=20e-6, centers=[0,0])
    print(beam)

    field = beam.field
    fres = fresnel_transform(field, grid, delta_z=1000e-6)
    four = fourier_transform(field, pad=1)

    print("Energies:\n"
          f" - Initial: {np.sum(np.abs(field)**2)}\n",
          f" - Fresnel: {np.sum(np.abs(fres)**2)}\n",
          f" - Fourier: {np.sum(np.abs(four)**2)}\n",
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    pl0 = axs[0].imshow(np.abs(field))
    axs[0].set_title("Field")
    plt.colorbar(pl0, ax=axs[0])

    pl1 = axs[1].imshow(np.abs(fres))
    axs[1].set_title("Fresnel")
    plt.colorbar(pl1, ax=axs[1])

    pl2 = axs[2].imshow(np.abs(four))
    axs[2].set_title("Fourier")
    plt.colorbar(pl2, ax=axs[2])




    
    plt.show()