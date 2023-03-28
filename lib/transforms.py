import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from beams import GaussianBeam


def fourier_transform(field, pad: float = 2):
    if pad is not None:
        init_shape = field.shape
        field = np.pad(field, pad_width=[int(init_shape[0] * pad/2), int(init_shape[1] * pad/2)], mode='constant')
    
    ft = np.fft.fftshift(np.fft.fft2(field))
    return ft


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
    beam.compute( amplitude=1, width=10e-6, centers=[0,0])
    print(beam)

    beam.plot(complex=True)

    field = fourier_transform(beam.field, pad=1)

    plt.figure()
    plt.imshow(np.abs(field))
    plt.colorbar()
    plt.show()