import numpy as np
import matplotlib.pyplot as plt

from mmfsim.grid import Grid
from mmfsim.beams import GaussianBeam


def crop_img(img, newsize):
    """ Crop image to new size, from the center."""
    diff_row = img.shape[0] - newsize[0]
    diff_col = img.shape[1] - newsize[1]
    crop_row, crop_col = diff_row // 2, diff_col // 2
    return img[crop_row:-crop_row, crop_col:-crop_col]


def _pad_checker(pad: float):
    if pad < 1:
        print('Coerced pad value to 1!')
        return 1
    else:
        return pad


def pad_img(img, pad: float = 1):
    """ Pad zeroes on each axis.
        Default pad value is 1, which means that the padded image is twice the original size on each axis.
    """
    return np.pad(img, pad_width=[int(img.shape[0] * pad / 2), int(img.shape[1] * pad / 2)], mode='constant')


def fourier_transform(field, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)
        
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    ft = ft / np.sqrt(ft.size)
    return crop_img(ft, init_shape) if pad is not None else ft


def fresnel_transform(field, grid: Grid, delta_z: float, wavelength: float = 1064e-9, pad: float = None):
    if pad is not None:
        init_shape = field.shape
        field = pad_img(field, pad)

    dNxy = 1/((pad+1) * grid.extents) # Size of a sample in the Fourier plane [1/m]
    limNxy = (field.shape[0] / 2) * dNxy ; # Interval boundaries [1/m]
    kx = 2 * np.pi * np.arange(start= -limNxy[0], stop=limNxy[0], step=dNxy[0]) # Angular frequencies vector (x-axis) [rad/m]
    ky = 2 * np.pi * np.arange(start= -limNxy[1], stop=limNxy[1], step=dNxy[1]) # Angular frequencies vector (y-axis) [rad/m]
    KX, KY = np.meshgrid(kx, ky)
    prop = np.exp(1j * delta_z * np.sqrt(np.abs(4 * np.square(np.pi / wavelength) - np.square(KX) - np.square(KY))))

    ft = np.fft.fftshift(np.fft.fft2(field))
    ft = ft * prop
    ift = np.fft.ifft2(np.fft.ifftshift(ft))
    return crop_img(ift, init_shape) if pad is not None else ift


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
    beam.compute(width=20e-6, centers=[0,0])
    beam.normalize_by_energy()

    field = beam.field
    fres = fresnel_transform(field, grid, delta_z=3000e-6)
    four = fourier_transform(field, pad=1)

    print("Energies:\n"
          f" - Initial: {np.sum(np.abs(field)**2)}\n",
          f" - Fresnel: {np.sum(np.abs(fres)**2)}\n",
          f" - Fourier: {np.sum(np.abs(four)**2)}\n",
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    pl0 = axs[0].imshow(np.square(np.abs(field)))
    axs[0].set_title("Initial intensity")
    plt.colorbar(pl0, ax=axs[0])

    pl1 = axs[1].imshow(np.square(np.abs(fres)))
    axs[1].set_title("Fresnel intensity")
    plt.colorbar(pl1, ax=axs[1])

    pl2 = axs[2].imshow(np.square(np.abs(four)))
    axs[2].set_title("Fourier intensity")
    plt.colorbar(pl2, ax=axs[2])

    plt.show()