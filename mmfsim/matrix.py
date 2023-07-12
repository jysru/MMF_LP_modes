import numpy as np
from scipy import linalg


def matrix_to_vector(matrix: np.ndarray, order: str = 'F') -> np.array:
        return np.reshape(matrix, newshape=(-1,), order=order)


def vector_to_matrix(vector, shape: tuple[int, int], order: str = 'F') -> np.array:
    return np.reshape(vector, newshape=shape, order=order)


def crop_center(img, crop):
    y, x = img.shape
    startx = x//2 - crop//2
    starty = y//2 - crop//2
    return img[starty:starty+crop, startx:startx+crop]


def vec32_to_vec36(vec):
    vec = np.ravel(vec)
    phases = np.angle(vec)
    return np.ravel(np.r_[
        np.r_[[0], phases[:4], [0]][None], 
        phases[4:-4].reshape(4, 6), 
        np.r_[[0], phases[-4:], [0]][None],
    ])[None]


def inverse_repeat(matrix, repeats, axis):
    if isinstance(repeats, int):
        indices = np.arange(matrix.shape[axis] / repeats, dtype=int) * repeats
    else:  # assume array_like of int
        indices = np.cumsum(repeats) - 1
    return matrix.take(indices, axis)


def matrix_to_partition(matrix, partition_size: tuple[int, int]):
    matrix = inverse_repeat(matrix, repeats=partition_size[0], axis=0)
    return inverse_repeat(matrix, repeats=partition_size[1], axis=1)


def partition_to_matrix(partition: np.ndarray, matrix: np.ndarray):
        repeater_axis0 = int(np.ceil(matrix.shape[0] / partition.shape[0]))
        repeater_axis1 = int(np.ceil(matrix.shape[1] / partition.shape[1]))
        matrix = np.repeat(partition, repeater_axis0, axis=0)
        matrix = np.repeat(matrix, repeater_axis1, axis=1)
        return matrix


def square_random(n, complex: bool = False):
    matrix = np.random.rand(n,n)

    if complex:
        phi = 2 * np.pi * np.random.rand(n, n)
        if n > 1:
            triu_nodiag = np.triu(phi, k=1)
            triu = np.triu(phi)
            phi = np.angle(np.exp(1j * (triu - triu_nodiag.T)))
        matrix = matrix * np.exp(1j * phi)

    return matrix


def square_random_toeplitz(n, norm_intens: bool = True, complex: bool = False, decay_width: float = None):
    vec = np.random.rand(n)

    if decay_width is not None:
        vec *= np.exp( -np.square(np.arange(0, n)) / (2 * np.square(decay_width)) )

    if norm_intens:
        vec = np.sqrt(vec / np.sum(vec))
    matrix = linalg.toeplitz(vec, vec)

    if complex:
        phi = 2 * np.pi * np.random.rand(n, n)
        if n > 1:
            triu_nodiag = np.triu(phi, k=1)
            triu = np.triu(phi)
            phi = np.angle(np.exp(1j * (triu - triu_nodiag.T)))
        matrix = matrix * np.exp(1j * phi)

    return matrix


def banana_random_toeplitz(n, norm_intens: bool = True, complex: bool = False, narrow_width: float = 1, wide_width: float = None):
    matrix = np.zeros((n,n), dtype=np.complex128)
    vec = np.random.rand(n)

    if norm_intens:
        vec = np.sqrt(vec / np.sum(vec))

    if wide_width is None:
        wide_width = n

    widths = np.linspace(narrow_width, wide_width, int(np.ceil(n/2)))
    if np.mod(n, 1):
        widths = np.concatenate([widths, np.flip(widths[:-1])], axis=0)
    else:
        widths = np.concatenate([widths, np.flip(widths)], axis=0)


    for i in range(n):
        matrix[i, :] = np.roll(vec, i-1) * np.exp( -np.square(np.arange(0, n)) / (2 * np.square(widths[i])) )[:n]
    
    matrix = np.triu(matrix) + np.transpose(np.triu(matrix, k=1))
    

    if complex:
        phi = 2 * np.pi * np.random.rand(n, n)
        if n > 1:
            triu_nodiag = np.triu(phi, k=1)
            triu = np.triu(phi)
            phi = np.angle(np.exp(1j * (triu - triu_nodiag.T)))
        matrix = matrix * np.exp(1j * phi)

    return matrix