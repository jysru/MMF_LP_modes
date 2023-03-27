import numpy as np


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


