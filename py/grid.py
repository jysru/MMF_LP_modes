import numpy as np


class CameraGrid():

    def __init__(
            self,
            pixel_size: float = 5.04e-6,
            pixel_numbers: tuple[int, int] = (128, 128),
            offsets: tuple[float, float] = (0.0, 0.0),
            ) -> None:
        
        self.pixel_size = pixel_size
        self.pixel_numbers = np.asarray(pixel_numbers)
        self.offsets = np.asarray(offsets)
        self.x = np.arange(start=-self.grid_sizes[0]/2, stop=self.grid_sizes[0]/2 - self.pixel_size, step=self.pixel_size) - self.offsets[0]
        self.y = np.arange(start=-self.grid_sizes[1]/2, stop=self.grid_sizes[1]/2 - self.pixel_size, step=self.pixel_size) - self.offsets[1]
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @property
    def grid_sizes(self):
        return self.pixel_size * self.pixel_numbers

    @property
    def R(self):
        return np.sqrt(np.square(self.X) + np.square(self.Y))

    @property
    def A(self):
        return np.arctan2(self.Y, self.X)


if __name__ == "__main__":
    grid = CameraGrid()