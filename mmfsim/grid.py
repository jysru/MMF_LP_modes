import numpy as np

camera_pixel_size = 5.04e-6
deformable_mirror_pixel_size = 300e-6


class Grid():

    def __init__(
            self,
            pixel_size: float = camera_pixel_size,
            pixel_numbers: tuple[int, int] = (128, 128),
            offsets: tuple[float, float] = (0.0, 0.0),
            ) -> None:
        
        self.pixel_size = pixel_size
        self.pixel_numbers = np.asarray(pixel_numbers)
        self.offsets = np.asarray(offsets)
        self._generate_grids()

    def _generate_grids(self):
        self._generate_vectors()
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def _generate_vectors(self):
        self.x = np.arange(start=-self.grid_sizes[0]/2, stop=self.grid_sizes[0]/2, step=self.pixel_size) + self.pixel_size/2 - self.offsets[0]
        self.y = np.arange(start=-self.grid_sizes[1]/2, stop=self.grid_sizes[1]/2, step=self.pixel_size) + self.pixel_size/2 - self.offsets[1]
        
    def _add_offsets(self, offsets: tuple[float, float] = None) -> None:
        if offsets:
            self.offsets += np.asarray(offsets)
            self._generate_vectors()

    def _set_offsets(self, offsets: tuple[float, float] = None) -> None:
        self.offsets = np.asarray(offsets)
        self._generate_vectors()

    def magnify_by(self, coeff: float):
        if coeff == 0:
            coeff = 1
            print('Coeff cannot be zero, coerced to 1')

        self.pixel_size *= coeff
        self.x *= coeff
        self.y *= coeff
        self.offsets *= coeff
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def reduce_by(self, coeff: float):
        if coeff == 0:
            coeff = 1
            print('Coeff cannot be zero, coerced to 1')

        self.pixel_size /= coeff
        self.x /= coeff
        self.y /= coeff
        self.offsets /= coeff
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
    
    @property
    def boundaries(self):
        x_bounds = np.min(self.x), np.max(self.x)
        y_bounds = np.min(self.y), np.max(self.y)
        return (np.asarray(x_bounds), np.asarray(y_bounds))
    
    @property
    def extents(self):
        x_bounds, y_bounds = self.boundaries
        return np.abs(np.asarray([np.diff(x_bounds)[0], np.diff(y_bounds)[0]]))
        
    def __str__(self) -> str:
        return (
            f"{__class__.__name__} instance with:\n"
            f"  - Pixel size: {self.pixel_size}\n"
            f"  - Pixels number: {self.pixel_numbers}\n"
            f"  - Centers: {self.offsets}\n"
            f"  - Extent: {self.extents}\n"
            f"  - Boundaries: {self.boundaries}\n"
        )


if __name__ == "__main__":
    grid = Grid()
    print(grid)