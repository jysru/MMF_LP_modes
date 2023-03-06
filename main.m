clear all

grid = CameraGrid(pixel_numbers=[256, 256], pixel_size=0.25e-6, offsets=[0,0]*1e-6);
fiber = GrinFiber();
speckle = GrinSpeckle(fiber, grid, noise=0.02, N_modes=4);
speckle.plot()


dset = GrinLPDataset(fiber, grid);

