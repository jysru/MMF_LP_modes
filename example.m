clear all


%% Define grid and GRIN fiber
grid = CameraGrid(pixel_numbers=[256, 256], pixel_size=0.25e-6, offsets=[0,0]*1e-6);
fiber = GrinFiber();


%% Check LP mode polarizations on grid and fiber
mode = GrinLPMode(2,2);
mode.compute(fiber, grid);
mode.plot(fig_num=1);


%% Generate speckle from first modes with some noise
speckle = GrinSpeckle(fiber, grid, noise=0.02, N_modes=25);
speckle.plot(fig_num=2)

