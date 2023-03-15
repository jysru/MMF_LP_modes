clear all


%% Define grid and GRIN fiber
grid = CameraGrid(pixel_numbers=[256, 256], pixel_size=0.25e-6, offsets=[0,0]*1e-6);
fiber = GrinFiber();


%% Check LP mode polarizations on grid and fiber
mode = GrinLPMode(3,1);
mode.compute(fiber, grid);
mode.plot(fig_num=1);


%% Generate speckle from first modes with some noise
speckle = GrinSpeckle(fiber, grid, noise=0.00, N_modes=25);
speckle.plot(fig_num=2);


%% Modal decomposition
modals = LPModalDecomposition(speckle, N_modes=25);
disp([abs(modals.modes_coeffs), abs(speckle.modes_coeffs)])

%% Generate dataset of individual LP modes
% dset = GrinLPDataset(fiber, grid, N_modes=opts.N_modes, noise=opts.noise);


%% Generate dataset of LP modes sums
% dset = GrinLPSpeckleDataset(fiber, grid, "N_modes",10, "length",10, "noise",0.01);