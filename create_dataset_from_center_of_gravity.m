dset_file = 'C:\Users\saucourt\Datasets\data_base_02_03_0deg.mat';
[x, y, s, em] = getDatasetCentroid(dset_file);


%%
cam_px = 5.04e-6;
cam_size = 128;

xp = (round(x) - cam_size/2)*5.04e-6;
yp = (round(y) - cam_size/2)*5.04e-6;

grid = CameraGrid(pixel_numbers=[128, 128], pixel_size=5.04e-6/12.62, offsets=[xp, yp]/12.62);
fiber = GrinFiber();

dset = GrinLPDataset(fiber, grid, N_modes=5000, noise=0.01);
intens = dset.intensity;

%%
img = dset.intensity(:,:,22);
en = sum(img, "all");
img = img/en*em;

figure(2)
imagesc(img)
colorbar

figure(3)
imagesc(s)
axis square