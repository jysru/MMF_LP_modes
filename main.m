clear all

consts = modes_grin_betas();
[intens, X, Y] = modal_sum(consts, n_modes=10, noise=0.02);

figure(1), clf
imagesc(X(1,:)*1e6, Y(:,1)*1e6, intens)
axis square
colorbar
colormap('hot')
xlabel('x [um]')
ylabel('y [um]')