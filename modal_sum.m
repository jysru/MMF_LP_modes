function [intens, X, Y] = modal_sum(consts, opts)
    arguments
        consts (:,4) double

        opts.n_modes (1,1) double {mustBeNonnegative} = 0
        opts.noise (1,1) double {mustBeNonnegative} = 0
    end
    
    max_fiber_modes = size(consts,1);
    
    if opts.n_modes==0
        max_modes = max_fiber_modes;
    else
        if opts.n_modes > max_fiber_modes
            max_modes = max_fiber_modes;
        else
            max_modes = opts.n_modes;
        end
    end

    Cp = modes_random_coeffs(max_modes, 1);
    polratio = rand(max_modes, 1);

    for i=1:max_modes
        m = consts(i, 4);
        n = consts(i, 3);
        
        c0 = modes_grin(m-1, n, grid_size=50e-6, grid_step=0.1e-6, offsets=[0, 0]*1e-6);
        [c90, X, Y] = modes_grin(m-1, n, grid_size=50e-6, grid_step=0.1e-6, offsets=[0, 0]*1e-6, theta0=90);
        c0 = c0/max(c0, [], 'all');
        c90 = c90/max(c90, [], 'all');
    
        if i==0
            champ0 = zeros([size(c0), max_modes]);
            champ90 = zeros([size(c90), max_modes]);
        end
        champ0(:,:,i) = c0;
        champ90(:,:,i) = c90;

    end

    Cp_reshaped = reshape(Cp, [1, 1, length(Cp)]);
    pol_reshaped = reshape(polratio, [1, 1, length(polratio)]);

    schamp = sum(champ0.*Cp_reshaped.*pol_reshaped + champ90.*Cp_reshaped.*(1-pol_reshaped), 3);
    intens = abs(schamp).^2;
    intens = intens / max(intens, [], 'all');
    intens = abs(intens + opts.noise*randn(size(intens)));
end

