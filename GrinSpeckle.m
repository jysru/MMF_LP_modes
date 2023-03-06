classdef GrinSpeckle < handle
    % Speckle from GRIN fiber
    
    properties (SetAccess = private, GetAccess = public)
        N_modes
        modes_coeffs
        polar_coeffs
        fiber
        grid
        field
        noise
    end

    properties (Dependent)
        intensity
    end
    
    methods
        function obj = GrinSpeckle(fiber, grid, opts)
            arguments
                fiber (1,1) GrinFiber
                grid (1,1) CameraGrid
                opts.N_modes (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.noise (1,1) double {mustBeNonnegative} = 0
            end

            if opts.N_modes > fiber.N_modes
                obj.N_modes = fiber.N_modes;
            else
                obj.N_modes = opts.N_modes;
            end
            obj.fiber = fiber;
            obj.grid = grid;
            obj.noise = opts.noise;

            obj.compute()
        end

 
        function modes_random_coeffs(obj)
            % Generate vector that sums up to one (intensity coefficients)
            Ip = rand(obj.N_modes, 1);
            Ip = Ip./sum(Ip, 1);
        
            % Generate random phases
            Phip = -pi + 2*pi*rand(obj.N_modes, 1);
        
            % Get the complex coefficients
            obj.modes_coeffs = sqrt(Ip) .* exp(1i * Phip);
            obj.polar_coeffs = rand(obj.N_modes, 1);
        end


        function compute(obj)
            champ0 = zeros([obj.grid.pixel_numbers, obj.N_modes]);
            champ90 = zeros([obj.grid.pixel_numbers, obj.N_modes]);
            obj.modes_random_coeffs();

            for i=1:obj.N_modes
                n = obj.fiber.neff_hnm(i, 3);
                m = obj.fiber.neff_hnm(i, 4);

                mode = GrinLPMode(n, m);
                mode.compute(obj.fiber, obj.grid);

                champ0(:,:,i) = mode.fields(:,:,1);
                champ90(:,:,i) = mode.fields(:,:,2);
            end

            Cp = reshape(obj.modes_coeffs, [1, 1, length(obj.modes_coeffs)]);
            Cpol = reshape(obj.polar_coeffs, [1, 1, length(obj.polar_coeffs)]);

            obj.field = sum(champ0.*Cp.*Cpol + champ90.*Cp.*(1-Cpol), 3);
            obj.field = obj.field / max(abs(obj.field), [], 'all');
        end


        function val = get.intensity(obj)
            val = abs(obj.field).^2;
            val = val / max(val, [], 'all');
            val = abs(val + obj.noise*randn(size(val)));
        end

        
        function hfig = plot(obj, opts)
            arguments
                obj (1,1) GrinSpeckle
                opts.fig_num (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.colormap char = 'hot'
            end

            hfig = figure(opts.fig_num); clf
                imagesc(obj.grid.x*1e6, obj.grid.y*1e6, obj.intensity)
                axis square
                colorbar
                colormap(opts.colormap)
                xlabel('x [um]')
                ylabel('y [um]')
                title(['GRIN fiber speckle (' num2str(obj.N_modes) ' modes)'])
        end

    end
end

