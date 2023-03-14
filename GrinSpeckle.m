classdef GrinSpeckle < handle
    % Speckle from GRIN fiber
    
    properties (SetAccess = private, GetAccess = public)
        N_modes
        modes_coeffs
        orient_coeffs
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
            Phip = 2*pi*rand(obj.N_modes, 1);
        
            % Get the complex coefficients
            obj.modes_coeffs = sqrt(Ip) .* exp(1i * Phip);
            obj.orient_coeffs = rand(obj.N_modes, 1);
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
            
            field = 0;
            for i=1:obj.N_modes
                n = obj.fiber.neff_hnm(i, 3);                
                Cp = obj.modes_coeffs(i);
                Cor = obj.orient_coeffs(i);
                
                if n == 0 % Centro-symmetric mode
                    field = field + champ0(:,:,i) * Cp;
                else % Non centro-symmetric mode -> Split power randomly on degenerates
                    r = rand();
                    Cp1 = sqrt(2) * Cp * sqrt(r) * exp(1i * 2*pi* rand());
                    Cp2 = sqrt(2) * Cp * sqrt(1-r) * exp(1i * 2*pi* rand());
                    field = field + Cp1 * (champ0(:,:,i) * sqrt(Cor) + champ90(:,:,i) * sqrt(1-Cor));
                    field = field + Cp2 * (champ0(:,:,i) * sqrt(r) + champ90(:,:,i) * sqrt(1-r));
                end
            end
            obj.field = field / max(abs(field), [], 'all');
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
            r = obj.fiber.radius*1e6;

            hfig = figure(opts.fig_num); clf
                imagesc(obj.grid.x*1e6, obj.grid.y*1e6, obj.intensity)
                rectangle('Position',[-r, -r, 2*r, 2*r], 'EdgeColor','w', 'Curvature',1, 'LineWidth',1.5);
                axis square
                colorbar
                colormap(opts.colormap)
                xlabel('x [um]')
                ylabel('y [um]')
                title(['GRIN fiber speckle (' num2str(obj.N_modes) ' modes)'])
        end

    end
end

