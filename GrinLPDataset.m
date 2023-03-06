classdef GrinLPDataset < handle
    % Speckle from GRIN fiber
    
    properties (SetAccess = private, GetAccess = public)
        N_modes
        fiber
        grid
        field
        noise
    end

    properties (Dependent)
        intensity
    end
    
    methods
        function obj = GrinLPDataset(fiber, grid, opts)
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


        function compute(obj)
            champ = zeros([obj.grid.pixel_numbers, 2*obj.N_modes]);

            for i=1:obj.N_modes
                k = 2*(i-1)+1;
                n = obj.fiber.neff_hnm(i, 3);
                m = obj.fiber.neff_hnm(i, 4);

                mode = GrinLPMode(n, m);
                mode.compute(obj.fiber, obj.grid);

                mode.fields(:,:,1) = mode.fields(:,:,1) / max(abs(mode.fields(:,:,1)), [], 'all');
                mode.fields(:,:,2) = mode.fields(:,:,2) / max(abs(mode.fields(:,:,2)), [], 'all');

                champ(:,:,2*(i-1)+1) = mode.fields(:,:,1);
                champ(:,:,2*(i-1)+2) = mode.fields(:,:,2);
                
            end

            obj.field = champ;
        end


        function val = get.intensity(obj)
            val = abs(obj.field).^2;
            val = abs(val + obj.noise*randn(size(val)));
        end

    end
end

