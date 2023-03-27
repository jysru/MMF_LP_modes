classdef GrinLPSpeckleDataset < handle
    % Pure modes from GRIN fiber
    
    properties (SetAccess = private, GetAccess = public)
        N_modes
        fiber
        grid
        field
        noise
        length
    end

    properties (Dependent)
        intensity
    end
    
    methods
        function obj = GrinLPSpeckleDataset(fiber, grid, opts)
            arguments
                fiber (1,1) GrinFiber
                grid (1,1) CameraGrid
                opts.N_modes (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.noise (1,1) double {mustBeNonnegative} = 0
                opts.length (1,1) double {mustBeInteger, mustBePositive} = 32
            end

            if opts.N_modes > fiber.N_modes
                obj.N_modes = fiber.N_modes;
            else
                obj.N_modes = opts.N_modes;
            end
            obj.fiber = fiber;
            obj.grid = grid;
            obj.noise = opts.noise;
            obj.length = opts.length;

            obj.compute()
        end


        function compute(obj)
            field = zeros([obj.grid.pixel_numbers, obj.length]);

            for i=1:obj.length
                speckle = GrinSpeckle(obj.fiber, obj.grid, noise=obj.noise, N_modes=obj.N_modes);
                field(:,:,i) = speckle.field;
            end

            obj.field = field;
        end


        function val = get.intensity(obj)
            val = abs(obj.field).^2;
            val = abs(val + obj.noise*randn(size(val)));
        end

    end
end
