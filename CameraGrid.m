classdef CameraGrid < handle
    % CameraGrid defines grids to handle computations

    properties (SetAccess = private, GetAccess = public)
        pixel_size
        pixel_numbers
        offsets
        x
        y
        X
        Y
    end


    properties (Dependent)
        grid_sizes
        R
        A
    end


    methods
        function obj = CameraGrid(opts)
            arguments
                opts.pixel_size (1,1) double {mustBeNumeric, mustBePositive} = 5e-6
                opts.pixel_numbers (1,2) double {mustBeNumeric, mustBePositive} = [128, 128]
                opts.offsets (1,2) double {mustBeNumeric} = [0, 0]
            end

            obj.pixel_size = opts.pixel_size;
            obj.pixel_numbers = opts.pixel_numbers;
            obj.offsets = opts.offsets;
            
            obj.x = (-obj.grid_sizes(1)/2 : obj.pixel_size : obj.grid_sizes(1)/2-obj.pixel_size) - obj.offsets(1);
            obj.y = (-obj.grid_sizes(2)/2 : obj.pixel_size : obj.grid_sizes(2)/2-obj.pixel_size) - obj.offsets(2);

            [obj.X, obj.Y] = meshgrid(obj.x, obj.y);
        end


        function val = get.grid_sizes(obj)
            val = obj.pixel_size * obj.pixel_numbers;
        end


        function val = get.R(obj)
            val = sqrt( obj.X.^2 + obj.Y.^2);
        end


        function val = get.A(obj)
            val = atan2(obj.Y, obj.X);
        end
    end
end