classdef GrinFiber < handle
    % Defines GRIN fiber characteristics for a given wavelength

    properties (SetAccess = public, GetAccess = public)
        wavelength
        n1
        n2
        radius
    end

    properties (Dependent)
        NA
        V
        N_modes
    end

    methods
        function obj = GrinFiber(opts)
            arguments
                opts.wavelength (1,1) double {mustBePositive, mustBeNonzero} = 1064e-9
                opts.n1 (1,1) double {mustBePositive, mustBeNonzero} = 1.465
                opts.n2 (1,1) double {mustBePositive, mustBeNonzero} = 1.45
                opts.radius (1,1) double {mustBePositive, mustBeNonzero} = 26e-6
            end

            obj.wavelength = opts.wavelength;
            obj.n1 = opts.n1;
            obj.n2 = opts.n2;
            obj.radius = opts.radius;
        end

        function val = get.NA(obj)
            val = sqrt(obj.n1^2 - obj.n2^2);
        end

        function val = get.V(obj)
            val = 2*pi*obj.radius*obj.NA / obj.wavelength;
        end

        function val = get.N_modes(obj)
            val = floor(obj.V^2/4);
        end
    end
end