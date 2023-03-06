classdef GrinFiber < handle
    % Defines GRIN fiber characteristics for a given wavelength

    properties (SetAccess = public, GetAccess = public)
        wavelength
        n1
        n2
        radius
    end

    
    properties (SetAccess = private, GetAccess = public)
        neff_hnm
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

            obj.compute_modes_constants()
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


        function compute_modes_constants(obj)        
            k0 = 2*pi/obj.wavelength;
            delta = obj.NA^2 / (2*obj.n1^2);
        
            storage = nan(obj.N_modes^2, 4, 'double'); % Columns with betah, h, n, m
            k = 1;
            for m=1:obj.N_modes+1
                for n=0:obj.N_modes
                    h = 2*n + m-1;
                    betah = k0*obj.n1 - h*sqrt(2*delta)/obj.radius;
                    nh = betah*obj.wavelength/(2*pi);
                    storage(k, :) = [nh, h, n, m];
                    k = k+1;
                end
            end
        
            sorted_constants = sortrows(storage, 'descend');
            if size(sorted_constants, 1) > obj.N_modes
                obj.neff_hnm = sorted_constants(1:obj.N_modes, :);
            else
                obj.neff_hnm = sorted_constants;
            end
        end

    end
end