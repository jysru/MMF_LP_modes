classdef GrinLPMode < handle
    % LPnm mode from a GRIN fiber
    
    properties (GetAccess = public, SetAccess = public)
        n
        m
        theta0
        fields
    end

    properties (Dependent)
        intensities
    end
    
    methods
        function obj = GrinLPMode(n, m, opts)
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                m (1,1) double {mustBeNonnegative, mustBeInteger}
                opts.theta0 (1,1) double {mustBeNumeric} = 0
            end

            obj.n = m-1;
            obj.m = n;
            obj.theta0 = opts.theta0*pi/180;
        end

        function compute(obj, fiber, grid)
            arguments
                obj (1,1) GrinLPMode
                fiber (1,1) GrinFiber
                grid (1,1) CameraGrid
            end

            fac_n = factorial(obj.n);
            fac_m_plus_n = factorial(obj.m + obj.n);

            if obj.m==0, delta0m = 1; else, delta0m = 0; end
            epsilon_mn = pi*fiber.radius^2*fac_m_plus_n*(1+delta0m)/(2*fiber.V*fac_n);
            ro = grid.R / fiber.radius * sqrt(fiber.V);
            
            Lnm = 0;
            for s=0:obj.n
                num = fac_m_plus_n * (-1)^s * ro.^(2*s);
                denom = factorial(obj.m + s) * factorial(obj.n - s) * factorial(s);
                Lnm = Lnm + num / denom;
            end
            
            fac1 = 1 ./ sqrt(epsilon_mn);
            fac2 = ro.^obj.m;
            fac3 = exp(-(ro.^2)/2);

            obj.fields = nan([length(grid.x) length(grid.y) 2]);
            obj.fields(:,:,1) = fac1.*fac2.*fac3.*Lnm.*cos(obj.m*grid.A + obj.theta0);
            obj.fields(:,:,2) = fac1.*fac2.*fac3.*Lnm.*cos(obj.m*grid.A + obj.theta0 + pi/2);
        end
        
        function val = get.intensities(obj)
            val = abs(obj.fields).^2;
        end

        function hfig = plot(obj, opts)
            arguments
                obj (1,1) GrinLPMode
                opts.fig_num (1,1) double {mustBePositive, mustBeInteger} = 1
            end

            hfig = figure(opts.fig_num); clf
            subplot(1,2,1)
                imagesc(squeeze(obj.fields(:,:,1)))
                axis square
                colorbar
                title(['LP_' num2str(obj.n) '_,_' num2str(obj.m) ' (0 deg)'])
            subplot(1,2,2)
                imagesc(squeeze(obj.fields(:,:,2)))
                axis square
                colorbar
                title(['LP_' num2str(obj.n) '_,_' num2str(obj.m) ' (90 deg)'])

        end
    end
end

