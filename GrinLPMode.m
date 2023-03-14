classdef GrinLPMode < handle
    % LPnm mode from a GRIN fiber
    
    properties (GetAccess = public, SetAccess = public)
        n
        m
        theta0
        fields
    end


    properties (GetAccess = public, SetAccess = private)
        fn
        fm
    end


    properties (Dependent)
        intensities
        energies
    end
    

    methods
        function obj = GrinLPMode(n, m, opts)
            arguments
                n (1,1) double {mustBeNonnegative, mustBeInteger}
                m (1,1) double {mustBeNonnegative, mustBeInteger}
                opts.theta0 (1,1) double {mustBeNumeric} = 0
            end

            obj.n = n;
            obj.m = m;

            obj.fn = m-1;
            obj.fm = n;
            obj.theta0 = opts.theta0*pi/180;
        end


        function compute(obj, fiber, grid)
            arguments
                obj (1,1) GrinLPMode
                fiber (1,1) GrinFiber
                grid (1,1) CameraGrid
            end

            fac_n = factorial(obj.fn);
            fac_m_plus_n = factorial(obj.fm + obj.fn);

            if obj.fm==0, delta0m = 1; else, delta0m = 0; end
            epsilon_mn = pi*fiber.radius^2*fac_m_plus_n*(1+delta0m)/(2*fiber.V*fac_n);
            ro = grid.R / fiber.radius * sqrt(fiber.V);
            
            Lnm = 0;
            for s=0:obj.fn
                num = fac_m_plus_n * (-1)^s * ro.^(2*s);
                denom = factorial(obj.fm + s) * factorial(obj.fn - s) * factorial(s);
                Lnm = Lnm + num / denom;
            end
            
            fac1 = 1 ./ sqrt(epsilon_mn);
            fac2 = ro.^obj.fm;
            fac3 = exp(-(ro.^2)/2);

            obj.fields = nan([length(grid.x) length(grid.y) 2]);
            obj.fields(:,:,1) = fac1.*fac2.*fac3.*Lnm.*cos(obj.fm*grid.A + obj.theta0);
            obj.fields(:,:,2) = fac1.*fac2.*fac3.*Lnm.*cos(obj.fm*grid.A + obj.theta0 + pi/2);

            obj.fields(:,:,1) = obj.fields(:,:,1);
            obj.fields(:,:,2) = obj.fields(:,:,2) / max(abs(obj.fields(:,:,2)), [], 'all') * max(abs(obj.fields(:,:,1)), [], 'all');
        end
        

        function val = get.intensities(obj)
            val = abs(obj.fields).^2;
        end


        function val = get.energies(obj)
            val = squeeze(sum(obj.intensities, [1, 2]));
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
                xlabel('x [um]')
                ylabel('y [um]')
            subplot(1,2,2)
                imagesc(squeeze(obj.fields(:,:,2)))
                axis square
                colorbar
                title(['LP_' num2str(obj.n) '_,_' num2str(obj.m) ' (90 deg)'])
                xlabel('x [um]')
                ylabel('y [um]')
        end
        
    end
end

