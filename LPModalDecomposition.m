classdef LPModalDecomposition < handle
    %LPMODALDECOMPOSITION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private, GetAccess = public)
        N_modes
        modes_coeffs
        speckle
    end
    
    methods
        function obj = LPModalDecomposition(speckle, opts)
            arguments
                speckle (1,1) GrinSpeckle
                opts.N_modes (1,1) double {mustBeInteger, mustBePositive} = 10
            end

            obj.speckle = speckle;
            obj.N_modes = opts.N_modes;
            obj.compute();
        end
        
        function compute(obj)
            obj.modes_coeffs = nan(obj.N_modes, 2);

            for i=1:obj.N_modes
                n = obj.speckle.fiber.neff_hnm(i, 3);
                m = obj.speckle.fiber.neff_hnm(i, 4);
                
                mode = GrinLPMode(n, m);
                mode.compute(obj.speckle.fiber, obj.speckle.grid);
                mode0_field = squeeze(mode.fields(:,:,1));
                mode90_field = squeeze(mode.fields(:,:,2));

                if n == 0 % Centro-symmetric mode
                    Cp = LPModalDecomposition.overlap_integral(obj.speckle.field, mode0_field);
                    obj.modes_coeffs(i,1) = Cp;
                else % Non centro-symmetric mode -> Split power randomly on degenerates, energy sum must be 2x
                    Cp1 = LPModalDecomposition.overlap_integral(obj.speckle.field, mode0_field);
                    Cp2 = LPModalDecomposition.overlap_integral(obj.speckle.field, mode90_field);
                    obj.modes_coeffs(i,:) = [Cp1, Cp2];
                end
            end

            obj.normalize_coeffs();
        end
    end

    methods (Access = protected)
        function normalize_coeffs(obj)
            coeffs_abs = abs(obj.modes_coeffs);
            coeffs_angles = angle(obj.modes_coeffs);

            coeffs_abs = coeffs_abs / sum(abs(coeffs_abs), 'all', 'omitnan');
            coeffs_angles = angle(exp(1i * (coeffs_angles - coeffs_angles(1,1))));

            obj.modes_coeffs = coeffs_abs .* exp(1i * coeffs_angles);
        end
    end

    methods (Static)
        function overlap = overlap_integral(field, mode)
            overlap = sum(field .* conj(mode), 'all') / sum(mode .* conj(mode), 'all');
        end
    end
end

