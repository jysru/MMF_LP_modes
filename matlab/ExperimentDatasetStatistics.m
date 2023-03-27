classdef ExperimentDatasetStatistics < handle
    % Experimental dataset is loaded and basic statistics are computed
    
    properties (SetAccess = private, GetAccess = public)
        path
        name
        ext
        
        length
        image_size
        intens
        average_intens
        energy
        centroid_px
    end

    properties (SetAccess = private, GetAccess = private)
        default_intens_varname = 'CP'
    end

    
    methods (Access = public)
        function obj = ExperimentDatasetStatistics(path)
            arguments
                path char
            end

            obj.load_dataset(path);
            obj.average_image();
            obj.energy_statistics();
            obj.compute_centroid();
        end


        function plot(obj, opts)
            arguments
                obj (1,1) ExperimentDatasetStatistics
                opts.fig_num (1,1) double {mustBeNumeric, mustBePositive} = 1
            end

            figure(opts.fig_num)
            imagesc(obj.average_intens)
            colormap("hot")
            colorbar
            axis square
            xlabel('x [px]')
            ylabel('y [px]')
            title('Average intensity image')
        end
    end


    methods (Access = protected)
        function load_dataset(obj, path)
            try
                dset = load(path);
                [obj.path, obj.name, obj.ext] = fileparts(path);
            catch
                error(['Could not load dataset from ' path])
            end

            try
                obj.intens = dset.(obj.default_intens_varname);
                obj.length = numel(obj.intens);
                obj.image_size = size(obj.intens{1});
            catch
                error(['Loaded dataset has no field named: ' obj.default_intens_varname])
            end
        end


        function average_image(obj)
            s = 0;
            for i=1:numel(obj.intens)
                s = s + obj.intens{i};
            end
            obj.average_intens = s / numel(obj.intens);
        end


        function energy_statistics(obj)
            obj.energy = struct('array', [], 'mean', [], 'std', [], ...
                        'median', [], 'max', [], 'min', []);

            obj.energy.array = cellfun(@(x) sum(x, 'all'), obj.intens);
            obj.energy.mean = mean(obj.energy.array, 'all');
            obj.energy.std = std(obj.energy.array, 0, 'all');
            obj.energy.median = median(obj.energy.array, 'all');
            obj.energy.min = min(obj.energy.array, [], 'all');
            obj.energy.max = max(obj.energy.array, [], 'all');
        end


        function compute_centroid(obj)
            % Normalize and get the total sum
            s = obj.average_intens / max(obj.average_intens, [], 'all');
            st = sum(s, 'all');

            % Define position vectors 
            xi = 0:(size(s,1)-1);
            yi = 0:(size(s,2)-1);

            % Compute center of gravity coordinates
            x = sum(sum(xi.'.*s, 1), 2)/st;
            y = sum(sum(yi.*s, 2), 1)/st;

            obj.centroid_px = round([y, x]);
        end

    end

end

