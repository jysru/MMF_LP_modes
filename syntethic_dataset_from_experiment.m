function syntethic_dataset_from_experiment(file, opts)
    arguments
        file
        opts.magnification (1,1) double {mustBeNumeric, mustBeNonzero} = 12.62
        opts.camera_pixel_size (1,1) double {mustBeNumeric, mustBePositive} = 5.04e-6
        opts.N_modes (1,1) double {mustBeNumeric, mustBePositive} = 20
        opts.length (1,1) double {mustBeNumeric, mustBePositive} = 32
        opts.noise (1,1) double {mustBeNumeric, mustBeNonnegative} = 0
        opts.normalize (1,1) logical {mustBeNumericOrLogical} = true
        opts.type char {mustBeMember(opts.type, {'pure', 'speckle'})} = 'pure'

        opts.show_results (1,1) logical {mustBeNumericOrLogical} = false
        opts.save_results (1,1) logical {mustBeNumericOrLogical} = true
        opts.save_name char = ''
        opts.save_path char = ''
    end


    % Parse inputs
    mag = opts.magnification;
    cam_px = opts.camera_pixel_size;
    

    % Get statistics from experimental dataset
    stats = ExperimentDatasetStatistics(file);
    cam_size = stats.image_size;


    % Define grid with offset, magnification, and centroids
    xp = round(stats.centroid_px(2) - cam_size(2)/2)*opts.camera_pixel_size;
    yp = round(stats.centroid_px(1) - cam_size(1)/2)*opts.camera_pixel_size;
    grid = CameraGrid(pixel_numbers=stats.image_size, pixel_size=cam_px/mag, offsets=[xp, yp]/mag);
    

    % Generate dataset
    fiber = GrinFiber();
    if opts.N_modes > fiber.N_modes
        opts.N_modes = fiber.N_modes;
    end

    if strcmpi(opts.type, 'pure')
        dset = GrinLPDataset(fiber, grid, N_modes=opts.N_modes, noise=opts.noise);
    else
        dset = GrinLPSpeckleDataset(fiber, grid, N_modes=opts.N_modes, noise=opts.noise, length=opts.length);
    end
    intens = dset.intensity;


    % Normalize generated dataset
    if opts.normalize
        for i=1:size(intens,3)
            img = intens(:,:,i);
            image_energy = sum(img, "all");
            intens(:,:,i) = img / image_energy * stats.energy.mean;
        end
    end


    % Save results if flag is true
    if opts.save_results
        if strcmp(opts.save_name, '')
            prefix = ['synth_dset_',  opts.type];
            suffix = '.mat';
            str_N_modes = ['_modes=', num2str(opts.N_modes, '%d')];
            str_noise = ['_noise=', num2str(opts.noise, '%.3f')];
            ds_name = ['_ds=', fullfile(stats.name)];
            if strcmpi(opts.type, 'pure')
                len = '';
            else
                len = ['_length=', num2str(opts.length, '%d')];
            end
            save_name = [prefix, str_N_modes,  str_noise, len, ds_name];
        else
            save_name = opts.save_name;
        end

        if strcmp(opts.save_path, '')
            save_path = stats.path;
        else
            save_path = opts.save_path;
        end

        centroid = [xp, yp];
        energy = stats.energy;
        fullpath = fullfile(save_path, [save_name, suffix]);
        save(fullpath, 'intens', 'centroid', 'energy', '-v6');
    end


    % Show some plots if flag is true
    if opts.show_results
        stats.plot(fig_num=1)
        
        img = intens(:,:,1);
        if ~opts.normalize
            image_energy = sum(img, "all");
            img = img / image_energy * stats.energy.mean;
        end
        
        figure(2)
        imagesc(img)
        colormap("hot")
        colorbar
        xlabel('x [px]')
        ylabel('y [px]')
        title('Example of generated image')
    end

end
