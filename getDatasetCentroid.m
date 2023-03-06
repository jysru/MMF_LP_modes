function [x, y, s, em] = getDatasetCentroid(dset_path)
    % Gets the center of gravity of intensity images from a dataset
    arguments
        dset_path char
    end
    
    % Load the dataset
    dset = load(dset_path);

    % Get the sum of all images
    s = 0;
    for i=1:numel(dset.CP)
        s = s + dset.CP{i};
    end
    s = s / numel(dset.CP);

    % Get energies
    sum_intens = cellfun(@(x) sum(x, 'all'), dset.CP);
    em = mean(sum_intens, 'all');

    % Normalize and get the total sum
    s = s / max(s, [], 'all');
    st = sum(s, 'all');

    % Define position vectors 
    xi = 0:(size(s,1)-1);
    yi = 0:(size(s,2)-1);
    
    % Compute center of gravity coordinates
    x = sum(sum(xi.*s, 1), 2)/st;
    y = sum(sum(yi.*s, 2), 1)/st;
end


