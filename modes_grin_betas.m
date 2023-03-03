function sorted_constants = modes_grin_betas( lambda, n1, n2, a)
    arguments
        lambda (1,1) double {mustBePositive, mustBeNonzero} = 1064e-9
        n1 (1,1) double {mustBePositive, mustBeNonzero} = 1.465
        n2 (1,1) double {mustBePositive, mustBeNonzero} = 1.45
        a (1,1) double {mustBePositive, mustBeNonzero} = 26e-6
    end

    NA = sqrt(n1^2 - n2^2);
    V = 2*pi*a * NA / lambda;
    N_modes = floor(V^2/4);
    k0 = 2*pi/lambda;
    delta = NA^2 / (2*n1^2);

    storage = nan(N_modes^2, 4, 'double'); % Columns with betah, h, n, m
    k = 1;
    for m=1:N_modes+1
        for n=0:N_modes
            h = 2*n + m-1;
            betah = k0*n1 - h*sqrt(2*delta)/a;
            nh = betah*lambda/(2*pi);
            storage(k, :) = [nh, h, n, m];
            k = k+1;
        end
    end

    sorted_constants = sortrows(storage, 'descend');
    if size(sorted_constants,1) > N_modes
        sorted_constants = sorted_constants(1:N_modes,:);
    end
end


