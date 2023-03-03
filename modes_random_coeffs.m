function [ Cp ] = modes_random_coeffs( p_modes, n )
    arguments
        p_modes (1,1) double {mustBeNonnegative, mustBeInteger} = 1
        n (1, 1) double {mustBePositive, mustBeInteger} = 1
    end

    % Generate vector that sums up to one (intensity coefficients)
    Ip = rand(p_modes, n);
    Ip = Ip./sum(Ip, 1);

    % Generate random phases
    Phip = -pi + 2*pi*rand(p_modes, n);

    % Get the complex coefficients
    Cp = sqrt(Ip) .* exp(1i * Phip );
end
