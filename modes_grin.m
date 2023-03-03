function [ champ, X, Y ] = modes_grin( m, n, options)
    arguments
        m (1,1) double {mustBeNonnegative, mustBeInteger}
        n (1,1) double {mustBeNonnegative, mustBeInteger}
        
        options.theta0 (1,1) double {mustBeNumeric} = 0
        options.lambda (1,1) double {mustBePositive, mustBeNonzero} = 1064e-9
        options.n1 (1,1) double {mustBePositive, mustBeNonzero} = 1.465
        options.n2 (1,1) double {mustBePositive, mustBeNonzero} = 1.45
        options.a (1,1) double {mustBePositive, mustBeNonzero} = 26e-6
        options.grid_size (1,1) double {mustBeNonnegative} = 0
        options.grid_step (1,1) double {mustBePositive, mustBeNonzero} = 0.2e-6
        options.offsets (1,2) double = [0, 0]
    end

    theta0 = options.theta0;
    lambda = options.lambda;
    n1 = options.n1;
    n2 = options.n2;
    a = options.a;
    if options.grid_size == 0
        grid_size = 2.5*a;
    else
        grid_size = options.grid_size;
    end
    grid_step = options.grid_step;


    theta0 = theta0*pi/180;
    ON = sqrt(n1^2-n2^2);
    Lnm = 0;
    V = 2*pi*a* ON / lambda;
    
    facn = factorial(n);
    facm_plus_n = factorial(m+n);
    
    % calcul du symbole de Kroneker delta0m
    if m==0, delta0m = 1; else, delta0m = 0; end
    
    epsilon_mn=pi*a^2*facm_plus_n*(1+delta0m)/(2*V*facn);
    
    % CALCUL DU CHAMP SCALAIRE (modes LP) DU MODE LPm,n
    x = (-grid_size/2:grid_step:+grid_size/2) - options.offsets(1);
    y = (-grid_size/2:grid_step:+grid_size/2) - options.offsets(2);
    [X, Y] = meshgrid(x, y);
    R = sqrt(X.^2+Y.^2);
    ro = R/a*sqrt(V);
    theta = atan2(Y,X);
    
        
    
    % calcul du polynome de laguerre gauss Lnm
    for s=0:n
        facs = factorial(s);
        facn_moins_s = factorial(n-s);
        facm_plus_s = factorial(m+s);
    
        num = facm_plus_n*(-1)^s*ro.^(2*s);
        denom = facm_plus_s*facn_moins_s*facs;
        term_somme = num/denom;
        Lnm = Lnm+term_somme;
    end
    
    % calcul du champ 
    fac1 = 1./sqrt(epsilon_mn);
    fac2 = ro.^m;
    fac3 = exp(-(ro.^2)/2);

    champ = fac1.*fac2.*fac3.*Lnm.*cos(m*theta+theta0);

end