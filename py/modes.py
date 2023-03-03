import numpy as np
import matplotlib.pyplot as plt

def modes_factorial(idx: int):
    fac = 1
    if idx==0:
        fac = 1
    else:
        for i in range(1, idx+1):
            fac *= i
    return fac


def modes_grin(m, n,
               n1:float=1.465, n2: float=1.45,
               wavelength: float=1064e-9, a: float=26e-6,
               theta0_deg: float=0,
               ):
    
    # n += 1
    theta0 = theta0_deg * np.pi / 180
    NA = np.sqrt(np.square(n1) - np.square(n2))
    V = 2 * np.pi * a * NA / wavelength
    
    # Compute factorials for n
    facn = modes_factorial(n)
    facm_plus_n = modes_factorial(m+n)

    # Compute Kronecker delta0m and epsilon
    delta0m = 1 if m==0 else 0
    epsilon_mn = np.pi * np.square(a) * facm_plus_n * (1+delta0m) / (2*V*facn)

    # Compute scalar field
    xmax = 1.2 * a
    nbre_pts = 4 * 512

    x = np.linspace(-xmax, xmax, nbre_pts)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(np.square(X) + np.square(Y))
    ro = R / a * np.sqrt(V)
    Theta = np.arctan2(Y, X)
    
    Lnm = 0
    for s in range(0, n+1):
        facs = modes_factorial(s)
        facn_moins_s = modes_factorial(n-s)
        facm_plus_s = modes_factorial(m+s)

        num = facm_plus_n * np.power(-1, s) * np.power(ro, 2*s)
        denom = facm_plus_s * facn_moins_s * facs
        Lnm += num / denom

    # Field computation
    fac1 = 1 / np.sqrt(epsilon_mn)
    fac2 = np.power(ro, m)
    fac3 = np.exp(-np.square(ro)/2)
    field = fac1 * fac2 * fac3 * Lnm * np.cos( m * Theta * theta0)
        
    return (field, X, Y)


if __name__ == '__main__':
    field, X, Y = modes_grin(m=2, n=3)

    plt.figure()
    plt.imshow(field)
    plt.colorbar()
    plt.show()
    