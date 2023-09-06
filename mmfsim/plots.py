import numpy as np
from matplotlib.colors import hsv_to_rgb


def complex2HSV(z, rmin, rmax, hue_start=0):
        # get amplidude of z and limit to [rmin, rmax]

        amp = np.abs(z)
        amp = np.where(amp < rmin, rmin, amp)
        amp = np.where(amp > rmax, rmax, amp)
        ph = np.angle(z, deg=True) + hue_start
        # HSV are values in range [0,1]
        h = (ph % 360) / 360
        s = 0.85 * np.ones_like(h)
        v = (amp - rmin) / (rmax - rmin)
        return hsv_to_rgb(np.dstack((h,s,v)))


def complex_image(complex_array, rmin: float = 0, rmax: float = None, hue_start: float = 0):
    if rmax is None:
          rmax = np.max(np.abs(complex_array))
    return complex2HSV(complex_array, rmin=rmin, rmax=rmax, hue_start=hue_start)
