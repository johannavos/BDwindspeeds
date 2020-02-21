#! /usr/bin/env python
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

import numpy as np
import pandas as pd
import scipy.signal

prod = 'll'

def main():
    with open(f'target.dynspec.{prod}.npy', 'rb') as f:
        mjds = np.load(f)
        freqs = np.load(f)
        data = np.load(f)

    # Good frequencies determined by collapsing `ure` along time axis, then
    # manually inspecting values as a function of frequency:

    goodfreqs = np.zeros(128, dtype=np.bool_)
    goodfreqs[28:59] = 1
    goodfreqs[82:103] = 1
    goodfreqs[119:124] = 1

    # Good times. NaN-median along frequency axis, then look for fractional
    # deviation against a median-smoothed version of that time series.

    zt = np.nanmedian(data[1], axis=1)
    mf = scipy.signal.medfilt(zt, 31)  # median filter to look for sharp spikes (RFI); width empirical
    fdev = (zt - mf) / mf  # fractional deviation from the median-filtered version
    goodtimes = (fdev < 0.2)  # empirical/visual threshold for not-high-noise times

    # Construct masked, 2D versions of all of our quantities

    re = np.ma.MaskedArray(data[0], copy=True)
    re.mask = ~np.isfinite(re)
    re.mask[:,~goodfreqs] = True
    re.mask[~goodtimes,:] = True

    ure = np.ma.MaskedArray(data[1], copy=True)
    ure.mask = re.mask

    im = np.ma.MaskedArray(data[2], copy=True)
    im.mask = re.mask

    uim = np.ma.MaskedArray(data[3], copy=True)
    uim.mask = re.mask

    # Average over the frequency axes to get time series.

    re_wt = ure**-2
    tot_re_wt_ts = re_wt.sum(axis=1)
    re_ts = (re * re_wt).sum(axis=1) / tot_re_wt_ts
    ure_ts = tot_re_wt_ts**-0.5

    im_wt = uim**-2
    tot_im_wt_ts = im_wt.sum(axis=1)
    im_ts = (im * im_wt).sum(axis=1) / tot_im_wt_ts
    uim_ts = tot_im_wt_ts**-0.5

    # Jy to uJy

    re_ts *= 1e6
    ure_ts *= 1e6
    im_ts *= 1e6
    uim_ts *= 1e6

    # Pack into a DataFrame ...

    keep = ~re_ts.mask

    df = pd.DataFrame(dict(
        mjd = mjds[keep],
        re = re_ts.data[keep],
        ure = ure_ts.data[keep],
        im = im_ts.data[keep],
        uim = uim_ts.data[keep],
    ))

    # Save

    df.to_csv(
        f'dynfilt.{prod}.txt',
        sep = '\t',
        index = False
    )


if __name__ == '__main__':
    main()
