#! /usr/bin/env python3
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

"""End-to-end periodicity analysis of the VLA data.

The fundamental fact here is that we just don't have enough data+knowledge to
really analyze the system in a definitively rigorous way. We do the best we
can and try to be sensible and conservative.

"""

import numpy as np

import photom


class PARAMS(object):
    data_path = 'dynfilt.ll.txt'
    data_nbin = 1

    # If we calculate weights and then run a fit *without* weights, visual
    # inspection suggests that the points that *would have been* more highly
    # weighted have systematically smaller residuals against the fit. This
    # suggests to me that it is indeed appropriate to weigh them more heavily
    # when doing the fit.
    take2_use_weights = True

    # Visual inspection of debug_toa_analysis shows that some "peaks" look to
    # be totally worthless, and widthfactor of 0.6 seems appropriate for
    # separating baseline and pulse.
    take2_nvals = photom.NROT_VISUAL_ITER2
    take2_widthfactor = 0.6

    mc_period_range = (0.0632, 0.0832)
    mc_size = 4096


def print_soln(soln, label):
    pstr = f'{soln.period:.7e}'
    t0str = f'{soln.t0:.7e}'
    rmsstr = f'{soln.rms_residual:.4e}'
    print(f'{label:10}: P = {pstr:>15s} days, t0 = {t0str:>15s} delta-MJD, rms_resid = {rmsstr:>12s} days')


def main():
    df = photom.load_and_reduce(PARAMS.data_path, PARAMS.data_nbin)

    take1 = photom.SOLN_TAKE1
    take1.rms_residual = np.nan
    print_soln(take1, 'Take1(*)')

    take2 = take1.iterate_toa_method(
        df,
        use_weights = PARAMS.take2_use_weights,
        nvals = PARAMS.take2_nvals,
        widthfactor = PARAMS.take2_widthfactor,
    )
    print_soln(take2, 'Take2')

    print(f'*** Best-fit period: {24 * take2.period} hr')

    print(f'RMS residual: {24 * take2.rms_residual} hr')
    norm_scatter = take2.rms_residual / take2.period
    print(f'Normalized RMS residual: {norm_scatter}')
    print(f'BAD naive least-squares period uncertainty: {24 * take2.naive_period_uncert} hr')

    np_lo, np_hi, u_naive = take2.naive_period_uncertainty()
    print(f'*** Naive period uncertainty: {u_naive} d = {24 * u_naive} hr')
    print(f'Bracketing values from naive analysis: {24 * np_lo} -- {24 * np_hi} hr')
    print(f'Preferred value from naive analysis: {12 * (np_lo + np_hi)} hr')


if __name__ == '__main__':
    main()
