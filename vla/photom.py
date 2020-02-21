# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

import numpy as np
import omega as om
import pandas as pd
from pwkit import numutil

# Saving results of manual analysis ...

MJD0 = 58403.

OBS_SPANS = [
    (58403.4569, 58403.9024),
    (58404.4542, 58404.8998),
    (58405.4514, 58405.8970),
]

# nrot values of all potential pulses inside our observation spans, given
# visual inspection of the data and the assumption that nothing wacky is
# happening.
NROT_ALL = np.array([
    0,  1,  2,  3,  4,  5,
    14, 15, 16, 17, 18, 19,
    27, 28, 29, 30, 31, 32,
])

# approximate peak heights -- taken from finding max(re) in windows around the
# times associated with NROT_ALL, for the TAKE1 ephemeris, using smoothed
# filtered data based in the LL lightcurve
APPROX_PEAK_HEIGHTS = np.array([
    167, 186, 147, 624, 133, 287,
    248, 684, 188, 410, 147, 221,
    211, 350, 679, 168, 133, 324,
])

# nrot values for the observed peaks, in approximate descending order of peak
# flux -- for an ill-defined lightcurve processing.
HIGHEST_PEAK_NROTS = np.array([
    15, 29, 3, 17, 28, 32, 5, 14, 19,
    27, 16, 1, 30, 0, 18, 2, 31, 4,
])

# TOAs from visual inspection of the data:
TOAS_TAKE1 = np.array([ # MJD - MJD0
    0.501, 0.573, 0.644, 0.718, 0.799, 0.864,
    1.516, 1.598, 1.664, 1.742,
    2.479, 2.552, 2.624, 2.843,
])

PERIOD_TAKE1 = 0.0732364 # days
T0_TAKE1 = 0.498741 # days, MJD - MJD0

# nrot values of the TOAS -- note that there are moments that we should
# observe a pulse, but do not.
NROT_TAKE1 = np.array([
    0,  1,  2,  3,  4,  5,
    14, 15, 16, 17,
    27, 28, 29, 32,
])
# Was "VISUAL" peak observed in the TAKE1 analysis?
PKOBS_TAKE1 = np.array([
    True, True, True, True, True, True,
    True, True, True, True, False, False,
    True, True, True, False, False, True,
])

# "take2" solution -- use the take1 solution to find TOAs from centroids and
# fit the result. The result is:

PERIOD_TAKE2 = 0.07319469  # days
T0_TAKE2 = 0.4983490609229193  # days, MJD - MJD0

# Update nrot array based on visual inspection with debug_toa_analysis
NROT_VISUAL_ITER2 = np.array([
    0,  1,  2,  3,  4,  5,
    14, 15, 16, 17, 19,
    28, 29, 30, 32,
])

BASELINE_MODEL_POLYNOMIAL_ORDER = 0  # i.e., constant


def load_and_reduce(path='target.phot.i.txt', nbin=10):
    df = pd.read_csv(path, sep='\t')

    # Phase cal visits are just ~1.6 minutes usually; slightly long breaks
    # between scans, or something, are 0.25 minutes. Breaks between days are
    # ~795 minutes.
    max_time_gap = 1. / 1440

    if nbin == 1:
        rdf = df
    else:
        rdf = numutil.reduce_data_frame_evenly_with_gaps(
            df, 'mjd',
            nbin,  # target bin size
            max_time_gap,
            uavg_cols = 're im'.split(),
            avg_cols = ['mjd'],
        )

    rdf['dmjd'] = rdf['mjd'] - MJD0
    rdf['dhr'] = rdf['dmjd'] * 24.

    return rdf


def phase_plot(df, period, yofs=200, phofs=0., lines=False, errs=True,
               psf='default', byrot=True):
    if psf == 'default':
        psf = om.stamps.Circle
    elif psf is None:
        psf = lambda: None

    df['nrot'] = (df['mjd'] - df['mjd'].min()) / period
    df['ph'] = (df['nrot'] + phofs) % 1.

    nmax = int(np.floor(df['nrot'].max()))

    p = om.RectPlot()
    nplot = 0

    for i in range(nmax + 1):
        subset = (df['nrot'] >= i) & (df['nrot'] < i + 1)
        if not subset.any():
            continue

        x = df[subset]['ph']
        u = df[subset]['ure']

        if byrot:
            y = df[subset]['re'] + i * yofs
        else:
            y = df[subset]['re'] + nplot * yofs

        if errs:
            p.addXYErr(
                x, y, u,
                None,
                lines = lines,
                pointStamp = psf(),
            )
        else:
            p.addXY(
                x, y,
                None,
                lines = lines,
                pointStamp = psf(),
            )

        nplot += 1

    p.setBounds(-0.05, 1.05)
    return p


def snip_plot(df, t0, period, yofs=200, lines=False, errs=True, psf='default',
              tcol='dmjd', widthfactor=0.5):
    if psf == 'default':
        psf = om.stamps.Circle
    elif psf is None:
        psf = lambda: None

    p = om.RectPlot()
    nplot = 0
    tmax = df[tcol].max()
    width = widthfactor * period
    tleft = t0 - 0.5 * width

    while tleft < tmax:
        subset = (df[tcol] >= tleft) & (df[tcol] < tleft + width)
        if not subset.any():
            tleft += width
            continue

        x = df[subset][tcol] - (tleft + 0.5 * width)
        y = df[subset]['re'] + nplot * yofs
        u = df[subset]['ure']

        if errs:
            p.addXYErr(
                x, y, u,
                None,
                lines = lines,
                pointStamp = psf(),
            )
        else:
            p.addXY(
                x, y,
                None,
                lines = lines,
                pointStamp = psf(),
            )

        nplot += 1
        tleft += width

    p.setBounds(-0.5 * width, 0.5 * width)
    return p


def snip_page(df, t0, period, lines=False, errs=True, psf='default',
              tcol='dmjd', widthfactor=0.5):
    if psf == 'default':
        psf = om.stamps.Circle
    elif psf is None:
        psf = lambda: None

    pg = om.makeDisplayPager()
    tmax = df[tcol].max()
    width = widthfactor * period

    def gen_data():
        tleft = t0 - 0.5 * width

        while tleft < tmax:
            subset = (df[tcol] >= tleft) & (df[tcol] < tleft + width)
            if not subset.any():
                tleft += width
                continue

            x = df[subset][tcol] - (tleft + 0.5 * width)
            y = df[subset]['re']
            u = df[subset]['ure']
            yield x, y, u
            tleft += width

    ymin = ymax = None

    for x, y, u in gen_data():
        if ymin is None:
            ymin = (y - u).min()
            ymax = (y + u).max()
        else:
            ymin = min(ymin, (y - u).min())
            ymax = max(ymax, (y + u).max())

    height = ymax - ymin
    if height == 0:
        height = 0.5 * ymax
    if height == 0:
        height = 1
    ymax = ymax + 0.05 * height
    ymin = ymin - 0.05 * height

    for x, y, u in gen_data():
        p = om.RectPlot()

        if errs:
            p.addXYErr(
                x, y, u,
                None,
                lines = lines,
                pointStamp = psf(),
            )
        else:
            p.addXY(
                x, y,
                None,
                lines = lines,
                pointStamp = psf(),
            )

        p.setBounds(-0.5 * width, 0.5 * width, ymin, ymax)
        pg.send(p)

    pg.done()


# Probably overkill object orientation

class Solution(object):
    def __init__(self, t0, period):
        self.t0 = t0 # MJD - MJD0
        self.period = period # days

    def n_to_mjd(self, n):
        return MJD0 + self.t0 + self.period * n

    def pk_mjds(self):
        return self.n_to_mjd(NROT_ALL)

    def pk_indices(self, mjds):
        i = [np.argmin(np.abs(mjds - pmjd)) for pmjd in self.pk_mjds()]
        return np.array(i)

    def bin_index(self, nbin, mjd):
        nrot = (mjd - (MJD0 + self.t0)) / self.period
        ph = nrot % 1
        return np.minimum(np.round(ph * nbin).astype(np.int_), nbin - 1)

    def slice(self, df, n, widthfactor=0.4):
        """Extract a subset of a dataframe around peak #n.

        The dataframe must have a column "mjd". The return value will include
        all points with an mjd within *width* of the peak, where `width =
        widthfactor * period`. It will have a new column, *dmjd*, which is the
        difference between mjd and the peak MJD.

        """
        mjd = self.n_to_mjd(n)
        halfwidth = 0.5 * self.period * widthfactor
        filt = (df['mjd'] >= mjd - halfwidth) & (df['mjd'] <= mjd + halfwidth)
        res = df[filt].copy()
        res['dmjd'] = res['mjd'] - mjd
        return res

    def grid_no_errors(self, mjd, val, n, dmjd):
        """Interpolate data onto a common grid relative to some nrot value.

        Given dates *mjd*, values *val* sampled at those dates, and n_rot
        value *n*, we first calculate dmjd coordinates for each sample
        relative to the moment specified by *n*. We then linearly interpolate
        between these values to derive measurements specified by the
        relative-time grid *dmjd*.

        Note: errors/uncertainties are ignored! And linear interpolation is
        lame.

        """
        samp_dmjd = mjd - self.n_to_mjd(n)
        return np.interp(dmjd, samp_dmjd, val)

    def rebin(self, mjd_edges, val, wt, n, dmjd_edges):
        """Rebin data onto a common grid relative to some nrot value.

        Inputs:

        *mjd_edges* -- MJD bin edges; there are `n_val+1` of them. E.g.
          the first measurement bin covers the time range `mjd_edges[0]`
          to `mjd_edges[1]`; the final covers `mjd_edges[n_val-1]` to
          `mjd_edges[n_val]`. Must be sorted.

        *val* -- data values; there are `n_val` of them. Value kind is
          not important.

        *wt* -- weights to assign to the data values; there are `n_val`
          of them.

        *n* -- an n_rot value specifying the dmjd reference point.

        *dmjd_edges* -- delta-MJD bin edges; the values are resampled onto
          bins spanning `dmjd_edges[0]` to `dmjd_edges[1]`, `dmjd_edges[1]` to
          `dmjd_edges[2]`, etc. There are `n_dmjd+1` of them. Must be sorted.

        Returns: (grid_wtvals, grid_wts), each of size *n_dmjd*. The former is
          a sum of `val*wt`, dropped onto the new binning with appropriate
          weighting between bins.

        """
        n_val = val.size
        n_dmjd = dmjd_edges.size - 1

        assert mjd_edges.size == n_val + 1
        assert wt.size == val.size

        grid_wtvals = np.zeros(n_dmjd)
        grid_wts = np.zeros(n_dmjd)

        dmjd_widths = np.diff(dmjd_edges)

        samp_dmjd_edges = mjd_edges - self.n_to_mjd(n)
        samp_widths = np.diff(samp_dmjd_edges)

        for i_in in range(n_val):
            # Left and right edges of the input sample bin, in dmjd values
            samp_dmjd_left = samp_dmjd_edges[i_in]
            samp_dmjd_right = samp_dmjd_edges[i_in+1]

            if samp_dmjd_right < dmjd_edges[0]:
                continue  # this value is not on the output grid at all

            if samp_dmjd_left > dmjd_edges[-1]:
                break  # assuming ordered inputs, we'll all done

            # Index of the output bin into which the left edge of the input sample falls
            # If the left edge falls beyond the output grid, that's OK.
            i_bin_left = np.searchsorted(dmjd_edges, samp_dmjd_left) - 1
            i_bin_left = np.maximum(i_bin_left, 0)

            # Index of the output bin into which the right edge of the input sample falls
            i_bin_right = np.searchsorted(dmjd_edges, samp_dmjd_right) - 1
            i_bin_right = np.minimum(i_bin_right, n_dmjd - 1)

            for i_out in range(i_bin_left, i_bin_right + 1):
                out_dmjd_left = np.maximum(samp_dmjd_left, dmjd_edges[i_out])
                out_dmjd_right = np.minimum(samp_dmjd_right, dmjd_edges[i_out+1])
                out_width = out_dmjd_right - out_dmjd_left

                # What fraction of the input bin lands on this output bin?
                in_frac = out_width / samp_widths[i_in]

                # Accumulate
                out_wt = wt[i_in] * in_frac
                grid_wts[i_out] += out_wt
                grid_wtvals[i_out] += out_wt * val[i_in]

        return grid_wtvals, grid_wts

    def rebinned_weighted_sum(self, mjd_edges, val, wt, dmjd_edges, nvals=None):
        """Sum data in dmjd space for a bunch of nrot values."""

        if nvals is None:
            nvals = NROT_ALL

        gwv = np.zeros(dmjd_edges.size - 1)
        gw = np.zeros(dmjd_edges.size - 1)

        for n in nvals:
            igwv, igw = self.rebin(mjd_edges, val, wt, n, dmjd_edges)
            gwv += igwv
            gw += igw

        return gwv / gw

    def centroid_toas(self, df, nvals=None, widthfactor=0.7):
        from pwkit.lsqmdl import PolynomialModel

        if nvals is None:
            nvals = NROT_ALL

        toas = np.zeros(nvals.size)
        moment2s = np.zeros(nvals.size)
        weights = np.zeros(nvals.size)

        for i, n in enumerate(nvals):
            # Get the slice around the peak in question

            sl = self.slice(df, n, widthfactor=widthfactor)
            sl['wt'] = sl['ure']**-2
            sl['wtflux'] = sl['re'] * sl['wt']
            sl['invsigma'] = 1. / sl['ure']

            # We break this slice into two groups: the inner half and the
            # outer edges. E.g. ,if this slice spans from dmjd = -0.1 to 0.1,
            # the inner part goes from -0.05 to 0.05, and the outer part is
            # the rest.

            half_inner_width = 0.25 * self.period * widthfactor
            is_inner = np.abs(sl['dmjd']) < half_inner_width

            # Fit a linear flux trend to the outer part and remove it from
            # the inner part.

            outer = sl[~is_inner]

            baseline = PolynomialModel(
                BASELINE_MODEL_POLYNOMIAL_ORDER,
                outer['dmjd'].values,
                outer['re'].values,
                outer['invsigma'].values,
            ).solve()

            inner = sl[is_inner]
            wtexcess = (inner['re'] - baseline.mfunc(inner['dmjd'])) * inner['wt']

            # weighted total excess flux in the inner slice is the weight
            # that we'll assign to the TOA:

            weights[i] = wtexcess.sum() / inner['wt'].sum()

            # centroid/center-of-mass with regards to weighted excess flux is the
            # TOA.

            toas[i] = (inner['mjd'] * wtexcess).sum() / wtexcess.sum()
            moment2s[i] = np.sqrt(((inner['mjd'] - toas[i])**2 * wtexcess).sum() / wtexcess.sum())

        return toas, moment2s, weights

    def iterate_toa_method(self, df, use_weights=False, nvals=NROT_ALL, **kwargs):
        """Derive a new solution, hopefully higher-quality, by determining TOAs and
        then fitting a period to them.

        """
        toas, moment2s, weights = self.centroid_toas(df, nvals=nvals, **kwargs)

        if use_weights:
            assert np.all(weights > 0)
        else:
            weights[:] = 1.0

        from pwkit.lsqmdl import PolynomialModel
        invsigma = weights**0.5
        ephem = PolynomialModel(1, nvals, toas, invsigma).solve()
        t0 = ephem.params[0] - MJD0
        soln = Solution(t0, ephem.params[1])
        soln.basis_nvals = nvals
        soln.basis_toas = toas
        soln.basis_moment2s = moment2s
        soln.basis_weights = weights
        soln.basis_residuals = ephem.resids
        soln.rms_residual = np.sqrt((soln.basis_residuals**2).mean())
        soln.naive_period_uncert = ephem.as_nonlinear().puncerts[1] * np.sqrt(ephem.rchisq)
        return soln

    def naive_period_uncertainty(self, ilo=0, ihi=-1):
        """Calculate a period uncertainty using the centroid second moments as naive
        uncertainties on the first and last TOAs. This is pretty simpleminded,
        but in my investigations it yields a value that is extremely
        reasonable, and I don't believe that there are any better-justified
        methods out there.

        """
        nrot = self.basis_nvals[ihi] - self.basis_nvals[ilo]
        p_hi = (((self.basis_toas[ihi] + self.basis_moment2s[ihi]) - (self.basis_toas[ilo] - self.basis_moment2s[ilo]))
                / nrot)
        p_lo = (((self.basis_toas[ihi] - self.basis_moment2s[ihi]) - (self.basis_toas[ilo] + self.basis_moment2s[ilo]))
                / nrot)
        return p_lo, p_hi, 0.5 * (p_hi - p_lo)


SOLN_TAKE1 = Solution(T0_TAKE1, PERIOD_TAKE1)
SOLN_TAKE2 = Solution(T0_TAKE2, PERIOD_TAKE2)

# This is what we get from `./periodicity-analysis.py`:
BEST_SOLN = Solution(4.9827137e-01, 7.3253950e-02)
BEST_WIDTHFACTOR = 0.6  # also from periodicity-analysis.py

def best_toas(df, nrot=NROT_ALL):
    toas, moment2s, weights = BEST_SOLN.centroid_toas(df, nvals=nrot, widthfactor=BEST_WIDTHFACTOR)
    return nrot, toas


def debug_toa_analysis(soln, df, toas, weights,
                       nrots = NROT_ALL,
                       widthfactor = 0.7,
                       sort = 'seq',
                       yrange = None,
):
    # Solve for the period so that we can figure out how to convert the
    # weights into uncertainties.

    from pwkit.lsqmdl import PolynomialModel
    invsigma = weights**0.5
    mtoa = toas.mean()
    ephem = PolynomialModel(1, nrots, toas - mtoa, invsigma).solve()
    ephem.print_soln()
    print('T0:', ephem.params[0] + mtoa - MJD0)
    print('RMS residual:', (ephem.resids**2).mean()**0.5)
    wtscale = np.sqrt(ephem.rchisq)

    uncerts = wtscale * weights**-0.5

    pg = om.makeDisplayPager()
    half_inner_width = 0.25 * soln.period * widthfactor

    if sort == 'seq':
        sort_idx = np.arange(nrots.size)
    elif sort == 'resid':
        sort_idx = np.argsort(ephem.resids)
    else:
        raise ValueError(f'unhandled sort {sort!r}')

    for i in sort_idx:
        n = nrots[i]
        sl = soln.slice(df, n, widthfactor=widthfactor)
        sl['invsigma'] = 1. / sl['ure']
        toa = toas[i]
        mjd = soln.n_to_mjd(n)

        # Repeat the solution of the baseline flux used to determine
        # the excess.

        is_inner = np.abs(sl['dmjd']) < half_inner_width
        outer = sl[~is_inner]
        baseline = PolynomialModel(
            BASELINE_MODEL_POLYNOMIAL_ORDER,
            outer['dmjd'].values,
            outer['re'].values,
            outer['invsigma'].values,
        ).solve()

        p = om.RectPlot()

        # The TOA for this slice with its uncertainty, as determined from its
        # weight and the RMS of the residuals to the period fit.
        p.add(om.rect.XBand(toa - uncerts[i], toa + uncerts[i], keyText=None), dsn=0)
        p.addVLine(toa, keyText='n=%d Centroid TOA' % n, dsn=0)

        # The actual data.
        p.addDF(sl[['mjd', 're', 'ure']], dsn=1)

        # The definitions of the inner/outer windows used for the baseline fit and
        # excess flux computation.
        p.addVLine(mjd - half_inner_width, dsn=2, lineStyle={'dashing': [3, 3]}, keyText=None)
        p.addVLine(mjd + half_inner_width, dsn=2, lineStyle={'dashing': [3, 3]}, keyText=None)

        # The baseline flux fit
        bl_dmjd = np.linspace(-2 * half_inner_width, 2 * half_inner_width, 3)
        p.addXY(bl_dmjd + mjd, baseline.mfunc(bl_dmjd), None, dsn=2)

        # The new ephemeris fit
        label = 'New ephem (resid = %.5f = %.1fσ)' % (ephem.resids[i], ephem.resids[i] / uncerts[i])
        mjd_ephem = ephem.mfunc(n) + mtoa
        p.addVLine(mjd_ephem, dsn=3, keyText=label)

        if yrange is not None:
            p.setBounds(ymin=yrange[0], ymax=yrange[1])

        p.bpainter.numFormat = '%.8g'

        pg.send(p)

    pg.done()
    return ephem


def monte_carlo_period_fit(
        toa_norm_scatter = None,
        nrot = NROT_ALL,
        true_period_range = (0.065, 0.085),
        nmc = 1000,
):
    from pwkit.lsqmdl import PolynomialModel

    frac_errors = np.zeros(nmc)

    for i in range(nmc):
        true_p = np.random.uniform(*true_period_range)
        true_toa = true_p * nrot
        obs_toa = true_toa + np.random.normal(scale=toa_norm_scatter, size=nrot.size) * true_p

        # NOTE: hardcoding that we're not weighting individual measurements
        soln = PolynomialModel(1, nrot, obs_toa).solve()
        obs_p = soln.params[1]

        frac_errors[i] = (obs_p - true_p) / true_p

    return frac_errors
