#! /usr/bin/env omegafig
# -*- mode: python; coding: utf-8 -*-
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

"""Plot the full VLA data set.

"""

import sys ; sys.path.insert (0, '.') # needed since we're omegafig.

from astropy.time import Time
import numpy as np
import omega as om

import photom
from plots import *

DMJD_XMIN = 0.44
DMJD_XMAX = 0.92
YMIN = -160
YMAX = 760

PULSES_IN_DAY = {
    0: 6,
    1: 7,
    2: 7,
}

def dhr_to_ut(dhr):
    mjd = dhr/24 + photom.MJD0
    t = Time(mjd, format='mjd', scale='utc')
    return t.utc.strftime('%H:%M')


def plot():
    df = photom.load_and_reduce('../target.phot.ll.txt')
    df['dhr'] = df['dmjd'] * 24
    soln = photom.BEST_SOLN

    toa_nrot, toas = photom.best_toas(df)
    toa_info = dict(zip(toa_nrot, toas - photom.MJD0))

    vb = om.layout.VBox(3)

    for dday in (0, 1, 2):
        subset = (df['dmjd'] >= dday) & (df['dmjd'] < (dday + 1))

        p = om.RectPlot()

        # Zero reference line
        p.addXY(
            [24 * (DMJD_XMIN + dday), 24 * (DMJD_XMAX + dday)], [0, 0],
            None,
            lineStyle = {'dashing': [3, 3], 'color': 'muted'},
            dsn = 0
        )

        # Actual data
        t = Time(np.median(df[subset]['dmjd']) + photom.MJD0, format='mjd', scale='utc')
        utdate = t.utc.strftime('%Y %b %d')
        p.addDF(df[subset][['dhr', 're', 'ure']], f'Day {dday+1} ({utdate} UT)')

        # Markers for the best-fit pulse ephemeris
        dmjdmin = DMJD_XMIN + dday
        nmin = (dmjdmin - soln.t0) / soln.period
        nmin = int(np.ceil(nmin))
        n = nmin

        while True:
            dmjd = soln.n_to_mjd(n) - photom.MJD0
            if dmjd > DMJD_XMAX + dday:
                break

            if n == nmin or n == nmin + PULSES_IN_DAY[dday] - 1:
                t = f'<span size="larger" weight="700">{n+1}</span>'
                p.add(XYText(24 * dmjd, 380, t))

            toa = toa_info.get(n)
            if toa is not None:
                p.addXY(
                    [24 * toa, 24 * toa], [150, 450],
                    None,
                    lineStyle = {'linewidth': 0.5, 'color': (0, 0, 0), 'dashing': [2, 2]},
                    dsn = 0
                )

            p.addXY(
                [24 * dmjd, 24 * dmjd], [250, 350],
                None,
                lineStyle = {'linewidth': 3, 'color': (0, 0, 0)},
                dsn = 0
            )
            n += 1

        # Labeling etc
        p.setBounds(24 * (DMJD_XMIN + dday), 24 * (DMJD_XMAX + dday), YMIN, YMAX)
        p.defaultKeyOverlay.hAlign = 0.97
        p.defaultKeyOverlay.vAlign = 0.07
        p.bpainter.numFormat = dhr_to_ut
        p.bpainter.minorTicks = 4

        p.add(TextOverlay(0.02, 0.04, '<span size="xx-large" weight="700">(%s)</span>' % (chr(ord('A') + dday))),
              rebound=False)

        vb[dday] = p

    vb[1].setYLabel('Flux density (μJy)')
    vb[2].setXLabel('Universal Time (UT)')

    return vb
