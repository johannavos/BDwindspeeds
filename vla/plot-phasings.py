#! /usr/bin/env omegafig
# -*- mode: python; coding: utf-8 -*-
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

"""Plot the light curves with different phasings to demonstrate the options.

"""

import sys ; sys.path.insert (0, '.') # needed since we're omegafig.

import numpy as np
import omega as om

import photom
from plots import *

n_trials = 5
delta_day = 0.007 / 24 # days
dt_sep_cutoff = 0.005 # days
stamp = lambda: om.stamps.Circle(size=3)
cmpt = 're'


def plot_one(plotnum, data, prd):
    data['n'] = (data['mjd'] - 58404.598) / prd + 0.5
    data['ph'] = data['n'] % 1.
    nmax = int(np.ceil(data['n'].max()))

    p = om.RectPlot()

    for i in range(nmax):
        s = data[(data['n'] >= i) & (data['n'] < i + 1)]

        # All this junk to avoid connecting over bandpass cal visits
        t = np.asarray(s.mjd)
        w = np.where((t[1:] - t[:-1]) > dt_sep_cutoff)[0]
        if w.size == 0:
            segments = [s]
        else:
            assert w.size == 1
            tcut = s.mjd.iloc[w[0]]
            segments = [s[s.mjd <= tcut], s[s.mjd > tcut]]

        for seg in segments:
            p.addDF(seg[['ph', cmpt, 'u'+cmpt]], None, dsn=i, lines=True,
                     pointStamp=stamp())

    p.addKeyItem('%.3f hr' % (24 * prd))
    for ap in p.bpainter, p.tpainter:
        ap.everyNthMajor = 2
        ap.minorTicks = 2
        ap.majorTickScale = 1.7
    p.setXLabel('Phase')
    p.lpainter.majorTickScale = p.rpainter.majorTickScale = 1.7

    if plotnum == 0:
        p.setYLabel('Flux Density(μJy)')
    else:
        p.lpainter.paintLabels = False

    p.defaultKeyOverlay.hAlign = 0.89
    p.defaultKeyOverlay.vAlign = 0.04
    p.setBounds(xmin=-0.05, xmax=1.05, ymin=-70, ymax=800)
    return p


def plot():
    df = photom.load_and_reduce('dynfilt.ll.txt')

    prd = photom.BEST_SOLN.period - (n_trials // 2) * delta_day
    hb = om.layout.HBox(n_trials)

    for i in range(n_trials):
        hb[i] = plot_one(i, df, prd)
        prd += delta_day

    hb.setWeight(0, 1.35) # extra width for left axis labeling
    hb.padSize = 6
    return hb
