# -*- mode: python; coding: utf-8 -*-
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

"""
Helpers for plotting (copied from goreham)

"""

import numpy as np
import omega as om

__all__ = '''
pangosub
figdefaults
add_dhr_top_axis
gen_flare_set_plots
plot_flare_zoom
plot_long_term_photom
BigArrow
KeyHighlightStamp
ManualStampKeyPainter
Rectangle
TextOverlay
XYText
'''.split()


def pangosub(s):
    return(s
           .replace('{^', '<span size="smaller" rise="2000">')
           .replace('{', '<span size="smaller" rise="-2000">')
           .replace('}', '</span>'))


figdefaults = {
    'omstyle': om.styles.ColorOnWhiteVector(),
    'pangofamily': 'Linux Biolinum O',
}


def add_dhr_top_axis(plot, data):
    dmjd0 = plot.bpainter.axis.min
    dmjd1 = plot.bpainter.axis.max
    dmjdmin = data.dmjd.min()
    dhr0 = 24. * (dmjd0 - dmjdmin)
    dhr1 = 24. * (dmjd1 - dmjdmin)

    plot.tpainter.axis = om.rect.LinearAxis(dhr0, dhr1)
    plot.tpainter.paintLabels = True
    plot.setSideLabel(plot.SIDE_TOP, 'Δt (hr)')


class KeyHighlightStamp(om.stamps.Circle):
    def __init__(self, box=False):
        super(KeyHighlightStamp, self).__init__(size=3, fill=True)
        self._box = box

    def _paintOne(self, ctxt, style, size):
        if self._box:
            om.stamps.symBox(ctxt, style, size, self.fill)
        else:
            om.stamps.symCircle(ctxt, style, size, self.fill)

        ctxt.set_source_rgb(0, 0, 0)
        om.stamps.symCircle(ctxt, style, 5, False)


class ManualStampKeyPainter(om.rect.GenericKeyPainter):
    def __init__(self, keytext, stamp, stampstyle=None):
        self.keytext = keytext
        self.stamp = stamp
        self.stampStyle = stampstyle
        stamp.setData('no data allowed for key-only stamp')

    def _getText(self):
        return self.keytext

    def _drawLine(self):
        return False

    def _drawStamp(self):
        return True

    def _drawRegion(self):
        return False

    def _applyStampStyle(self, style, ctxt):
        style.apply(ctxt, self.stampStyle)

    def _getStamp(self):
        return self.stamp


class BigArrow(om.rect.FieldPainter):
    needsDataStyle = False
    dsn = None
    style = {'color': 'foreground', 'linewidth': 2}
    length = 36
    headsize = 16
    direction = None

    def __init__(self, x, y, direction):
        super(BigArrow, self).__init__()
        self.x = float(x)
        self.y = float(y)
        self.direction = direction

    def getDataBounds(self):
        return self.x, self.x, self.y, self.y

    def getKeyPainter(self):
        return None

    def doPaint(self, ctxt, style):
        super(BigArrow, self).doPaint(ctxt, style)

        x = self.xform.mapX(self.x)
        y = self.xform.mapY(self.y)

        ctxt.save()
        style.apply(ctxt, self.style)
        om.stamps.arrow(ctxt, x, y, self.direction, self.length, self.headsize)
        ctxt.restore()


class XYText(om.rect.FieldPainter):
    """Paint text at an X/Y location on a plot.

    The precise position is set by the combination of the `x`, `y`, `hAnchor`,
    and `vAnchor` parameters. `x` and `y` specify a location on the plot.
    `hAnchor` and `vAnchor` specify where the text lands relative to the
    location. `hAnchor = 0` means that the left edge of the text is found at
    `x`; `vAnchor = 1` means that the bottom edge of the text is found at `y`;
    and so on. The default setting of `hAnchor = vAnchor = 0.5` means that the
    exact center of the text is positioned at `x, y`.

    """
    style = None
    color = (0, 0, 0)
    hAnchor = 0.5
    vAnchor = 0.5
    needsDataStyle = False
    dsn = None
    _ts = None

    def __init__(self, x, y, text, hAnchor=0.5, vAnchor=0.5):
        super(XYText, self).__init__()
        self.text = str(text)
        self.x = float(x)
        self.y = float(y)
        self.hAnchor = float(hAnchor)
        self.vAnchor = float(vAnchor)

    def getDataBounds(self):
        return self.x, self.x, self.y, self.y

    def getKeyPainter(self):
        return None

    def doPaint(self, ctxt, style):
        super(XYText, self).doPaint(ctxt, style)

        if self._ts is None:
            self._ts = om.TextStamper(self.text)
        x = self.xform.mapX(self.x)
        y = self.xform.mapY(self.y)
        w, h = self._ts.getSize(ctxt, style)

        ctxt.save()
        style.apply(ctxt, self.style)
        self._ts.paintAt(
            ctxt,
            x - self.hAnchor * w,
            y - self.vAnchor * h,
            self.color
        )
        ctxt.restore()


class TextOverlay(om.rect.FieldPainter):
    """Overlay text on the plot.

    """
    style = None
    color = (0, 0, 0)
    _ts = None

    def __init__(self, h, v, text):
        super(TextOverlay, self).__init__()
        self.text = str(text)
        self.h = float(h)
        self.v = float(v)

    def getDataBounds(self):
        return None, None, None, None

    def getKeyPainter(self):
        return None

    def doPaint(self, ctxt, style):
        super(TextOverlay, self).doPaint(ctxt, style)

        if self._ts is None:
            self._ts = om.TextStamper(self.text)

        w, h = self._ts.getSize(ctxt, style)

        ctxt.save()
        style.apply(ctxt, self.style)
        self._ts.paintAt(
            ctxt,
            self.h * (self.xform.width - w),
            self.v * (self.xform.height - h),
            self.color
        )
        ctxt.restore()


class Rectangle(om.rect.FieldPainter):
    """Plot a rectangle."""

    stroke = True
    fill = False
    style = 'bgLinework'

    def __init__(self, x0, x1, y0, y1, stroke=True, fill=False, style='bgLinework'):
        super(Rectangle, self).__init__()

        self.stroke = stroke
        self.fill = fill
        self.style = style

        if x0 > x1:
            x0, x1 = x1, x0
        self.xmin, self.xmax = x0, x1

        if y0 > y1:
            y0, y1 = y1, y0
        self.ymin, self.ymax = y0, y1

    def getDataBounds(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def getKeyPainter(self):
        return None

    def doPaint(self, ctxt, style):
        super(Rectangle, self).doPaint(ctxt, style)

        x0, x1 = self.xform.mapX(np.asarray([self.xmin, self.xmax]))
        y0, y1 = self.xform.mapY(np.asarray([self.ymin, self.ymax]))

        # We may not have x1 > x0, depending on the axis transform.
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        ctxt.save()
        style.apply(ctxt, self.style)
        ctxt.rectangle(x, y, w, h)
        if self.fill:
            ctxt.fill_preserve()
        if self.stroke:
            ctxt.stroke()
        ctxt.new_path() # clear path if we didn't stroke; restore() doesn't!
        ctxt.restore()


pltp_sep1, pltp_sep2 = 0.368, 0.515


def plot_long_term_photom(dsrc, dref, dpdm, plotinfo):
    add_unbinned = plotinfo.get('add_unbinned', True)
    add_refsrc = plotinfo.get('add_refsrc', True)

    from . import data
    ssrc = data.photom_bin_by_visits(dsrc, half_scans=True)
    if add_refsrc:
        sref = data.photom_bin_by_visits(dref, half_scans=True)

    print('# typical integration time:', np.median(ssrc.inttime))
    mjd0 = dsrc.mjd.min() - dsrc.dmjd.min()
    prd = dpdm.both_prd / 24.

    c0 = plotinfo.comp0
    c1 = plotinfo.comp1
    dsn0 = plotinfo.dsn_base

    if not plotinfo.has('group_sep1'):
        groups = [slice(None)]
    else:
        groups = [(ssrc.dmjd < plotinfo.group_sep1),
                 (ssrc.dmjd > plotinfo.group_sep1) &(ssrc.dmjd < plotinfo.group_sep2),
                 (ssrc.dmjd > plotinfo.group_sep2)]

    p = om.RectPlot()

    # Un-binned source data

    if add_unbinned:
        p.addXY(dsrc.dmjd, dsrc[c0], None,
                 lines=0, pointStamp=om.stamps.Circle(fill=1, size=2),
                 stampStyle={'color': (1, plotinfo.unavg_intens, plotinfo.unavg_intens)})

        p.addXY(dsrc.dmjd, dsrc[c1], None,
                 lines=0, pointStamp=om.stamps.Circle(fill=1, size=2),
                 stampStyle={'color': (plotinfo.unavg_intens, plotinfo.unavg_intens, 1)})

    # Horizontal zero lines(source and reference)

    p.addHLine(0, keyText=None, lineStyle={'color': (0,0,0), 'linewidth': 3})
    if add_refsrc:
        p.addHLine(plotinfo.ref_offset, keyText=None, lineStyle=plotinfo.ref_style)

    # Binned source and reference data

    kt1 = plotinfo.src_kt0
    kt2 = plotinfo.src_kt1

    if add_refsrc:
        kt3 = plotinfo.ref_kt0
        kt4 = plotinfo.ref_kt1
        sref['plot_'+c0] = sref[c0] + plotinfo.ref_offset
        sref['plot_'+c1] = sref[c1] + plotinfo.ref_offset

    for g in groups:
        p.addDF(ssrc[g][['dmjd', c0, 'u'+c0]], kt1,
                 lines=1, pointStamp=om.stamps.Circle(fill=1, size=3),
                 lineStyle={'linewidth': 2},
                 dsn=dsn0+0, zheight=5)

        p.addDF(ssrc[g][['dmjd', c1, 'u'+c1]], kt2,
                 lines=1, pointStamp=om.stamps.Circle(fill=1, size=3),
                 lineStyle={'linewidth': 2},
                 dsn=dsn0+1, zheight=5)

        if add_refsrc:
            p.addDF(sref[g][['dmjd', 'plot_'+c0, 'u'+c0]], kt3,
                     lines=1, pointStamp=om.stamps.Circle(fill=1, size=3),
                     lineStyle={'linewidth': 2},
                     dsn=2)

            p.addDF(sref[g][['dmjd', 'plot_'+c1, 'u'+c1]], kt4,
                     lines=1, pointStamp=om.stamps.Circle(fill=1, size=3),
                     lineStyle={'linewidth': 2},
                     dsn=3)

        kt1 = kt2 = kt3 = kt4 = None

    # Single-period PDM phasing, potentially with boxes identifying flare
    # zoom-ins.

    if plotinfo.has('zoom_ybounds'):
        for i in xrange(plotinfo.zoom_n_lines):
            t = (plotinfo.zoom_mjd0 - mjd0) + i * prd

            p.add(Rectangle(t - 0.5 * plotinfo.zoom_window_dur,
                              t + 0.5 * plotinfo.zoom_window_dur,
                              *plotinfo.zoom_ybounds,
                              style={'color': 'muted', 'dashing': [7,3]}),
                   zheight=-5, rebound=False)

            if plotinfo.zoom_desc is not None:
                p.add(XYText(t, plotinfo.zoom_ybounds[1],
                               '<big><b>%s%d</b></big>' %(plotinfo.zoom_desc, i + 1),
                               vAnchor=-0.03),
                       rebound=False)

    p.defaultKeyOverlay.hAlign = plotinfo.key_halign
    p.defaultKeyOverlay.vAlign = plotinfo.key_valign
    p.setBounds(*plotinfo.bounds)
    p.setYLabel('Flux density(μJy)')

    if plotinfo.label_bot:
        p.setXLabel('MJD - %.0f(day)' % mjd0)
    else:
        p.bpainter.paintLabels = False

    if plotinfo.add_dhr_top_axis:
        add_dhr_top_axis(p, dsrc)

    p.lpainter.everyNthMajor = 2
    return p


flareset_dt_split = 1.0 # minutes

def plot_flare_zoom(dphot, plotinfo):
    from . import data

    dur = plotinfo.dmjd1 - plotinfo.dmjd0
    w = (
        (dphot.dmjd > plotinfo.dmjd0 - 0.1 * dur) &
        (dphot.dmjd < plotinfo.dmjd1 + 0.1 * dur)
    )
    subset = dphot[w]

    p = om.RectPlot()
    med_uncerts = []
    max_val = None
    max_loc = None

    for icmp, cinfo in enumerate(plotinfo.components):
        kt1 = cinfo.key if plotinfo.showkey else None

        med_uncerts.append(subset['u'+cinfo.name].median())

        ml = subset[cinfo.name].argmax()
        if max_val is None or subset[cinfo.name][ml] > max_val:
            max_val = subset[cinfo.name][ml]
            max_loc = ml

        for mask in data.photom_generate_indices(subset):
            visit = subset.take(mask)
            venv = om.rect.VEnvelope(kt1)
            yhi = np.asarray(visit[cinfo.name])
            ylo = 0 * yhi
            w = np.where(ylo > yhi)
            tmp = ylo[w]
            ylo[w] = yhi[w]
            yhi[w] = tmp
            venv.setFloats(np.asarray(visit.dmjd), ylo, yhi)
            venv.style = {'color': cinfo.color}
            p.add(venv)
            kt1 = None

    if plotinfo.showkey:
        p.defaultKeyOverlay.hAlign = 0.05
        p.defaultKeyOverlay.vAlign = 0.06

    med_uncert = np.median(med_uncerts)
    p.addXYErr([plotinfo.uncert_loc[0]], [plotinfo.uncert_loc[1]], [med_uncert], None,
               lineStyle={'color': (0,0,0)}, stampStyle={'color': (0,0,0)})
    p.addHLine(0, None, zheight=-5, lineStyle=plotinfo.refline_style)
    p.setBounds(plotinfo.dmjd0, plotinfo.dmjd1, *plotinfo.ybounds)

    tp = om.TextPainter('<big><b>%s</b></big>' % plotinfo.desc)
    co = p.add(om.rect.AbsoluteFieldOverlay(tp), rebound=False)
    co.hAlign = 0.95
    co.vAlign = 0.05

    if plotinfo.paintleft:
        p.lpainter.everyNthMajor = 3
    else:
        p.lpainter.paintLabels = False

    for ap in p.bpainter, p.tpainter:
        ap.autoBumpThreshold = 0
        ap.minorTicks = 10
    p.bpainter.labelMinorTicks = True
    p.bpainter.everyNthMinor = 5

    for ap in p.lpainter, p.rpainter:
        ap.minorTicks = 2

    if plotinfo.labelleft:
        p.setYLabel('Flux density(μJy)')

    if plotinfo.labelbot:
        mjd0 = np.floor(dphot.mjd.min())
        p.setXLabel('MJD - %.0f(day)' % mjd0)

    # time away from peak top-axis label
    dmjdpeak = subset.dmjd[max_loc]
    ds0 = 86400 *(plotinfo.dmjd0 - dmjdpeak)
    ds1 = 86400 *(plotinfo.dmjd1 - dmjdpeak)
    p.tpainter.axis = om.rect.LinearAxis(ds0, ds1)
    p.tpainter.paintLabels = True
    p.tpainter.autoBumpThreshold = 10.
    p.tpainter.minorTicks = 4
    p.tpainter.everyNthMajor = 2
    p.setSideLabel(p.SIDE_TOP, pangosub('t – t{peak}(sec)'))

    return p


def gen_flare_set_plots(dphot, prd, plotinfo):
    nrows = plotinfo.nflares // plotinfo.ncols

    for iflare in xrange(plotinfo.nflares):
        iw = iflare % plotinfo.ncols
        ih = iflare // plotinfo.ncols
        yield plot_flare_zoom(
            dphot, plotinfo,
            plotinfo.pmjd0 + iflare * prd / 24.,
            iflare,
            showkey = (iflare == 0),
            paintleft = (iw > 0),
            labelleft = (iw == 0 and ih == 1),
            labelbot = (ih == nrows - 1)
        )
