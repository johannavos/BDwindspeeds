#! /usr/bin/env python
# Copyright 2020 Peter Williams and collaborators
# Licensed under the General Public License version 3 (GPLv3).

from __future__ import absolute_import, division, print_function

import sys, numpy as np
from pwkit.io import Path
from pwkit.environments.casa import util, tasks
b = util.sanitize_unicode

util.logger('info')

# Config defaults

bpfdfield = None
bpfdimage = None
gnphfield = None
refant = None
targfield = None
images = {}
sources = {}
ftsubs = {}

gluemapping = '0-7,8-15'
meanbp = 'meanbp-c.npy' # set to None for no mean-bandpass division
have_antpos = False
do_polcal = False
fluxstandard = 'Perley-Butler 2010'
bp_minblperant = 2 # bug? sometimes need weirdly-small value here
bp_minsnr = 2.
g0cal_spw = '*:204~244' # used to roughly gpcal bp data before bp solve
gpstp_solint = 'int' # solution interval for short-term phase cal of GP cal
gpstp_interp = 'nearest'
gp_solint = '5min' # basic solution interval for gp cal source
peel1_solint = '1min'
peel2_solint = '5min'
peelbp_solint = 'inf,32ch'
plot_maxtimegap = 5 # minutes
average_timebin = 10. # average target data to this # of seconds after cal
default_images = ['firstlook']
default_sources = ['target']
default_ftsubs = ['target']
peels = []

# File names: not recommended to change

sdm = 'raw.sdm'
ungluedvis = 'unglued.ms'
bpvis = 'bp.ms'
gpvis = 'gp.ms'
targvis = 'targ.ms'
mainvises = [bpvis, gpvis, targvis]
peelrefstem = 'peelref'
peelvis = 'peel.ms'
peelmodelfmt = 'peelmodel_{idx}.cl'
peelworkfmt = 'peelwork_{idx}.ms'
tavgvis = 'tavg.ms'

antpos = 'ap.cal'
gc = 'gc.cal'
sp = 'sp.cal'
op = 'op.cal'
g0 = 'g0.cal'
dl = 'dl.cal'
bp = 'bp.cal'
g1 = 'g1.cal'
kc = 'kc.cal'
df = 'df.cal'
xf = 'xf.cal'
gpstp = 'gpstp.cal'
g2 = 'g2.cal'
fs = 'fs.cal'
gpeel1fmt = 'peel1_{idx}.cal'
gpeel2fmt = 'peel2_{idx}.cal'
peelbpfmt = 'peelbp_{idx}.cal'
peelsmbpfmt = 'peelsmbp_{idx}.cal'

# Now load the configuration. By default we just execfile 'config.py', but
# there is a mechanism for looking up individually-named files in another
# directory.

with Path('.rxpackage_ident').try_open(null_if_noexist=True) as ident_file:
    dirident = ident_file.readline().strip()

if not len(dirident): # standalone?
    datadir = Path('')
else: # centralized:
    dirident = '_' + dirident
    datadir = Path(__file__).readlink().parent

cfg_path = str(datadir / ('config%s.py' % dirident))
exec(compile(open(cfg_path).read(), cfg_path, 'exec'))

# Dynamically-generated docs, printed when run with no arguments. The idea is
# that reduction happens in "blocks" that end in some output that likely needs
# manual review.

utilities = '''
x_restore_import_flags
x_gen_meanbp
x_reset_bpmodel
x_applycal_bp
x_img_bp
x_applycal_gp
x_img_gp
'''.split()

blocks = []

block0 = 'sdm_to_ms plot_ants gen_gaincurve gen_swpow gen_antpos'.split()
if not have_antpos:
    block0[-1] += ':NA'
block0 += 'gen_opacity initweightspectrum applyapriori'.split()
blocks.append(block0)

block1 = 'glue save_flags_import manflag aoflag_bp setjy g0cal plot_g0'.split()
blocks.append(block1)

block2 = 'delays bandpass plot_bp'.split()
blocks.append(block2)

block3 = 'bpflag'.split()
if do_polcal:
    block3 += 'g1cal setjy_poln kcross leakage polpa'.split()
block3 += 'applycal_gp aoflag_gp gpstpcal g2cal fluxscale plot_fs'.split()
blocks.append(block3)

block4 = 'applycal_targ aoflag_targ average'.split()
if not len(sources.get('target', {}).get('pos', '')):
    block4 += 'image:firstlook'.split()
    blocks.append(block4)
else:
    blocks.append(block4)
    block5 =('setimage:final image pbcor iminfo imexport').split()
    blocks.append(block5)

pblock = 'make_peel_models fill_peel_model:N subsplit_peel_work:N fill_pwork_model:N peelcal1:N plotpeel1:N'.split()
blocks.append(pblock)
pblock = 'peelcal2:N plotpeel2:N'.split()
blocks.append(pblock)
pblock = 'peelbp:N plotpeelbp:N smooth_peelbp:N plotpeelsmbp:N'.split()
blocks.append(pblock)
pblock = 'finishpeel:N'.split()
blocks.append(pblock)
pblock = 'done_peeling'.split()
blocks.append(pblock)

if len(sources):
    block6 =('setsource:target fitsource spindex').split()
    blocks.append(block6)

if len(ftsubs):
    block7 =('setftsub:target make_ftsub_model ftsub photom:i dynspec:i').split()
    blocks.append(block7)

blocks_doc = '\n'.join(
    ['Utilities: ' + ' '.join(utilities), ''] +
    [' '.join(step) for step in blocks] +
    ['',
     'Images: ' + ' '.join(sorted(k for k in images.keys() if k[0] != '_')),
     'Sources: ' + ' '.join(sorted(sources.keys())),
     'FT-subs: ' + ' '.join(sorted(ftsubs.keys()))
     ]
)

# Calibration foo

cal1_gaintables = [dl, bp]
cal1_gainfields = ['', '']
cal1_interpolns = ['nearest', 'nearest']

cal2_gaintables = cal1_gaintables
cal2_gainfields = cal1_gainfields
cal2_interpolns = cal1_interpolns

if do_polcal:
    cal2_gaintables = cal1_gaintables + [kc, df, xf]
    cal2_gainfields = cal1_gainfields + ['', '', '']
    cal2_interpolns = cal1_interpolns + ['nearest', 'nearest', 'nearest']

mk_cal3_gaintables = lambda: cal2_gaintables + [fs] # for uniformity
mk_cal3_gainfields = lambda f: cal2_gainfields + [f]
mk_cal3_interpolns = lambda i: cal2_interpolns + [i]


# Reduction steps
# Phase 1: pre-glue information-gathering and a-priori calibrations

def sdm_to_ms():
    tasks.importevla(sdm, ungluedvis)

    with open('listobs.txt', 'w') as f:
        for line in tasks.listobs(ungluedvis):
            print(line, file=f)


def plot_ants():
    tasks.plotants(ungluedvis, 'antpos.pdf')


def gen_antpos(nastr=None):
    # 'nastr' is just so that one can copy/paste directly from the output of
    # running process.py without arguments when there's no antpos information
    # available. The ":NA" in the output is just a quick indicator for when
    # have_antpos has been set to False.

    if not have_antpos:
        print('skipping gen_antpos: no corrections available')
        return

    cfg = tasks.GencalConfig()
    cfg.vis = ungluedvis
    cfg.caltable = antpos
    cfg.caltype = 'antpos'
    tasks.gencal(cfg)


def gen_gaincurve():
    cfg = tasks.GencalConfig()
    cfg.vis = ungluedvis
    cfg.caltable = gc
    cfg.caltype = 'gceff'
    tasks.gencal(cfg)


def gen_swpow():
    """Note: the VLA pipeline uses 'rq' caltype, but the docs state pretty
    clearly that 'swpow' is better. CASA docs are often misleading, though ...

    """
    cfg = tasks.GencalConfig()
    cfg.vis = ungluedvis
    cfg.caltable = sp
    cfg.caltype = 'swpow'
    tasks.gencal(cfg)


def gen_opacity(nastr=None):
    # See gen_antpos re: nastr

    opacity = tasks.getopacities(ungluedvis, 'weather.png')
    print('opacities:', opacity)

    cfg = tasks.GencalConfig()
    cfg.vis = ungluedvis
    cfg.caltable = op
    cfg.caltype = 'opac'
    cfg.parameter = opacity
    cfg.spw = ','.join(str(i) for i in range(len(opacity)))
    tasks.gencal(cfg)


def initweightspectrum():
    """Convert the scalar WEIGHT column to WEIGHT_SPECTRUM so that we get
    correct weights for each sub-spw after gluing the raw-data spws into our
    wideband ones.

    """
    cb = util.tools.calibrater()
    cb.open(ungluedvis, compress=False, addcorr=False, addmodel=False)
    cb.initweights(wtmode=b'nyq', dowtsp=True)
    cb.close()


def applyapriori():
    """Apply the a-priori calibrations before gluing. Besides being
    convenient, opacity and swpow are per-spw in terms of the raw data spws,
    not our glued spws, so we need to apply those tables before gluing. (Or,
    at least, life is easier if we do.)

    """
    cfg = tasks.ApplycalConfig()
    cfg.vis = ungluedvis
    cfg.calwt = True
    cfg.gaintable = [gc, sp, op]
    cfg.gainfield = ['', '', '']
    cfg.interp = ['linear', 'nearest', 'nearest']

    if have_antpos:
        cfg.gaintable += [antpos]
        cfg.gainfield += ['']
        cfg.interp += ['nearest']

    tasks.applycal(cfg)


def glue():
    argv = ['rubbl-rxpackage', 'spwglue', '--mapping', 'correct']

    if meanbp is not None:
        argv += ['--meanbp', str(datadir / meanbp)]

    def translate(item):
        t1, t2 = list(map(int, item.split('-')))
        assert t2 >= t1, 'can\'t do reverse-order mappings'
        return ['-w', '%d-%d' % (t1, t2)]

    for item in gluemapping.split(','):
        argv += translate(item)

    argv += [
        '-f', str(bpfdfield), bpvis,
        '-f', str(gnphfield), gpvis,
        '-f', str(targfield), targvis,
        ungluedvis,
    ]

    shell(argv, shell=False)


def save_flags_import(*vises):
    if not len(vises):
        vises = mainvises

    for vis in vises:
        saveflags(vis, 'import')


# Phase 2: visibility flagging and calibration

def x_restore_import_flags(*vises):
    # The "x_" signifies that this is relevant in this general section of the pipeline,
    # but isn't part of the main data flow.

    if not len(vises):
        vises = mainvises

    for vis in vises:
        restoreflags(vis, 'import')


def manflag(*vises):
    if not len(vises):
        vises = mainvises

    casaflags = datadir / ('manual%s.cflags' % dirident)
    docasa = casaflags.exists() and(casaflags.stat().st_size > 0)
    arfflags = datadir / ('manual%s.flags' % dirident)
    doarf = arfflags.exists() and(arfflags.stat().st_size > 0)

    # Even if we have nothing to flag, we still save the flags under the name
    # "manual" for bpflag

    if docasa:
        cfg = tasks.FlaglistConfig()
        cfg.inpfile = str(casaflags)

    for vis in vises:
        if doarf:
            shell('arf flag \'%s\' %s' % (vis, arfflags))
        if docasa:
            cfg.vis = vis
            tasks.flaglist(cfg)
        saveflags(vis, 'manual')


def aoflag_bp():
    shell('aoflagger -strategy %s \'%s\'' % (datadir / 'vla-bumpybp-forreal.rfis', bpvis))
    saveflags(bpvis, 'ao')


def x_reset_bpmodel():
    tasks.delcal(bpvis)

    cb = util.tools.calibrater()
    cb.open(bpvis, addcorr=False, addmodel=False)
    cb.delmod(otf=True, scr=True)
    cb.close()


def setjy():
    """Note: if fluxstandard and/or modimage are not supported, this warns but
    doesn't raise an error!"""
    cfg = tasks.SetjyConfig()
    cfg.vis = bpvis
    cfg.field = bpfdfield # silence warning
    cfg.standard = fluxstandard
    cfg.modimage = bpfdimage
    tasks.setjy(cfg)


def g0cal():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = g0
    cfg.gaintype = 'G'
    cfg.calmode = 'ap'
    cfg.solint = 'int'
    cfg.refant = refant
    cfg.minsnr = 1
    cfg.spw = g0cal_spw
    tasks.gaincal(cfg)


def plot_g0():
    cfg = tasks.GpplotConfig()
    cfg.caltable = g0
    cfg.dest = 'g0.pdf'
    cfg.maxtimegap = plot_maxtimegap
    tasks.gpplot(cfg)


def delays():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = dl
    cfg.gaintype = 'K'
    cfg.combine = ['scan']
    cfg.solint = 'inf'
    cfg.refant = refant
    cfg.minsnr = 4
    cfg.gaintable = [g0]
    cfg.gainfield = ['']
    cfg.interp = ['nearest']
    tasks.gaincal(cfg)


def bandpass():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = bp
    cfg.gaintype = 'B'
    cfg.combine = ['scan']
    cfg.solint = 'inf'
    cfg.minblperant = bp_minblperant
    cfg.refant = refant
    cfg.minsnr = bp_minsnr
    cfg.solnorm = True
    cfg.gaintable = [g0, dl]
    cfg.gainfield = ['', '']
    cfg.interp = ['nearest', 'nearest']
    tasks.gaincal(cfg)


def plot_bp():
    cfg = tasks.BpplotConfig()
    cfg.caltable = bp
    cfg.dest = 'bp.pdf'
    tasks.bpplot(cfg)


def x_gen_meanbp(mode='median', minval=0.2):
    if meanbp is not None:
        print('**** WARNING data were glued with a meanbp, must concatenate transforms ****')

    tb = util.tools.table()
    tb.open(bp, nomodify=True)
    p = tb.getcol(b'CPARAM') # shape is [npol, nchan, nant*nspw]
    f = tb.getcol(b'FLAG') # same shape; True -> flagged/bad
    tb.close()

    p[np.where(f)] = 0.
    counts = f.sum(axis=(0,2))
    wbad =(counts == 0)

    if wbad.sum():
        # We could of course try to interpolate or something, but for now
        # this doesn't seem to come up in practice.
        raise Exception('there are %d channels without solutions' % wbad.sum())

    amps = np.abs(p)
    nchan = amps.shape[1]

    if mode == 'mean':
        # I'm not sure if I've done something wrong, but this simple mean
        # can lead to very spiky meanbp shapes that I'm not thrilled about.
        values = amps.sum(axis=(0,2)) / counts
    elif mode == 'median':
        values = np.median(np.transpose(amps, (1, 0, 2)).reshape((nchan, -1)), axis=1)
    else:
        raise Exception('unrecognized meanbp mode %r' % mode)

    values = np.maximum(values, minval) # ensure that no correction is crazy large
    values /= np.sqrt((values**2).mean()) # normalize to rms=1

    with open('new-meanbp.npy', 'wb') as f:
        np.save(f, values)


def bpflag(*vises):
    if not len(vises):
        vises = mainvises

    with open('bp.cflags', 'w') as f:
        tasks.extractbpflags(bp, f)

    cfg = tasks.FlaglistConfig()
    cfg.inpfile = 'bp.cflags'

    for vis in vises:
        cfg.vis = vis

        if vis != bpvis:
            # These haven't been aoflagged yet, so they're easy.
            print('+ applying', cfg.inpfile, 'to', cfg.vis)
            tasks.flaglist(cfg)
            saveflags(vis, 'bpman')
        else:
            # bp has been aoflagged, so we need to get fancy.
            restoreflags(vis, 'manual')
            print('+ applying', cfg.inpfile, 'to', cfg.vis)
            tasks.flaglist(cfg)
            saveflags(vis, 'bpman')
            restoreflags(vis, 'ao')
            print('+ applying', cfg.inpfile, 'to', cfg.vis)
            tasks.flaglist(cfg)
            saveflags(vis, 'bpao')


def g1cal():
    cfg = tasks.GaincalConfig()
    cfg.caltable = g1
    cfg.gaintype = 'G'
    cfg.calmode = 'ap'
    cfg.solnorm = False
    cfg.combine = ['scan']
    cfg.gaintable = cal1_gaintables
    cfg.gainfield = cal1_gainfields
    cfg.interp = cal1_interpolns
    cfg.parang = True
    cfg.refant = refant

    cfg.vis = bpvis
    cfg.solint = 'int'
    cfg.minsnr = 1e-2
    tasks.gaincal(cfg)


def setjy_poln():
    """We only do this after the bandpass and g1 cal since the polarized
    calibration can't take advantage of the detailed Stokes I model image."""

    tasks.delcal(bpvis) # del {MODEL,CORRECTED}_DATA columns
    tasks.delmod_cli(['delmod', bpvis], alter_logger=False) # del on-the-fly model info
    shell('casatask polmodel vis=\'%s\' field=\'%s\'' % (bpvis, bpfdfield))


def kcross():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = kc
    cfg.gaintype = 'KCROSS'
    cfg.refant = refant
    cfg.solint = 'inf'
    cfg.combine = ['scan']
    cfg.gaintable = cal1_gaintables + [g1]
    cfg.gainfield = cal1_gainfields + [bpfdfield]
    cfg.interp = cal1_interpolns + ['nearest']
    cfg.parang = True
    tasks.gaincal(cfg)


def leakage():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = df
    cfg.gaintype = 'Df+QU'
    cfg.calmode = 'ap'
    cfg.refant = refant
    cfg.solint = 'inf,2ch'
    cfg.combine = ['scan']
    cfg.preavg = 300
    cfg.minblperant = 12
    #cfg.minsnr = 5
    cfg.gaintable = cal1_gaintables + [g1, kc]
    cfg.gainfield = cal1_gainfields + [bpfdfield, '']
    cfg.interp = cal1_interpolns + ['nearest', 'nearest']
    cfg.parang = True
    tasks.gaincal(cfg)


def polpa():
    cfg = tasks.GaincalConfig()
    cfg.vis = bpvis
    cfg.caltable = xf
    cfg.gaintype = 'Xf'
    cfg.calmode = 'ap'
    cfg.refant = refant
    cfg.solint = 'inf'
    cfg.combine = ['scan']
    cfg.gaintable = cal1_gaintables + [g1, kc, df]
    cfg.gainfield = cal1_gainfields + [bpfdfield, '', '']
    cfg.interp = cal1_interpolns + ['nearest', 'nearest', 'nearest']
    cfg.parang = True
    tasks.gaincal(cfg)


def applycal_gp():
    cfg = tasks.ApplycalConfig()
    cfg.vis = gpvis
    cfg.calwt = True
    cfg.parang = True
    cfg.gaintable = cal2_gaintables
    cfg.gainfield = cal2_gainfields
    cfg.interp = cal2_interpolns
    tasks.applycal(cfg)


def aoflag_gp():
    shell('aoflagger -strategy %s -column CORRECTED_DATA \'%s\'' %
          (datadir / 'vla-flatbp-forreal.rfis', gpvis))
    saveflags(gpvis, 'ao')


def gpstpcal():
    # short-timescale phase cal of gp.ms
    cfg = tasks.GaincalConfig()
    cfg.caltable = gpstp
    cfg.gaintype = 'G'
    cfg.calmode = 'p'
    cfg.solnorm = True
    cfg.combine = ['scan']
    cfg.gaintable = cal2_gaintables
    cfg.gainfield = cal2_gainfields
    cfg.interp = cal2_interpolns
    cfg.parang = True
    cfg.refant = refant
    cfg.vis = gpvis
    cfg.solint = gpstp_solint
    cfg.minsnr = 3.
    tasks.gaincal(cfg)

    cfg = tasks.GpdetrendConfig()
    cfg.caltable = gpstp
    tasks.gpdetrend(cfg)


def g2cal():
    cfg = tasks.GaincalConfig()
    cfg.caltable = g2
    cfg.gaintype = 'G'
    cfg.calmode = 'ap'
    cfg.solnorm = False
    cfg.combine = ['scan']
    cfg.gaintable = cal2_gaintables
    cfg.gainfield = cal2_gainfields
    cfg.interp = cal2_interpolns
    cfg.parang = True
    cfg.refant = refant

    cfg.vis = bpvis
    cfg.solint = 'int'
    cfg.minsnr = 1e-2
    tasks.gaincal(cfg)

    cfg.append = True # important!
    set_cal_ms_name(g2, gpvis) # imporant workaround!
    cfg.vis = gpvis
    cfg.solint = gp_solint
    cfg.minsnr = 1e-2
    cfg.gaintable += [gpstp]
    cfg.gainfield += ['']
    cfg.interp += [gpstp_interp]
    tasks.gaincal(cfg)


def fluxscale():
    cfg = tasks.FluxscaleConfig()
    cfg.vis = gpvis
    cfg.caltable = g2
    cfg.fluxtable = fs
    cfg.reference = bpfdfield
    cfg.transfer = gnphfield
    cfg.listfile = 'secondaryflux.txt'
    tasks.fluxscale(cfg)
    shell('cat secondaryflux.txt') # lazy


def plot_fs():
    cfg = tasks.GpplotConfig()
    cfg.caltable = fs
    cfg.dest = 'fs.pdf'
    cfg.maxtimegap = plot_maxtimegap
    tasks.gpplot(cfg)


def x_applycal_bp():
    cfg = tasks.ApplycalConfig()
    cfg.vis = bpvis
    cfg.calwt = True
    cfg.parang = True
    cfg.gaintable = mk_cal3_gaintables()
    cfg.gainfield = mk_cal3_gainfields(bpfdfield)
    cfg.interp = mk_cal3_interpolns('nearest')
    tasks.applycal(cfg)


def x_wtf_bp():
    cfg = tasks.ApplycalConfig()
    cfg.vis = bpvis
    cfg.calwt = True
    cfg.parang = True
    cfg.gaintable = [dl, bp, g0]
    cfg.gainfield = ['', '', bpfdfield]
    cfg.interp = ['nearest', 'nearest', 'nearest']
    tasks.applycal(cfg)


def x_img_bp():
    _image(bpvis, 'xbpimg', no_config_ok=True)


def x_applycal_gp():
    cfg = tasks.ApplycalConfig()
    cfg.vis = gpvis
    cfg.calwt = True
    cfg.parang = True
    cfg.gaintable = mk_cal3_gaintables()
    cfg.gainfield = mk_cal3_gainfields(gnphfield)
    cfg.interp = mk_cal3_interpolns('nearest')
    tasks.applycal(cfg)


def x_img_gp():
    _image(gpvis, 'xgpimg', no_config_ok=True)


def applycal_targ():
    cfg = tasks.ApplycalConfig()
    cfg.vis = targvis
    cfg.calwt = True
    cfg.parang = True
    cfg.gaintable = mk_cal3_gaintables()
    cfg.gainfield = mk_cal3_gainfields(gnphfield)
    cfg.interp = mk_cal3_interpolns('linear')
    tasks.applycal(cfg)


def aoflag_targ():
    shell('aoflagger -strategy %s -column CORRECTED_DATA \'%s\'' %
          (datadir / 'vla-flatbp-forreal.rfis', targvis))
    saveflags(targvis, 'ao')

# Maybe add a 'statwt' invocation around here? My initial tests seem to show
# that its effect is imperceptible, though.

def average():
    cfg = tasks.SplitConfig()
    cfg.vis = targvis
    cfg.out = peelvis
    cfg.timebin = average_timebin
    cfg.col = 'corrected_data'
    cfg.antenna = '*&*' # discard autos
    # You can't average 2 ways in one step, because derp.
    # cfg.step = 2
    tasks.split(cfg)

# New stuff for peeling!!

def x_img_peel():
    _image(peelvis, 'xpeel')

def make_peel_models():
    for idx, peelsrc in enumerate(peels):
        cl = util.tools.componentlist()
        modelpath = peelmodelfmt.format(idx = idx)

        for comp in peelsrc['components']:
            cl.addcomponent(**comp)

        cl.rename(modelpath)
        cl.close(log=False)

def fill_peel_model(idx):
    from os.path import isdir

    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    imcfg = _img_config(peelvis, 'prepeel')
    ref_mod_paths = [_ttstem(imcfg, peelrefstem, i) for i in range(imcfg.nterms)]

    for p in ref_mod_paths:
        if not isdir(p):
            raise Exception(f'required reference CLEAN-component model image {p} not found')

    # First, create the temporary CLEAN component image(s) we'll use for the
    # baseline UV model. This consists of the reference image with this
    # source, and the prior peeled sources, zero'ed out. We zero out this
    # source to because we need to preserve it for calibration! We zero out
    # the prior sources since we'll augment the model with their post-peel
    # models, which will be more accurate than the pre-peel reference image.

    work_mod_paths = [_ttstem(imcfg, 'fillpeeltemp', i) for i in range(imcfg.nterms)]

    for i in range(imcfg.nterms):
        ref_path = ref_mod_paths[i]
        work_path = work_mod_paths[i]

        # Blow away the work and copy the reference
        shell('rm -rf \'%s\'' % work_path)
        shell('cp -r \'%s\' \'%s\'' % (ref_path, work_path))

        # Zero out the relevant slices
        for peelsrc in peels[:idx + 1]:
            x = peelsrc['xslice']
            y = peelsrc['yslice']
            _set_slice_rect(work_path, x[0], x[1], y[0], y[1], 0)

    # Now, FT this temporary image into peelvis:MODEL_DATA.

    cfg = tasks.FtConfig()
    cfg.vis = peelvis
    cfg.usescratch = True # important! silently fails otherwise
    cfg.model = work_mod_paths
    cfg.incremental = False
    cfg.wprojplanes = imcfg.wprojplanes
    tasks.ft(cfg)

    # Done with the temporary model images (note: might want to keep
    # these for debugging)

    for work_path in work_mod_paths:
        shell('rm -rf \'%s\'' % work_path)

    # Now, update the model data with the FTs from all of the peeled
    # sources that have been done up until this point

    for i2, peelsrc in enumerate(peels[:idx]):
        peelwork = peelworkfmt.format(idx=i2)
        argv = ['rubbl-rxpackage', 'peel', '--incremental', peelvis, peelwork]
        shell(argv, shell=False)

def subsplit_peel_work(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelwork = peelworkfmt.format(idx=idx)

    # uvsub subtracts MODEL_DATA from CORRECTED_DATA, defaulting
    # CORRECTED_DATA to DATA if it doesn't exist. The way we work, we punch
    # out individual sources when peeling, so modifying CORRECTED_DATA
    # iteratively doesn't work very well as a workflow. So, kill
    # CORRECTED_DATA to get it to reset to DATA.
    tb = util.tools.table()
    tb.open(peelvis, nomodify=False)
    if 'CORRECTED_DATA' in tb.colnames():
        tb.removecols(['CORRECTED_DATA'])
    tb.close()

    cfg = tasks.UvsubConfig()
    cfg.vis = peelvis
    cfg.reverse = False
    tasks.uvsub(cfg)

    cfg = tasks.SplitConfig()
    cfg.vis = peelvis
    cfg.out = peelwork
    cfg.col = 'corrected_data'
    tasks.split(cfg)

def fill_pwork_model(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelmodel = peelmodelfmt.format(idx=idx)
    peelwork = peelworkfmt.format(idx=idx)
    imcfg = _img_config(peelwork, 'prepeel')

    cfg = tasks.FtConfig()
    cfg.vis = peelwork
    cfg.usescratch = True # important! silently fails otherwise
    cfg.complist = peelmodel
    cfg.incremental = False
    cfg.wprojplanes = imcfg.wprojplanes
    tasks.ft(cfg)

def peelcal1(idx):
    "Shorter, phase-only cal for peel processing"
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelwork = peelworkfmt.format(idx=idx)
    gpeel1 = gpeel1fmt.format(idx=idx)

    cfg = tasks.GaincalConfig()
    cfg.caltable = gpeel1
    cfg.gaintype = 'T'
    cfg.calmode = 'p'
    cfg.solnorm = False
    cfg.combine = ['scan']
    cfg.gaintable = []
    cfg.gainfield = []
    cfg.interp = []
    cfg.parang = True
    cfg.refant = refant
    cfg.vis = peelwork
    cfg.solint = peel1_solint
    cfg.minsnr = 1e-2
    tasks.gaincal(cfg)

def plotpeel1(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    gpeel1 = gpeel1fmt.format(idx=idx)

    cfg = tasks.GpplotConfig()
    cfg.caltable = gpeel1
    cfg.dest = f'peel1_{idx}.pdf'
    cfg.maxtimegap = plot_maxtimegap
    cfg.phaseonly = True
    tasks.gpplot(cfg)

def peelcal2(idx):
    "Slower, amp-only cal for peel processing"
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelwork = peelworkfmt.format(idx=idx)
    gpeel1 = gpeel1fmt.format(idx=idx)
    gpeel2 = gpeel2fmt.format(idx=idx)

    cfg = tasks.GaincalConfig()
    cfg.caltable = gpeel2
    cfg.gaintype = 'T'
    cfg.calmode = 'a'
    cfg.solnorm = False
    cfg.combine = ['scan']
    cfg.gaintable = [gpeel1]
    cfg.gainfield = ['']
    cfg.interp = ['linear']
    cfg.parang = True
    cfg.refant = refant
    cfg.vis = peelwork
    cfg.solint = peel2_solint
    cfg.minsnr = 1e-2
    tasks.gaincal(cfg)

def plotpeel2(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    gpeel2 = gpeel2fmt.format(idx=idx)

    cfg = tasks.GpplotConfig()
    cfg.caltable = gpeel2
    cfg.dest = f'peel2_{idx}.pdf'
    cfg.maxtimegap = plot_maxtimegap
    tasks.gpplot(cfg)

def peelbp(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelwork = peelworkfmt.format(idx=idx)
    gpeel1 = gpeel1fmt.format(idx=idx)
    gpeel2 = gpeel2fmt.format(idx=idx)
    bppeel = peelbpfmt.format(idx=idx)

    cfg = tasks.GaincalConfig()
    cfg.vis = peelwork
    cfg.caltable = bppeel
    cfg.gaintype = 'B'
    cfg.combine = ['scan']
    cfg.solint = peelbp_solint
    cfg.minblperant = bp_minblperant
    cfg.refant = refant
    cfg.minsnr = bp_minsnr
    cfg.solnorm = False
    cfg.gaintable = [gpeel1, gpeel2]
    cfg.gainfield = ['', '']
    cfg.interp = ['linear', 'linear']
    tasks.gaincal(cfg)

def plotpeelbp(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    bppeel = peelbpfmt.format(idx=idx)
    bppdf = 'peelbp_{idx}.pdf'.format(idx=idx)

    cfg = tasks.BpplotConfig()
    cfg.caltable = bppeel
    cfg.dest = bppdf
    tasks.bpplot(cfg)

def smooth_peelbp(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    bppeel = peelbpfmt.format(idx=idx)
    smbppeel = peelsmbpfmt.format(idx=idx)

    # classy: stub out the full-rez smoothed bandpass solution
    # by copying our basic bandpass solution

    shell('rm -rf \'%s\'' % smbppeel)
    shell('cp -r \'%s\' \'%s\'' % (bp, smbppeel))

    # The magic incantations to do the smoothing and write it out

    model_data = BandpassData(bppeel)
    clobber_data = BandpassData(smbppeel)

    mdl0 = model_data.fit(0)
    mdl1 = model_data.fit(1)
    clobber_data.clobber_model(0, mdl0)
    clobber_data.clobber_model(1, mdl1) # spws 1 and 2 are the same
    clobber_data.clobber_model(2, mdl1)
    clobber_data.rewrite(smbppeel)

def plotpeelsmbp(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    smbppeel = peelsmbpfmt.format(idx=idx)
    smbppdf = 'peelsmbp_{idx}.pdf'.format(idx=idx)

    cfg = tasks.BpplotConfig()
    cfg.caltable = smbppeel
    cfg.dest = smbppdf
    tasks.bpplot(cfg)

def finishpeel(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)

    peelwork = peelworkfmt.format(idx=idx)
    gpeel1 = gpeel1fmt.format(idx=idx)
    gpeel2 = gpeel2fmt.format(idx=idx)
    smbppeel = peelsmbpfmt.format(idx=idx)

    cfg = tasks.ApplycalConfig()
    cfg.vis = peelwork
    cfg.parang = True
    cfg.gaintable = [gpeel1, gpeel2, smbppeel]
    cfg.gainfield = ['', '', '']
    cfg.interp = ['linear', 'linear', 'nearest']
    tasks.applycal(cfg)

def x_img_pwork(idx):
    idx = int(idx)
    assert idx >= 0 and idx < len(peels)
    peelwork = peelworkfmt.format(idx=idx)
    _image(peelwork, 'xpwork', no_config_ok=True, override_imbase=f'peelwork_{idx}.im/')

def done_peeling():
    """Rewrite peelvis:MODEL_DATA to contain only the fancy peeled visibilities,
    and *not* the pre-peel static model of the non-peeled sources. Then uvsub
    and split out, averaging pretty heavily in frequency space. So, if we
    image the resulting dataset, we'll see everything except the sources that
    have been peeled out.

    """
    from os.path import isdir

    for idx in range(len(peels)):
        peelwork = peelworkfmt.format(idx=idx)
        if not isdir(peelwork):
            raise Exception(f'missing peel-work MS {peelwork}')

    for idx in range(len(peels)):
        peelwork = peelworkfmt.format(idx=idx)

        argv = ['rubbl-rxpackage', 'peel']
        if idx > 0:
            argv += ['--incremental']
        argv += [peelvis, peelwork]

        shell(argv, shell=False)

    # uvsub subtracts MODEL_DATA from CORRECTED_DATA, defaulting
    # CORRECTED_DATA to DATA if it doesn't exist. The way we work, we punch
    # out individual sources when peeling, so modifying CORRECTED_DATA
    # iteratively doesn't work very well as a workflow. So, kill
    # CORRECTED_DATA to get it to reset to DATA.
    tb = util.tools.table()
    tb.open(peelvis, nomodify=False)
    if 'CORRECTED_DATA' in tb.colnames():
        tb.removecols(['CORRECTED_DATA'])
    tb.close()

    cfg = tasks.UvsubConfig()
    cfg.vis = peelvis
    cfg.reverse = False
    tasks.uvsub(cfg)

    # We use this opportunity to freq-average, since we've already
    # time-averaged. Might be better to reverse the ordering!
    cfg = tasks.SplitConfig()
    cfg.vis = peelvis
    cfg.out = tavgvis
    cfg.col = 'corrected_data'
    cfg.step = 16  # blah, lazy hardcoding
    tasks.split(cfg)


def x_img_peel():
    _image(peelvis, 'xpeelimg', no_config_ok=True)

# New stuff for selfcal!
#
# XXX I can't get selfcal to converge, I think there's just not enough flux in
# this field. Peeling out the bright sources contributes to that ...

def fill_selfcal_model():
    imcfg = _img_config(tavgvis, 'prepeel')

    cfg = tasks.FtConfig()
    cfg.vis = tavgvis
    cfg.usescratch = True # important! silently fails otherwise
    cfg.complist = '../repo/scmodel0.cl'
    cfg.incremental = False
    cfg.wprojplanes = imcfg.wprojplanes
    tasks.ft(cfg)

def selfcal1():
    "Shorter, T-type, phase-only selfcal"
    cfg = tasks.GaincalConfig()
    cfg.caltable = 'sc1.cal'
    cfg.gaintype = 'T'
    cfg.calmode = 'p'
    cfg.solnorm = False
    cfg.combine = ['spw']
    cfg.gaintable = []
    cfg.gainfield = []
    cfg.interp = []
    cfg.parang = False
    cfg.refant = 'ea24'  # XXX customized, preferred refants drop out
    cfg.vis = tavgvis
    cfg.solint = 'inf' # XXX '1min'
    cfg.minsnr = 3 # XXX
    cfg.minblperant = 6 # XXX
    tasks.gaincal(cfg)

def plotsc1():
    cfg = tasks.GpplotConfig()
    cfg.caltable = 'sc1.cal'
    cfg.dest = 'sc1.pdf'
    cfg.maxtimegap = 15 # plot_maxtimegap
    cfg.phaseonly = True
    tasks.gpplot(cfg)

def xdebug():
    _image('mdlcheck.ms', 'xpeel0', override_imbase='mdlcheck0.im/')
    _image('mdlcheck.ms', 'xpeel1', override_imbase='mdlcheck1.im/')

def applysc1():
    cfg = tasks.ApplycalConfig()
    cfg.vis = tavgvis
    cfg.calwt = False
    cfg.parang = False
    cfg.gaintable = ['sc1.cal']
    cfg.gainfield = ['']
    cfg.interp = ['nearest'] # XXX ['linear']
    cfg.spwmap = [[0, 0, 0]]
    tasks.applycal(cfg)

# Phase 3: imaging and image-based analysis

def setimage(*names):
    global default_images
    default_images = names


def setsource(*names):
    global default_sources
    default_sources = names


def setftsub(*names):
    global default_ftsubs
    default_ftsubs = names


def _source_pos_equatorial(srcname):
    from pwkit import astutil
    pos = sources[srcname]['pos']
    rastr, decstr = pos.split()
    return astutil.parsehours(rastr), astutil.parsedeglat(decstr)


def _source_pos_xy(srcname, imgcfg):
    from pwkit import astimage, astutil

    lon, lat = _source_pos_equatorial(srcname)
    imagepath = _ttstem(imgcfg, imgcfg.imbase + 'image', 0)
    y, x = astimage.CasaCImage(b(imagepath), 'r').simple().topixel([lat, lon])
    return int(round(x)), int(round(y))


def _img_config(vis, name, no_config_ok=False):
    info = dict(images['_defaults'])

    custom = images.get(name)
    if custom is None:
        if not no_config_ok:
            raise Exception('missing required image config %s' % name)
    else:
        for k, v in custom.items():
            info[k] = v

    cfg = tasks.MfscleanConfig()
    for k, v in info.items():
        setattr(cfg, k, v)

    cfg.vis = vis
    cfg.imbase = name + '.im/'
    cfg.cell = info['cellsize']
    cfg.stokes = info['stokes'].upper()
    cfg.imsize = [info['imsize'], info['imsize']]
    cfg.ftmachine = 'wproject'

    if 'phase_source' in info:
        from pwkit import astutil
        ra, dec = _source_pos_equatorial(info['phase_source'])
        cfg.phasecenter = 'J2000 ' + astutil.fmtradec(ra, dec, raseps='hm', decseps='dm')

    return cfg


def _image(vis, imagename, no_config_ok=False, override_imbase=None):
    cfg = _img_config(vis, imagename, no_config_ok=no_config_ok)

    if override_imbase is not None:
        assert override_imbase.endswith('.im/')
        cfg.imbase = override_imbase

    import os
    try:
        os.mkdir(cfg.imbase)
    except Exception as e:
        raise Exception('couldn\'t create directory "%s": %s' % (cfg.imbase, e))

    tasks.mfsclean(cfg)


def image(*names):
    if not len(names):
        names = default_images

    for name in names:
        _image(tavgvis, name)


def _ttstem(cfg, base, idx):
    if cfg.nterms == 1:
        return base
    return base + ('.tt%d' % idx)


def pbcor(*names):
    if not len(names):
        names = default_images

    ia = util.tools.image()

    for name in names:
        cfg = _img_config(tavgvis, name)
        pbimage = cfg.imbase + 'flux'

        for i in range(cfg.nterms):
            ia.open(_ttstem(cfg, cfg.imbase + 'image', i))
            corr = ia.pbcor(pbimage=pbimage, outfile=_ttstem(cfg, cfg.imbase + 'pbcor', i),
                             cutoff=0.05, overwrite=True)
            ia.close()
            corr.close()


def iminfo(*names):
    if not len(names):
        names = default_images

    for name in names:
        cfg = _img_config(tavgvis, name)
        shell('imtool info %s |tee %s.im/info.txt' %
              (_ttstem(cfg, cfg.imbase + 'pbcor', 0), name))


def imexport(*names):
    if not len(names):
        names = default_images

    for name in names:
        cfg = _img_config(tavgvis, name)
        fits = name + '.fits'
        tasks.image2fits(_ttstem(cfg, cfg.imbase + 'pbcor', 0), fits, overwrite=True)


def fitsource(srcname=None, imgname=None):
    if srcname is None:
        srcname = default_sources[0]
    if imgname is None:
        imgname = default_images[0]

    cfg = _img_config(tavgvis, imgname)
    x, y = _source_pos_xy(srcname, cfg)
    outfile = 'fitsrc.%s.%s.txt' % (srcname, imgname)
    imagepath = _ttstem(cfg, cfg.imbase + 'image', 0)

    point = sources[srcname].get('fit_force_point', False)
    if point:
        pstr = '-p '
    else:
        pstr = ''

    shell('imtool fitsrc %s%s %d %d |tee %s' % (pstr, imagepath, x, y, outfile))

    with open(outfile, 'at') as f:
        print('\n# in PB corrected image:\n', file=f)

    imagepath = _ttstem(cfg, cfg.imbase + 'pbcor', 0)
    shell('imtool fitsrc %s%s %d %d |tee -a %s' % (pstr, imagepath, x, y, outfile))


def _getpixel(impath, x, y):
    ia = util.tools.image()
    ia.open(impath)
    c = ia.getchunk([x, y], [x, y]) # args are(bot-left-corner, top-right-corner)
    ia.close()
    return c[0,0,0,0]


def spindex(srcname=None, imgname=None):
    if srcname is None:
        srcname = default_sources[0]
    if imgname is None:
        imgname = default_images[0]

    cfg = _img_config(tavgvis, imgname)
    if cfg.nterms < 2:
        print('spindex is a noop for', imgname)
        return

    x, y, = _source_pos_xy(srcname, cfg)
    outfile = 'spindex.%s.%s.txt' % (srcname, imgname)

    idx = _getpixel('%simage.alpha' % cfg.imbase, x, y)
    err = _getpixel('%simage.alpha.error' % cfg.imbase, x, y)

    with open(outfile, 'w') as f:
        print('# NOTE: CASA seems to underestimate errors in this analysis', file=f)
        print('spindex=%f' % idx, file=f)
        print('spi_error=%f' % err, file=f)


def _setrect(impath, x, y, halfwidth, value):
    blc = [x - halfwidth, y - halfwidth]
    trc = [x + halfwidth, y + halfwidth]

    ia = util.tools.image()
    ia.open(impath)
    c = ia.getchunk(blc, trc)
    c.fill(value)
    ia.putchunk(c, blc)
    ia.close()


def _set_slice_rect(impath, x0, x1, y0, y1, value):
    assert x1 > x0
    assert y1 > y0

    blc = [x0, y0]
    trc = [x1, y1]

    ia = util.tools.image()
    ia.open(impath)
    c = ia.getchunk(blc, trc)
    c.fill(value)
    ia.putchunk(c, blc)
    ia.close()


def make_ftsub_model(*names):
    """Note that the model we make for source X is a model of all of the
    emission *except* for source X, so that we can subtract everything else
    out and do DFT analysis on the visibilities assuming X is the only source
    of emission.

    """
    if not len(names):
        names = default_ftsubs

    for name in names:
        ftcfg = ftsubs[name]
        imcfg = _img_config(tavgvis, ftcfg['image'])
        x, y = _source_pos_xy(ftcfg['source'], imcfg)
        halfwidth = 6

        for i in range(imcfg.nterms):
            destpath = _ttstem(imcfg, name + '.ftmodel', i)
            src = _ttstem(imcfg, imcfg.imbase + 'model', i)
            shell('rm -rf \'%s\'' % destpath)
            shell('cp -r \'%s\' \'%s\'' % (src, destpath))
            _setrect(destpath, x, y, halfwidth, 0)


def ftsub(*names):
    if not len(names):
        names = default_ftsubs

    cb = util.tools.calibrater()
    vis = tavgvis

    for name in names:
        ftcfg = ftsubs[name]
        imcfg = _img_config(vis, ftcfg['image'])

        cfg = tasks.FtConfig()
        cfg.vis = vis
        cfg.usescratch = True # important! silently fails otherwise
        cfg.model = [_ttstem(imcfg, name + '.ftmodel', i) for i in range(imcfg.nterms)]
        cfg.incremental = False
        cfg.wprojplanes = imcfg.wprojplanes
        tasks.ft(cfg)

        cfg = tasks.UvsubConfig()
        cfg.vis = vis
        cfg.reverse = False
        tasks.uvsub(cfg)

        cfg = tasks.SplitConfig()
        for k, v in ftcfg.items():
            setattr(cfg, k, v)
        cfg.vis = vis
        cfg.out = name + '.sub.ms'
        cfg.col = 'corrected_data'
        cfg.antenna = '*&*' # discard autos
        tasks.split(cfg)

    # It is important to clear the CORRECTED_DATA column now, because otherwise
    # it will be used if/when we image again and everything except the source of
    # interest will disappear!

    import casadef
    if casadef.casa_version.startswith('5.'):
        cb.setvi(old=True, quiet=False) # as is evident, this is needed in CASA 5.x
    cb.open(vis, addcorr=False, addmodel=False)
    cb.initcalset(calset=True)
    cb.close()


def justsub(*names):
    """HACK redo sub?"""
    if not len(names):
        names = default_ftsubs

    cb = util.tools.calibrater()
    vis = tavgvis

    for name in names:
        ftcfg = ftsubs[name]
        imcfg = _img_config(vis, ftcfg['image'])

        cfg = tasks.FtConfig()
        cfg.vis = vis
        cfg.usescratch = True # important! silently fails otherwise
        cfg.model = [_ttstem(imcfg, name + '.ftmodel', i) for i in range(imcfg.nterms)]
        cfg.incremental = False
        cfg.wprojplanes = imcfg.wprojplanes
        tasks.ft(cfg)

        cfg = tasks.UvsubConfig()
        cfg.vis = vis
        cfg.reverse = False
        tasks.uvsub(cfg)

        cfg = tasks.SplitConfig()
        for k, v in ftcfg.items():
            setattr(cfg, k, v)
        cfg.vis = vis
        cfg.out = name + '.sub.ms'
        cfg.col = 'corrected_data'
        cfg.antenna = '*&*' # discard autos
        tasks.split(cfg)

    # It is important to clear the CORRECTED_DATA column now, because otherwise
    # it will be used if/when we image again and everything except the source of
    # interest will disappear!

    import casadef
    if casadef.casa_version.startswith('5.'):
        cb.setvi(old=True, quiet=False) # as is evident, this is needed in CASA 5.x
    cb.open(vis, addcorr=False, addmodel=False)
    cb.initcalset(calset=True)
    cb.close()


def photom(*args):
    spws = []
    names = []

    if not len(args):
        polarization = 'i'
    else:
        polarization = args[0]
        for arg in args[1:]:
            try:
                spw = int(arg)
            except ValueError:
                names.append(arg)
            else:
                spws.append(arg)

    if not len(names):
        names = default_ftsubs

    if polarization == 'i':
        polspec = 'RR,LL'
    else:
        polspec = polarization.upper()

    if not len(spws):
        fileident = polarization
    else:
        fileident = '%s.%s' % (polarization, '_'.join(spws))

    from pwkit.environments.casa.dftphotom import Config, dftphotom, PandasOutputFormat

    for name in names:
        ftcfg = ftsubs[name]
        ra, dec = _source_pos_equatorial(ftcfg['source'])

        with open('%s.phot.%s.txt' % (name, fileident), 'w') as f:
            cfg = Config()
            cfg.vis = name + '.sub.ms'
            cfg.rephase =(ra, dec)
            cfg.polarization = polspec
            if len(spws):
                cfg.spw = ','.join(str(spw) for spw in spws)
            cfg.outstream = f
            cfg.format = PandasOutputFormat()
            dftphotom(cfg)


def dynspec(*args):
    spws = []
    names = []

    if not len(args):
        polarization = 'i'
    else:
        polarization = args[0]
        for arg in args[1:]:
            try:
                spw = int(arg)
            except ValueError:
                names.append(arg)
            else:
                spws.append(arg)

    if not len(names):
        names = default_ftsubs

    if polarization == 'i':
        polspec = 'RR,LL'
    else:
        polspec = polarization.upper()

    if not len(spws):
        fileident = polarization
    else:
        fileident = '%s.%s' % (polarization, '_'.join(spws))

    from pwkit.environments.casa.dftdynspec import Config, dftdynspec

    for name in names:
        ftcfg = ftsubs[name]
        ra, dec = _source_pos_equatorial(ftcfg['source'])

        with open('%s.dynspec.%s.npy' % (name, fileident), 'wb') as f:
            cfg = Config()
            cfg.vis = name + '.sub.ms'
            cfg.rephase =(ra, dec)
            cfg.polarization = polspec
            if len(spws):
                cfg.spw = ','.join(str(spw) for spw in spws)
            cfg.outstream = f
            dftdynspec(cfg)


# Bandpass data structure for the peel bandpass smoothing operation.
#
# NOTE that some methods here hardcode all sorts of bad assumptions specific
# to the hadley dataset: that there are 3 spws, and that spws #1 and #2 are
# the same.

class BandpassData(object):
    antpols = None
    polnames = None
    seenspws = None
    vals = None
    flags = None
    freqs = None
    widths = None
    npol = None
    nchan = None
    nsoln = None

    def __init__(self, caltable):
        import os.path
        tb = util.tools.table()

        tb.open(caltable, nomodify=True)
        spws = tb.getcol('SPECTRAL_WINDOW_ID')
        ants = tb.getcol('ANTENNA1')
        vals = tb.getcol('CPARAM')
        flags = tb.getcol('FLAG')
        tb.close()

        tb.open(os.path.join(caltable, 'ANTENNA'), nomodify=True)
        names = tb.getcol('NAME')
        tb.close()

        tb.open(os.path.join(caltable, 'SPECTRAL_WINDOW'), nomodify=True)
        freqs = tb.getcol('CHAN_FREQ')  # center freqs of each channel, Hz, shape (nchan, nspw)
        widths = tb.getcol('CHAN_WIDTH')  # width of each channel, Hz, shape (nchan, nspw)
        tb.close()

        polnames = 'RL' # XXX: identification doesn't seem to be stored in cal table

        npol, nchan, nsoln = vals.shape

        # see what we've got

        antpols = {}
        seenspws = set()

        for ipol in range(npol):
            for isoln in range(nsoln):
                if not flags[ipol,:,isoln].all():
                    k = (ants[isoln], ipol)
                    byspw = antpols.get(k)
                    if byspw is None:
                        antpols[k] = byspw = []

                    byspw.append((spws[isoln], isoln))
                    seenspws.add(spws[isoln])

        seenspws = sorted(seenspws)

        # collect

        self.seenspws = seenspws
        self.antpols = antpols
        self.polnames = polnames
        self.vals = vals
        self.flags = flags
        self.freqs = freqs
        self.widths = widths
        self.npol = npol
        self.nchan = nchan
        self.nsoln = nsoln

    def plot(self, ispw, amp=True):
        import omega as om

        p = om.RectPlot()

        for iant, ipol in sorted(six.iterkeys(self.antpols)):
            for this_ispw, isoln in self.antpols[iant,ipol]:
                if this_ispw != ispw:
                    continue

                f = self.flags[ipol,:,isoln]
                w = np.where(~f)[0]

                if amp:
                    v = np.abs(self.vals[ipol,:,isoln])
                else:
                    v = np.angle(self.vals[ipol,:,isoln], deg=True)

                for s in numutil.slice_around_gaps(w, 1):
                    wsub = w[s]

                    if wsub.size == 0:
                        continue # Should never happen, but eh.

                    lines = (wsub.size > 1)
                    p.addXY(wsub, v[wsub], None, lines=lines, dsn=iant)

        return p

    def fit(self, spw):
        """Fit a smooth function to the median amplitude bandpass for the specified
        spectral window. Returns a function that evaluates the smoothed
        bandpass as a function of frequency, measured in Hz.

        """
        stack = []

        # XXX hardcoding!
        if spw == 0:
            allow_spws = (0,)
        elif spw == 1:
            # spws 1 and 2 are interchangeable in this dataset
            allow_spws = (1, 2)
        else:
            raise ValueError(f'illegal ispw {ispw}')

        for (iant, ipol), chunks in self.antpols.items():
             for ispw, isoln in chunks:
                 if ispw not in allow_spws:
                     continue
                 f = self.flags[ipol,:,isoln]
                 v = np.abs(self.vals[ipol,:,isoln])
                 v[f] = np.nan
                 stack.append(v)

        stack = np.nanmedian(stack, axis=0)
        ok = np.isfinite(stack)
        freqs = self.freqs[:,spw]

        # XXX hardcoding!
        # manual masking of edge channels with bad solutions:
        if spw == 0:
            ok[:2] = False
        elif spw == 1:
            ok[-2:] = False

        from pwkit import lsqmdl
        max_exponent = 3
        soln = lsqmdl.PolynomialModel(max_exponent, freqs[ok], stack[ok]).solve()
        return soln.mfunc

    def clobber_model(self, spw, fitfunc):
        bychan = fitfunc(self.freqs[:,spw])

        for (iant, ipol), chunks in self.antpols.items():
            for ispw, isoln in chunks:
                if ispw == spw:
                    self.flags[ipol,:,isoln] = False
                    self.vals[ipol,:,isoln] = bychan

    def rewrite(self, caltable):
        tb = util.tools.table()

        tb.open(caltable, nomodify=False)
        tb.putcol('CPARAM', self.vals)
        tb.putcol('FLAG', self.flags)
        tb.close()


# Infrastructure

def shell(command, shell=True, **kwargs):
    """Pretty much like check_call, but we print out the command that we're
    running, and default to using the shell."""

    import signal, subprocess
    print('+', command, file=sys.stderr)
    retcode = subprocess.call(command, shell=shell, **kwargs)
    if retcode > 0:
        raise RuntimeError('Command "%s" failed with exit code %d' % (command, retcode))
    elif retcode == -signal.SIGINT:
        raise KeyboardInterrupt()
    elif retcode < 0:
        raise RuntimeError('Command "%s" killed by signal %d' % (command, -retcode))


def saveflags(vis, versionname):
    print('+ saving current flags in %s as "%s"' % (vis, versionname))
    tasks.flagmanager_cli(['flagmanager', 'save', vis, versionname], alter_logger=False)


def restoreflags(vis, versionname):
    print('+ restoring saved "%s" flags in %s' % (versionname, vis))
    tasks.flagmanager_cli(['flagmanager', 'restore', vis, versionname], alter_logger=False)


def set_cal_ms_name(calpath, vispath):
    """Work around the(ill-conceived) restriction that cal tables should only
    be applied to MSes with the same name as they came from."""

    tb = util.tools.table()
    tb.open(calpath, nomodify=False)
    tb.putkeyword('MSName', vispath)
    tb.close()


if __name__ == '__main__':
    import sys, time
    from pwkit import cli

    cli.propagate_sigint()
    cli.unicode_stdio()

    if len(sys.argv) == 1:
        print(blocks_doc)
        sys.exit(0)

    g = globals()

    for name in sys.argv[1:]:
        name = name.split(':', 1)[0]
        if name not in g:
            print('error: no step "%s"' % name, file=sys.stderr)
            sys.exit(1)

    for name in sys.argv[1:]:
        pieces = name.split(':', 1)
        name = pieces[0]
        if len(pieces) > 1:
            args = pieces[1].split(',')
        else:
            args = []

        tstart = time.time()
        g[name](*args)
        tstop = time.time()
        print('+ step "%s" duration: %8.2f m' % (name, (tstop - tstart) * 1./60))
