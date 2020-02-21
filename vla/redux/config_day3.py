# a-priori cal parameters:
#have_antpos = True # default: False

# glue parameters (make sure these are correct before gluing!):
bpfdfield = '2'
bpfdimage = 'nrao/VLA/CalModels/3C286_C.im'
gnphfield = '0'
targfield = '1'
meanbp = 'meanbp-c3bit.npy'
# Fun times: what should be spw 63 seems to be missing. Just drop the whole
# baseband window since it ought to be redundant with 16-31 anyway.
gluemapping = '16-31,32-47,63-78'

# calibration parameters:
refant = 'ea13 ea21 ea26'.split ()
#do_polcal = True # default: False
#fluxstandard = 'Perley-Butler 2013' # default: 'Perley-Butler 2010'

peels = [
    # Peel source 0
    {
        'xslice': [580, 650],
        'yslice': [600, 690],
        'components': [
            {
                'flux': 0.00228,
                'fluxunit': 'Jy',
                'shape': 'point',
                'dir': 'J2000 10h48m31.431 +21d15m50.03',
            },
        ],
    },

    # Peel source (pair) 1
    {
        'xslice': [768, 817],
        'yslice': [928, 993],
        'components': [
            {
                'flux': 0.00278,
                'fluxunit': 'Jy',
                'shape': 'point',
                'dir': 'J2000 10h48m06.179 +21d25m56.91',
            },
            {
                'flux': 0.000661,
                'fluxunit': 'Jy',
                'shape': 'point',
                'dir': 'J2000 10h48m07.516 +21d25m58.87',
            },
        ],
    },
]

# imaging/analysis parameters:

default_images = ['final']

images = {
    'firstlook': {
        'niter': 50,
        'imsize': 2048,
    },

    'prepeel': {
        'niter': 1000,
        'nterms': 2,
    },

    'postcomp': {
        # for comparing to `prepeel_phased`
        'niter': 5000,
        'nterms': 2,
        'phase_source': 'target',
    },

    'tt2': {
        'niter': 5000,
        'nterms': 3,
    },

    'final': {
        'niter': 5000,
        'phase_source': 'target',
    },

    '_defaults': {
        'cellsize': 2.0, # arcsec
        'niter': 10,
        'imsize': 1801,
        'wprojplanes': 128,
        'nterms': 3,
        'stokes': 'i',
    },
}

sources = {
    'target': {
        'pos': '10:47:51.472 +21:24:10.51',
    },
}

ftsubs = {
    'target': {
        'image': 'final',
        'source': 'target',
    },
}
