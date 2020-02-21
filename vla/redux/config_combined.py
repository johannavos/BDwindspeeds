# For imaging the combined data, where we're not doing any calibration.

default_images = ['final']

images = {
    'n5000': {
        'niter': 5000,
    },

    'n5000lo': {
        'niter': 5000,
        'spw': '0',
    },

    'n5000hi': {
        'niter': 5000,
        'spw': '1,2,3',
    },

    'n10000': {
        'niter': 10000,
    },

    'tt2': {
        'niter': 5000,
        'nterms': 3,
    },

    'phased': {
        'niter': 3000,
        'phase_source': 'target',
    },

    '_defaults': {
        'cellsize': 2.0, # arcsec
        'niter': 10,
        'imsize': 1801,
        'wprojplanes': 128,
        'nterms': 2,
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
        'image': 'phased',
        'source': 'target',
    },
}
