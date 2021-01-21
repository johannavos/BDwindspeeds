import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from astropy.table import Table
import emcee
import corner
import scipy.optimize as op
import IPython
from astropy.io import fits

# read in data
data = Table.read('2M1047_calibch2_ap3_epoch2.fits',hdu=2)

### JV edits for new .fits file:
mjd = data['MJD'][0]
mjd = mjd - mjd[0]
t = mjd*24.
flux = data['FLUX'][0]
ap=3
binsize=1
###

fluxerr = np.zeros_like(flux) + (np.std(flux - np.roll(flux, 1)))/np.sqrt(2)


# try least squares fit first
guess_mean = np.mean(flux)
guess_std = 3*np.std(flux)/(2**0.5)
guess_phase = 0
guess_period = 2.0

# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(((t)/x[2])*2.*np.pi + x[1]) + x[3] - flux
est_std, est_phase, est_period, est_mean = op.leastsq(optimize_func, [guess_std, guess_phase, guess_period, guess_mean])[0]

print ('least sq. fits', est_std, est_phase, est_period, est_mean)


# set up MCMC

# parameters are: amplitude, period, phase, and offset

# define likelihood
def lnlike(theta, t, flux, fluxerr):
    amp, period, phase, offset = theta
    model = amp * np.sin((t/period)*2.*np.pi + phase) + offset
    inv_sigma2 = 1.0/(fluxerr**2) # + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((flux-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# maximize likelihood to determine first guess
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [est_std, est_period, est_phase, est_mean], args=(t, flux, fluxerr))
amp_ml, period_ml, phase_ml, offset_ml = result["x"]

# define priors
def lnprior(theta):
    amp, period, phase, offset = theta
    if 0 < amp < 1.0 and 0.0 < period < 30.0 and 0 < phase < 2.*np.pi and 0 < offset < 2.0:
        return 0.0
    return -np.inf

# define posterior
def lnprob(theta, t, flux, fluxerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, t, flux, fluxerr)

#initialize MCMC
ndim, nwalkers = 4, 1000
#pos = [ [abs(est_std), est_period, est_phase, est_mean] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos = [ [0.01, 1.7, np.pi, 1.0] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

#IPython.embed()

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, fluxerr))

print ("running MCMC")
# run the MCMC for 500 steps starting from the tiny ball defined above:
sampler.run_mcmc(pos, 8000)


samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
samples1 = samples

samples[:, 0] = samples[:, 0] * 100.
fig = corner.corner(samples,  labels=["$\mathrm{amplitude}$", "$\mathrm{period (hr)}$", "$\mathrm{phase}$", "$\mathrm{mean}$"], quantiles = [0.16, 0.5, 0.84], show_titles=True, labels_args={"fontsize":40}, title_fmt=".3f",plot_datapoints=False)
fig.savefig("results/2M0147_period_triangle_ap%d_bin%d.png" % (ap, binsize))
plt.clf()



amp = np.median(samples[:,0])
period = np.median(samples[:,1])
phase = np.median(samples[:,2])
offset = np.median(samples[:,3])
mean=offset




fits.writeto('results/2M1047_sin_sampler_ap%d.fits' % (ap), sampler.chain,overwrite=True)

fits.writeto('results/2M1047_sin_flat_chain_ap%d.fits' % (ap), samples,overwrite=True)
