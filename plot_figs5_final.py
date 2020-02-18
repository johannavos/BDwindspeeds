import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from astropy.io import fits
from astropy.stats import LombScargle
from astropy.table import Table
from sklearn.utils import resample

radio = Table.read('target.phot.ll.txt', format='ascii')
period_bs = fits.getdata('period_bs_fullsamp.fits')
period_mc = fits.getdata('period_mc_1e5samples.fits')

minfreq = 2.0
maxfreq = 20
samppeak = 100

ls = LombScargle(radio['mjd'], radio['re'])      

frequency, power = ls.autopower(minimum_frequency=minfreq, maximum_frequency=maxfreq, samples_per_peak=samppeak)
best_frequency = frequency[np.argmax(power)]
best_period = (1. / best_frequency)*24.
fap = ls.false_alarm_probability(power.max())

fig = plt.figure(figsize=(20, 10))
fig.tight_layout()

grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)

ax1 = fig.add_subplot(grid[ :, 0])
ax1.plot((1. / frequency)*24., power, label='peak period:'+'{:10.3f}'.format(best_period)+' hours', color='orange')
ax1.set_xlabel('period (hr)', fontsize=30)
ax1.set_ylabel('periodogram power', fontsize=30)
ax1.set_xlim((1, 6))
ax1.text(0.5, 0.037, 'A', size=40)

ax1.legend(fontsize=15, bbox_to_anchor=(0.3,1.005))    



ax1.tick_params(labelsize=20, labeltop=True, labelright=True, which="both", top=True, right=True)

ax1 = fig.add_subplot(grid[0,1])
ax1.hist(period_bs, bins=20, color='orange')
ax1.set_ylim((0, 1750))
ax1.set_xlim((1.750, 1.765))

ax1.plot([best_period, best_period], [0, 1750.], color='black')
ax1.plot([best_period + np.std(period_bs), best_period + np.std(period_bs)], [0, 1750.], color='black', linestyle=':', label='peak periodogram period')
ax1.plot([best_period - np.std(period_bs), best_period - np.std(period_bs)], [0, 1750.], color='black', linestyle=':', label='1-'+r'$\sigma$'+' period range')
ax1.legend()

ax1.set_title('boot-strapping distribution', fontsize=25)
ax1.set_xlabel('period (hr)', fontsize=20)
ax1.set_ylabel('n', fontsize=30)

ax1.tick_params(labelsize=20)
ax1.text(1.765, 1800, 'B', size=40)

ax1 = fig.add_subplot(grid[1,1])
ax1.hist(period_mc, bins=20, color='orange')
ax1.set_ylim((0, 175000))
ax1.set_xlim((1.750, 1.765))
ax1.plot([best_period, best_period], [0, 175000.], color='black', label='best period')
ax1.plot([best_period + np.std(period_mc), best_period + np.std(period_mc)], [0, 175000.], color='black', linestyle=':', label=r'1-$\sigma$')
ax1.plot([best_period - np.std(period_mc), best_period - np.std(period_mc)], [0, 175000.], color='black', linestyle=':')
ax1.set_xlabel('period (hr)', fontsize=20)
ax1.set_ylabel('n', fontsize=30)
ax1.set_title('Monte Carlo distribution', fontsize=25)

ax1.tick_params(labelsize=20)
ax1.text(1.765, 180000, 'C', size=40)

fig.savefig('FigS5_updated_v1.png')
