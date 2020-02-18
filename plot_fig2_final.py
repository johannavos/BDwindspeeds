import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import LombScargle
from astropy.table import Table

minfreq = 2.0
maxfreq = 20
samppeak = 100
figname = 'Fig2_updated_v2.png'

# open file


data = Table.read('2M1047_calibch2_ap3_epoch2.fits',hdu=2)
jd2 = data['MJD']
flux2 = data['FLUX'] / np.nanmedian(data['FLUX'])


mask =  (flux2 > 0.0) & (flux2 < 1.05)


jd2 = jd2[mask]
flux2 = flux2[mask]


ls2 = LombScargle(jd2, flux2)
frequency2, power2 = ls2.autopower(minimum_frequency=minfreq, maximum_frequency=maxfreq, samples_per_peak=samppeak)
best_frequency2 = frequency2[np.argmax(power2)]
best_period = (1. / best_frequency2)*24.
fap2 = ls2.false_alarm_probability(power2.max())

# plot results
fig = plt.figure(figsize=(20, 10))
fig.tight_layout()

grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)

ax2 = fig.add_subplot(grid[ :, 0])
ax2.tick_params(labelsize=20, labeltop=True, labelright=True, which="both", top=True, right=True)
ax2.plot((1. / frequency2)*24., power2, color='teal', label='peak period: '+'{:10.3f}'.format(best_period)+' hours')

ax2.set_xlabel('period (hours)', fontsize=30)
ax2.set_ylabel('periodogram power', fontsize=30)

ax2.set_xlim((1, 6))
ax2.text(0.5, 0.16, 'A', size=40)

ax2.legend(fontsize=15, bbox_to_anchor=(0.3,1.005))    



ax1 = fig.add_subplot(grid[0,1])
period_jk = fits.getdata('period_bs_fullsamp_IR_final_1e5samples.fits')
nsamp = len(jd2)

ax1.hist(period_jk, bins=10, color='teal')
ax1.set_ylim((0, 40000))
ax1.plot([best_period, best_period], [-10, 40000.], color='black', label='peak periodogram period')
ax1.plot([best_period + np.std(period_jk), best_period + np.std(period_jk)], [-10, 40000.], color='black', linestyle=':', label='1-'+r'$\sigma$'+' period range')
ax1.plot([best_period - np.std(period_jk), best_period - np.std(period_jk)], [-10, 40000.], color='black', linestyle=':')
ax1.set_xlabel('period (hr)', fontsize=20)
ax1.set_ylabel('n', fontsize=30)
ax1.legend()
plt.gcf().subplots_adjust(bottom=0.15)
ax1.text(1.77, 41000, 'B', size=40)
ax1.set_title('boot-strapping distribution', fontsize=25)

ax1.tick_params(labelsize=20)

period_jk = fits.getdata('period_mc_IR_1e5samples_finalreduction.fits')
nsamp = 100000

ax1 = fig.add_subplot(grid[1,1])
ax1.hist(period_jk, bins=10, color='teal')
ax1.set_ylim((0, nsamp))
ax1.plot([best_period, best_period], [-10, nsamp], color='black')
ax1.plot([best_period + np.std(period_jk), best_period + np.std(period_jk)], [-10, nsamp], color='black', linestyle=':')
ax1.plot([best_period - np.std(period_jk), best_period - np.std(period_jk)], [-10, nsamp], color='black', linestyle=':')
ax1.set_xlabel('period (hr)', fontsize=20)
ax1.set_ylim((0, 40000))
ax1.set_ylabel('n', fontsize=30)
plt.gcf().subplots_adjust(bottom=0.15)
ax1.text(1.77, 41000, 'C', size=40)
ax1.set_title('Monte Carlo distribution', fontsize=25)

ax1.tick_params(labelsize=20)

fig.savefig(figname)
