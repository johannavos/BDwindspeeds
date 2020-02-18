import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from astropy.table import Table
import emcee
import corner
import IPython
from astropy.io import fits

hdulist = fits.open('2M1047_sin_flat_chain_ap3_bin5_epoch2.fits') #epoch 2

samples = hdulist['PRIMARY'].data
                    
samples[:, 0] = samples[:, 0] * 100.
fig = corner.corner(samples,  labels=["$\mathrm{Amp }~(\%)$", "$~~~~~~\mathrm{  Period}~\mathrm{(hr)}$", "$\mathrm{Phase}$",
	 "$\mathrm{Mean}$"], quantiles = [0.16, 0.5, 0.84], show_titles=True, label_kwargs={"fontsize":18},
	 title_kwargs={"fontsize":15}, title_fmt=".3f",plot_datapoints=False, top_ticks = False,color="teal",
	 quantiles_color="black",plot_contours=True,bins=100)
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis="x", direction="inout")
    ax.tick_params(axis="y", direction="inout")
    ax.tick_params(bottom=True, top=True, left=True, right=True)
fig.savefig("results/2M1047_tri_epoch2_pub.pdf",dpi=100 )
plt.clf()

