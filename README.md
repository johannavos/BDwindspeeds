# BDwindspeeds

This public repo contains data and code used in Allers et al. (2020):

# Data

**2M1047_epoch1_ap2.sav** -- Spitzer corrected photometry for 2M1047 epoch 1 data

**2M1047_epoch2_ap3.sav** -- Spitzer corrected photometry for 2M1047 epoch 2 data

**2M1122_epoch1_ap2.sav**	-- Spitzer corrected photometry for W1122 epoch 1 data

**2M1047_calibch2_ap2_epoch1.fits** -- Spitzer corrected photometry for 2M1047 epoch 1 data used in python codes

**2M1047_calibch2_ap3_epoch2.fits** -- Spitzer corrected photometry for 2M1047 epoch 2 data used in python codes

**target.phot.ll.txt** -- VLA monitoring data


# Code

**calc_FAP.pro**	      -- IDL code to calculate periodograms of target and reference stars, as well as false-alarm
                       probability using corrected Spitzer photometry. (Figure S2)  
                       
**plot_fig2_final.py**	-- Lomb-Scargle boot-strapping method -- create Figure 2 in paper using corrected Spitzer photometry   

**plot_figs5_final.py** -- Lomb-Scargle boot-strapping method -- create Figure S5 in paper using VLA data   

**2M1047_sin_emcee.py** -- find sinusoidal fit posterior distributions using MCMC.

**niceplot_2M1047_sin.py** -- create nice corner plot using MCMC results. (Fig 1, S3, modified for S4).


