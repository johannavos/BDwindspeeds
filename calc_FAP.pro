;;; IDL procedure to calculate false-alarm probability and periodograms of 
;;; target and reference stars from corrected Spitzer photometry.
;;; Written by Johanna Vos Jan 2020


FUNCTION RandPerm, numberOfElements, SEED=seed
	x = Lindgen( numberOfElements)
	return, x[Sort(Randomu(seed,numberOfElements))]
END



PRO calc_FAP,dir

!p.font = 0
plotsym, 0, 1, /fill

targ_name= '2MASS J1047+21'

restore, '2M1047_epoch1_ap2_new.sav'
actual_t_ch2 =t
corrected_flux_ch2 = reform(corrected_flux[ap,*,*])
bint_ch2  = reform(bint[0,*])
binflux_ch2 = binflux
resultsin2 = resultsin

actual_t_ch2 = actual_t_ch2 - actual_t_ch2[0]
bint_ch2 = bint_ch2 + actual_t_ch2[0]
full_t = actual_t_ch2


indtarg = 0
	
		
	refs = findgen( n_elements(corrected_flux_ch2[*,0]) )
	rms1 = fltarr( n_elements(corrected_flux_ch2[*,0] ) )
	bad_refs = []

	;;throw out reference stars with nans
	FOR l = 1 , n_elements(corrected_flux_ch2[*,0]) - 1 DO BEGIN
		rms1[l] = robust_sigma( corrected_flux_ch2[l,*] - shift(corrected_flux_ch2[l,*]-1,1) )
		nans = where (corrected_flux_ch2[l,*] NE corrected_flux_ch2[l,*])
		if nans[0] gt 0 then bad_refs = [bad_refs,l]
	ENDFOR
	if n_elements(bad_refs) GE 1 then remove, bad_refs, refs
	corrected_flux_ch2 = corrected_flux_ch2[refs,*]
	refs = findgen(n_elements(corrected_flux_ch2[*,0]) )


	remove, [indtarg,5], refs
	targfcal = corrected_flux_ch2[indtarg,*]
	ref_flux = corrected_flux_ch2[ refs, * ]	
	perp = 1000
	perm = 1000
	refpower = fltarr(n_elements(refs))
	powerref = fltarr(n_elements(refs),2,perp)

	
	;; go through each ref curve
	peakpower = fltarr( n_elements(refs), perm )
	FOR j = 0, n_elements(refs) - 1 DO BEGIN
		nf = n_elements( bint[j,*] )
		powerref[j,*,*]=periodogram(actual_t_ch2[*],ref_flux[j,*],per=[0.25,2*max(actual_t_ch2)],npts=perp)
		;plot,power(0,*),power(1,*)
		print, max( powerref[j,1,*] )
		refpower[j] = max( powerref[j,1,*] )
		

		;; randomly permutate reference star lightcurves and calculate highest peak
		FOR k = 0, n_elements(peakpower[0,*])-1 DO BEGIN
			x = RandPerm( nf )
			newcurve = ref_flux[ j, x ]
			noise = robust_sigma(newcurve, goodvec=q)
			newcurve = newcurve[q]
			tcurve = actual_t_ch2[q]
			power=periodogram(tcurve,newcurve,per=[0.1,2*max(tcurve)],npts=500)
			;plot,power(0,*),power(1,*)
			print, max( power[1,*] )
			peakpower[j,k] = max(power[1,*] )
			
		ENDFOR
	ENDFOR
	peakpower = peakpower[*]
	sorted = sort( peakpower )
	peakpower = peakpower ( sorted )
	; index of 95 threshold
	t_index95 = size(peakpower, /n_elements) * 0.95 
	t_index99 = size(peakpower, /n_elements) * 0.99
	;value of 95 thrsehold
	t_v95 = peakpower( t_index95 )
	t_v99 = peakpower( t_index99 )
	
	mod2 =  resultsin2[0]*(sin (((full_t-resultsin2[1]) / resultsin2[2])  *2.*!DPI)) + resultsin2[3]
	
		set_plot,'ps'
	device, filename = 'plots/periodogram_2M1047_ch2.eps', /encapsulate, /color, bits_per_pixel = 24,XSIZE=7, YSIZE=6, /INCHES
	;plot lightcurve
	loadct,5
	!p.multi = [0,1,2]
	plot,bint_ch2[*],binflux_ch2[0,*],psym=3,xtitle='Elapsed Time (hr)',ytitle='Relative Flux',$
		yrange=[0.97,1.03],xrange=[min(bint_ch2[0,*]),max(bint_ch2[0,*])],/xstyle,/ystyle,symsize=0.5,charsize=1.3
  	oplot,t,fltarr(n_elements(t))+1,thick=2 ,linestyle=2
  	plotsym, 0, 1
	oplot,bint_ch2,binflux_ch2[0,*],psym=8, color=cgcolor('black'),symsize=0.7
	plotsym, 0, 1, /fill 
	oplot,bint_ch2,binflux_ch2[0,*],psym=8, color=cgcolor('tg6'),symsize=0.7
    xyouts,max(full_t)/20., 1.022, targ_name
    xyouts,max(full_t)/20., 1.015, '2017-04-07'
    xyouts,max((full_t)*17)/20., 1.022, textoidl('4.5 \mum')



	;; periodogram of target
	power=periodogram(actual_t_ch2,corrected_flux_ch2[0,*],per=[0.1,2*max(actual_t_ch2)],npts=5000)
	plot,findgen( max(bint[0,*])*2 ),fltarr(3000) + t_v99	,linestyle =2,xrange =[0, max(bint[0,*]) * 2.],$
		/xstyle,yrange=[0,60],ytitle='Power',xtitle='Period (hr)',charsize=1.3
	oplot,power(0,*),power(1,*),thick=3
	loadct,0
	FOR z=0, n_elements(refs) -1 DO BEGIN
		oplot, powerref[z,0,*],powerref[z,1,*],color=80
	ENDFOR
	oplot,findgen( max(bint[0,*])*5 ),fltarr(3000) + t_v99	,linestyle =2 , color=cgcolor('tg7'),thick=5
	oplot,findgen( max(bint[0,*])*5 ),fltarr(3000) + t_v95	,linestyle =2 , color=cgcolor('tg5'),thick=5

		!p.multi=0	
	device,/close
	set_plot,'x'


t99_ch2 = t_v99
t95_ch2 = t_v95
power_ch2 = power
powerref_ch2 = powerref



save, filename = 'periodogram_results_2M1047+12.sav', fcal, cal, peakvals, t, airmass, dimm_fwhm, indtarg, xpos, ypos,ap,indtarg, binfcal,bint,err,power,t_v99,t_v95,refpower,targfcal,binfcal,refpower,resultsin,trf,power1,power2,refs,powerref,corrected_flux,rms_targ
stop
plot_panels
set_plot,'x'
print,'********************************'
print,'Target Peak Power:'
print, max(power[1,*])
print,'********************************'
print,'Reference Star Peak Powers:'
print,refpower
print,'********************************'
stop

END