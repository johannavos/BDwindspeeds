# Very Large Array analysis

VLA proposal ID: **18A-427**. Approved for three consecutive observing blocks
on 2M 1047+21 of 11 hr each, LST range 05:04-16:32, C band, scheduling
priority A, any configuration. Basically, sit and stare.

The file `obs.sb` gives the Scheduling Block (SB) specification used to
define the observations. Observations happened on MJDs 58403--58405. Referred
to as "day{1,2,3}" in subsequent analysis.


## Data availability

The raw data are available from the NRAO Science Data Archive under the
proposal ID given above. The script `redux/populate.sh` will rename the raw
data files into the form expected by the reduction scripts.

The final result of the raw reduction process is to extract
`target.dynspec.ll.npy`. Other data files and images are useful for sanity
checking, etc., but in the end that's the only input to the next analysis
stage. This file is available from [Zenodo](https://zenodo.org/) as DOI
[10.5281/zenodo.3678623](https://doi.org/10.5281/zenodo.3678623) (55 MiB).

The script `./dynspec-to-ts.py` does some final RFI filtering on the dynamic
spectrum to obtain a timeseries file that is used for all subsequent analysis.
That timeseries file is version-controlled in this repository as
`../target.phot.ll.txt`.


## Raw data reduction

The raw data reduction assets are in `redux/`, driven by a script named
`process.py` that uses a Python framework based on [CASA] and [pwkit].
Analysis was generally standard except for the use of the peeling tool
described in Williams et al.
([2019; 10.3847/2515-5172/ab35d5](http://doi.org/10.3847/2515-5172/ab35d5)).
If you use the reduction scripts for your own work, you should credit Peter K.
G. Williams appropriately.

[CASA]: https://casa.nrao.edu/
[pwkit]: https://github.com/pkgw/pwkit

I tried hard to self-cal the data, but eventually gave up. I could get spw0 to
show self-cal solutions that didn't look unreasonable (because it has more
flux than spw1) but nothing that ever actually improved my image. One day. To
support this work, I explored a lot about how to try to properly model the
spectra of all the sources in the field, because it turns out that MFS with 1
Taylor term is not enough for some sources. But other sources are too faint to
do any more. Tried a lot with pyBDSF, which is not easy to integrate into a
workflow. Never got anything super satisfactory.


## Timeseries extraction

Conducted in this directory with `target.dynspec.ll.npy` as an input, using
the script `./dynspec-to-ts.py`. The output file is `target.phot.ll.txt`,
which is also mirrored in this repository in the parent directory.


## Periodicity analysis

Conducted in this directory with `target.phot.ll.txt` as an input. As with
the reduction, lots of exploration and false starts leading up to the fairly
simple analysis described in the paper.

The Python scripts use [Astropy](https://www.astropy.org/),
[Numpy](https://numpy.org/), [SciPy](https://scipy.org/),
[Pandas](https://pandas.pydata.org/), [pwkit], and
[omegaplot](https://github.com/pkgw/omegaplot) for plotting.

Then `./periodicity-analysis.py` executes the final preferred analysis, and
`./plot-*.py` make the plots based on it. The file `photom.py` hardcodes the
"best" periodicity fit for plotting convenience.

Plotting commands:

```
$ ./plot-alldata.py margin=4 dims=1200,1200 omstyle=ColorOnWhiteVector out=vla_alldata.pdf
$ ./plot-phasings.py margin=4 dims=1200,500 omstyle=ColorOnWhiteVector out=vla_phasings.pdf
```

“Thick>" versions, better somewhat better for talks:

```
$ ./plot-alldata.py margin=4 dims=600,600 omstyle=ColorOnWhiteVector out=vla_alldata_thick.pdf
$ ./plot-phasings.py margin=4 dims=600,250 omstyle=ColorOnWhiteVector out=vla_phasings_thick.pdf
```
