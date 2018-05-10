HOW TO RUN FEATURIZE:

Make sure you have a MLB_PITCHFX_[YEAR] folder in the DATA directory
and that you are in CODE
If you haven't yet, run:
$python3 resortpfx.py [year]

To featurize, run
$python3 featurize2.py [year] [N]

where N is the number of imaginary regression atbats. I've been using 
N=50

