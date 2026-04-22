import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from lorenz04 import Lorenz04, cartdist, lgetkf_ms, gaspcohn

if len(sys.argv) == 1:
   msg="""
python lgetkf_cvms.py hcovlocal_scales band_cutoffs crossbandcov_facts
   hcovlocal_scales = horizontal localization scale(s)
   band_cutoffs = filter waveband cutoffs 
   crossbandcov_facts = cross-band covariance factors
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scales = eval(sys.argv[1])
nlscales = len(hcovlocal_scales)
band_cutoffs = eval(sys.argv[2])
nband_cutoffs = len(band_cutoffs)
if nband_cutoffs != nlscales-1:
    raise SystemExit('band_cutoffs should be one less than hcovlocal_scales')
crossbandcov_facts = eval(sys.argv[3])
if len(crossbandcov_facts) != nband_cutoffs:
    raise SystemExit('band_cutoffs and crossbandcov_facts should be same length')
crossband_covmat = np.ones((nlscales,nlscales),np.float32)
for i in range(nlscales):
    for j in range(nlscales):
        if j != i:
            crossband_covmat[j,i] = crossbandcov_facts[np.abs(i-j)-1] 

filename_in = 'getkf_cv_1000_400mem.nc'

oberrstdev = 1. # ob error standard deviation in K
ncin = Dataset(filename_in)
x = ncin.variables['x'][:]
nx = len(x)

nanals = 5 # ensemble members
ngroups = nanals  # number of groups for cross-validation (ngroups=nanals//n is "leave n out")
ntin = -1
nxob = int(sys.argv[4])

nobs = 1
zens = ncin['z_b'][ntin,:nanals]
zensfull = ncin['z_b'][ntin]
zensmean_b = zens.mean(axis=0)
zensmean_b_full = zensfull.mean(axis=0)
model = Lorenz04(z=zens[0],model_size=ncin.model_size,\
forcing=ncin.forcing,dt=ncin.dt,space_time_scale=ncin.space_time_scale,\
K=ncin.K,smooth_steps=ncin.smooth_steps)

print("# hcovlocal_scales=%s nanals=%s ngroups=%s" %\
        (repr(hcovlocal_scales),nanals,ngroups))
print('# band_cutoffs=%s crossbandcov_facts=%s' % (repr(band_cutoffs),repr(crossbandcov_facts)))

wavenums = np.abs((nx*np.fft.fftfreq(nx))[0:(nx//2)+1])
kmax = (nx//2)+1

oberrvar = oberrstdev**2*np.ones(nobs,np.float64)
covlocal = np.empty(nx,np.float64)
covlocal_tmp = np.empty((nlscales,nobs,nx),np.float32)

# first-guess spread
zprime = zens - zensmean_b
zsprd_b = ((zensmean_b-zens)**2).sum(axis=0)/(nanals-1)

if nxob < 0:
    nxob = np.argmax(zsprd_b)
zob = zensmean_b[nxob] + 1.
xob = np.array([x[nxob]])
# compute covariance localization function for each ob
for nl in range(nlscales):
   for nob in range(nobs):
       dist = cartdist(xob[nob],x,ncin.model_size)
       covlocal = gaspcohn(dist/hcovlocal_scales[nl])
       covlocal_tmp[nl,nob,:] = covlocal

# hxens is ensemble in observation space.
hxens = np.empty((nanals,nobs),np.float64)
for nanal in range(nanals):
    hxens[nanal] = zens[nanal,nxob]
hxensmean_b = hxens.mean(axis=0)
hxprime_orig = hxens - hxensmean_b

# filter background perturbations into different scale bands
if nlscales == 1:
    zens_filtered_lst=[zprime]
else:
    zens_filtered_lst=[]
    zfilt_save = np.zeros_like(zprime)
    zspec = np.fft.rfft(zprime)
    for n,cutoff in enumerate(band_cutoffs):
        zfiltspec = np.where(wavenums[np.newaxis,...] < cutoff, zspec, 0.+0.j)
        if len(band_cutoffs) == 1 and cutoff == 999:
            # for 2 scales and band_cutoff=999, use model filter
            zfilt = np.empty_like(zens)
            for nanal in range(nanals):
                zfilt[nanal], _ = models[nanal].z2xy(zprime[nanal])
        else:
            zfiltspec = np.where(wavenums[np.newaxis,...] < cutoff, zspec, 0.+0.j)
            zfilt = np.fft.irfft(zfiltspec)
        zens_filtered_lst.append(zfilt-zfilt_save)
        #plt.figure()
        #plt.plot(x,(zfilt-zfilt_save)[0,...])
        #plt.title('scale = %s' % n)
        zfilt_save=zfilt
    zsum = np.zeros_like(zprime)
    for n in range(nband_cutoffs):
        zsum += zens_filtered_lst[n]
    zens_filtered_lst.append(zprime-zsum)
    #plt.plot(x,(zprime-zsum)[0,...])
    #plt.title('scale = %s' % nlscales)
    #plt.show()
    #raise SystemExit
zens_filtered = np.asarray(zens_filtered_lst)
zprime = np.dot(zens_filtered.T,crossband_covmat).T

hxprime = np.empty((nanals*nlscales,nobs),np.float32)
xprime = zprime.reshape(nanals*nlscales, nx)
for nanal in range(nanals*nlscales):
    hxprime[nanal] = xprime[nanal,nxob] 

# EnKF update
# (note zens contains unfiltered (original) ensemble, xprime has filtered perturbations separated into wave bands).
print(zob, hxensmean_b, zsprd_b[nxob])
zens = lgetkf_ms(nlscales,zens,xprime,hxprime,hxprime_orig,zob-hxensmean_b,oberrvar,covlocal_tmp,ngroups=ngroups)

zensmean_a = zens.mean(axis=0)
plt.plot(x,zensmean_a-zensmean_b)
plt.xlim(nxob-25,nxob+25)
plt.ylim(-0.4,0.4)
plt.show()
