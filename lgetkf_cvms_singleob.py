import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from scipy.linalg import eigh
from lorenz04 import Lorenz04, cartdist, lgetkf_ms, lgetkf, gaspcohn, getkf_bloc

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

filename_in = 'getkf_cv_150_500mem.nc'

oberrstdev = 1. # ob error standard deviation in K
ncin = Dataset(filename_in)
x = ncin.variables['x'][:]
nx = len(x)

nanals = 5 # ensemble members
ngroups = nanals  # number of groups for cross-validation (ngroups=nanals//n is "leave n out")
ntin = -1
nxob = int(sys.argv[4])

nobs = 1
nanalsfull = ncin['z_b'].shape[1]
zens = ncin['z_b'][ntin,:nanals]
zensfull = ncin['z_b'][ntin]
zensmean_b = zens.mean(axis=0)
zensmean_b_full = zensfull.mean(axis=0)

#model = Lorenz04(z=zens[0],model_size=ncin.model_size,\
#forcing=ncin.forcing,dt=ncin.dt,space_time_scale=ncin.space_time_scale,\
#K=ncin.K,smooth_steps=ncin.smooth_steps)

print("# hcovlocal_scales=%s nanals=%s ngroups=%s" %\
        (repr(hcovlocal_scales),nanals,ngroups))
print('# band_cutoffs=%s crossbandcov_facts=%s' % (repr(band_cutoffs),repr(crossbandcov_facts)))

wavenums = np.abs((nx*np.fft.fftfreq(nx))[0:(nx//2)+1])
kmax = (nx//2)+1

oberrvar = oberrstdev**2*np.ones(nobs,np.float64)
covlocal = np.empty(nx,np.float64)
lscale_full = hcovlocal_scales[0]*10

# full model-space horizontal localization matrix (and it's square root)
covlocal_model = np.empty((nx,nx),np.float64)
for n in range(nx):
    dist = cartdist(x[n],x,ncin.model_size)
    covlocal_model[n,:] = gaspcohn(dist/lscale_full)
evals, evecs = eigh(covlocal_model, driver='evd')
evals = evals.clip(min=np.finfo(evals.dtype).eps)
percentvar_cutoff = 0.99
neig = 1
for i in range(1,nx):
     percentvar = evals[-i:].sum()/evals.sum()
     if percentvar > percentvar_cutoff: # perc variance cutoff truncation
         neig = i
         break
evecs_norm = np.dot(evecs, np.diag(np.sqrt(evals/percentvar))).T
sqrtcovlocal_model = evecs_norm[-neig:,:].astype(np.float32)
print('# neig = %s' % neig)

# first-guess spread
zprime = zens - zensmean_b
zsprd_b = ((zensmean_b-zens)**2).sum(axis=0)/(nanals-1)
zsprd_b_full = ((zensmean_b_full-zensfull)**2).sum(axis=0)/(nanalsfull-1)

if nxob < 0:
    nxob = np.argmax(zsprd_b)

# 1. full ensemble

xob = np.array([x[nxob]])
covlocal_tmp = np.empty((nobs,nx),np.float64)
# compute covariance localization function for each ob
for nob in range(nobs):
    dist = cartdist(xob[nob],x,ncin.model_size)
    covlocal = gaspcohn(dist/lscale_full)
    covlocal_tmp[nob] = covlocal
# EnKF update
zob = zensmean_b_full[nxob] + 1.
# hxens is ensemble in observation space.
hxensfull = np.empty((nanalsfull,nobs),np.float64)
for nanal in range(nanalsfull):
    hxensfull[nanal] = zensfull[nanal,nxob]
hxensmean_b_full = hxensfull.mean(axis=0)
#zensfull = lgetkf(zensfull,hxensfull,zob,oberrvar,covlocal_tmp,nerger=True,ngroups=10)
zensfull = getkf_bloc(zensfull, zob-hxensmean_b_full, oberrvar, sqrtcovlocal_model, np.array([nxob]), ngroups=10)
zensmean_a = zensfull.mean(axis=0)
inc_fullens = zensmean_b_full - zensmean_a

# 2. small ensemble, single scale

# compute covariance localization function for each ob
for nob in range(nobs):
    dist = cartdist(xob[nob],x,ncin.model_size)
    covlocal = gaspcohn(dist/hcovlocal_scales[1])
    covlocal_tmp[nob] = covlocal
zob = zensmean_b[nxob] + 1.
hxens = np.empty((nanals,nobs),np.float64)
for nanal in range(nanals):
    hxens[nanal] = zens[nanal,nxob]
zens1 = lgetkf(zens,hxens,zob,oberrvar,covlocal_tmp,nerger=True,ngroups=nanals)
zensmean_a = zens1.mean(axis=0)
inc_1scale = zensmean_b - zensmean_a


# 2. small ensemble, two scales

# compute covariance localization function for each ob
covlocal_tmp = np.empty((nlscales,nobs,nx),np.float32)
for nl in range(nlscales):
   for nob in range(nobs):
       dist = cartdist(xob[nob],x,ncin.model_size)
       covlocal = gaspcohn(dist/hcovlocal_scales[nl])
       covlocal_tmp[nl,nob,:] = covlocal
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
zens2 = lgetkf_ms(nlscales,zens,xprime,hxprime,hxprime_orig,zob-hxensmean_b,oberrvar,covlocal_tmp,ngroups=ngroups)
zensmean_a = zens2.mean(axis=0)
inc_2scales = zensmean_b - zensmean_a

plt.plot(x,inc_fullens,color='k',label='500 members, L=%s' % lscale_full)
plt.plot(x,inc_1scale,color='b',label='5 members, L=%s' % hcovlocal_scales[1])
plt.plot(x,inc_2scales,color='r',label='5 members, L=%s' % repr(hcovlocal_scales))
plt.xlim(nxob-25,nxob+25)
#plt.ylim(-0.4,0.4)
plt.legend()
plt.show()
