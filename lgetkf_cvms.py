import matplotlib
matplotlib.use('agg')
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
crossband_covmatr = np.ones((nlscales,nlscales),np.float32)
for i in range(nlscales):
    for j in range(nlscales):
        if j != i:
            crossband_covmat[j,i] = crossbandcov_facts[np.abs(i-j)-1] 
            crossband_covmatr[j,i] = -crossbandcov_facts[np.abs(i-j)-1] 

profile = False # turn on profiling?

# if savedata not None, netcdf data will be saved with filename 'savedata'
savedata = None
exptname = os.getenv('exptname','test')
#savedata = 'lgetkfcv_%s.nc' % exptname
nassim = 1320  # assimilation times to run
nassim_spinup = 120

nanals = 8 # ensemble members
nerger = True # use Nerger regularization for R localization
ngroups = nanals  # number of groups for cross-validation (ngroups=nanals//n is "leave n out")

oberrstdev = 1. # ob error standard deviation in K

# nature run created using lorenz04_run.py.
filename_climo = 'lorenz04_truth.nc' # file name for forecast model climo
# perfect model
filename_truth = 'lorenz04_truth.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
#rsics = np.random.RandomState() # varying seed for initial conditions
rsics = np.random.RandomState(24) # fixed seed for initial conditions

# get model info
nc_climo = Dataset(filename_climo)
x = nc_climo.variables['x'][:]
nx = len(x)
dt = nc_climo.dt
zens = np.empty((nanals,nx),np.float64)
z_climo = nc_climo.variables['z']
indxran = rsics.choice(z_climo.shape[0],size=nanals,replace=False)
models = []
for nanal in range(nanals):
    zens[nanal] = z_climo[indxran[nanal]]
    #print(nanal, zens[nanal].min(), zens[nanal].max())
    models.append(\
    Lorenz04(z=zens[nanal],model_size=nc_climo.model_size,\
    forcing=nc_climo.forcing,dt=nc_climo.dt,space_time_scale=nc_climo.space_time_scale,\
    K=nc_climo.K,smooth_steps=nc_climo.smooth_steps))

print("# hcovlocal_scales=%s nanals=%s ngroups=%s" %\
        (repr(hcovlocal_scales),nanals,ngroups))
print('# band_cutoffs=%s crossbandcov_facts=%s' % (repr(band_cutoffs),repr(crossbandcov_facts)))

# if fixednetwork=True, every nx//nobs grid point is observed.
# otherwise, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
fixednetwork = False
nobs = nx//12

# nature run
nc_truth = Dataset(filename_truth)
z_truth = nc_truth.variables['z']
# set up arrays for obs and localization function
if fixednetwork:
    print('# fixed network nobs = %s' % nobs)
else:
    print('# random network nobs = %s' % nobs)

oberrvar = oberrstdev**2*np.ones(nobs,np.float64)
covlocal = np.empty(nx,np.float64)
covlocal_tmp = np.empty((nlscales,nobs,nx),np.float32)

obtimes = nc_truth.variables['t'][:]
ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# assim interval = %s secs (%s time steps)' % (assim_interval,assim_timesteps))
print('# ntime,zerr_a,zsprd_a,zerr_b,zsprd_b')

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[ntstart]

# initialize output file.
if savedata is not None:
   nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
   nc.model_size = models[0].model_size
   nc.forcing = models[0].forcing
   nc.dt = models[0].dt
   nc.space_time_scale = models[0].space_time_scale
   nc.coupling = models[0].coupling
   nc.K = models[0].K
   nc.smooth_steps = models[0].smooth_steps
   nc.nanals = nanals
   nc.hcovlocal_scales = hcovlocal_scales
   nc.band_cutoffs = band_cutoffs
   nc.crossbandcov_facts = crossband_covfacts
   nc.oberrstdev = oberrstdev
   nc.dt = models[0].dt
   nc.filename_climo = filename_climo
   nc.filename_truth = filename_truth
   xdim = nc.createDimension('x',models[0].model_size)
   tdim = nc.createDimension('t',None)
   obs = nc.createDimension('obs',nobs)
   ens = nc.createDimension('ens',nanals)
   z_t =\
   nc.createVariable('z_t',np.float32,('t','x'),zlib=True)
   z_b =\
   nc.createVariable('z_b',np.float32,('t','ens','x'),zlib=True)
   z_a =\
   nc.createVariable('z_a',np.float32,('t','ens','x'),zlib=True)
   z_obs = nc.createVariable('obs',np.float32,('t','obs'))
   x_obs = nc.createVariable('x_obs',np.float32,('t','obs'))
   xvar = nc.createVariable('x',np.float32,('x',))
   tvar = nc.createVariable('t',np.float32,('t',))
   ensvar = nc.createVariable('ens',np.int32,('ens',))
   xvar[:] = np.arange(0,models[0].model_size)
   ensvar[:] = np.arange(1,nanals+1)

# initialize kinetic energy error/spread spectra
zspec_errmean = None; zspec_sprdmean = None

ncount = 0

wavenums = np.abs((nx*np.fft.fftfreq(nx))[0:(nx//2)+1])
kmax = (nx//2)+1

for ntime in range(nassim):

    # check model clock
    if np.abs(models[0].t - obtimes[ntime+ntstart]) > 1.e-6:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()
    if fixednetwork:
        nskip = nx//nobs
        indxob = np.arange(nx)[::nskip]
    else:
        indxob = np.sort(rsobs.choice(nx,nobs,replace=False))
    zob = z_truth[ntime+ntstart,indxob]
    zob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x[indxob]
    # compute covariance localization function for each ob
    for nl in range(nlscales):
       for nob in range(nobs):
           dist = cartdist(xob[nob],x,nc_climo.model_size)
           covlocal = gaspcohn(dist/hcovlocal_scales[nl])
           covlocal_tmp[nl,nob,:] = covlocal

    # first-guess spread
    zensmean = zens.mean(axis=0)
    zprime = zens - zensmean

    fsprd = (zprime**2).sum(axis=0)/(nanals-1)

    # compute forward operator on modulated ensemble.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float64)

    for nanal in range(nanals):
        hxens[nanal] = zens[nanal,indxob] 
    hxensmean_b = hxens.mean(axis=0)
    hxprime_orig = hxens - hxensmean_b
    zensmean_b = zens.mean(axis=0).copy()
    zerr_b = (zensmean_b-z_truth[ntime+ntstart])**2
    zsprd_b = ((zensmean_b-zens)**2).sum(axis=0)/(nanals-1)

    if savedata is not None:
        z_t[ntime] = z_truth[ntime+ntstart]
        z_b[ntime,:,:] = zens
        z_obs[ntime] = zob
        x_obs[ntime] = xob

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
                #filtfact = np.exp(-(wavenums[np.newaxis,...]/cutoff)**8)
                #zfiltspec = filtfact*zspec
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
        hxprime[nanal] = xprime[nanal,indxob] # surface pv obs

    # EnKF update
    # (note zens contains unfiltered (original) ensemble, xprime has filtered perturbations separated into wave bands).
    zens = lgetkf_ms(nlscales,zens,xprime,hxprime,hxprime_orig,zob-hxensmean_b,oberrvar,covlocal_tmp,ngroups=ngroups)

    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    zensmean_a = zens.mean(axis=0)

    # print out analysis error, spread and innov stats for background
    zerr_a = (zensmean_a-z_truth[ntime+ntstart])**2
    zsprd_a = ((zensmean_a-zens)**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(zerr_a.mean()),np.sqrt(zsprd_a.mean()),\
     np.sqrt(zerr_b.mean()),np.sqrt(zsprd_b.mean())))

    # save data.
    if savedata is not None:
        z_a[ntime,:,:] = zens
        tvar[ntime] = obtimes[ntime+ntstart]
        nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        zens[nanal] = models[nanal].advance(timesteps=assim_timesteps,z=zens[nanal])
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)
    if not np.all(np.isfinite(zens)):
        raise SystemExit('non-finite values detected after forecast, stopping...')

    # compute spectra of error and spread
    if ntime >= nassim_spinup:
       zfcstmean = zens.mean(axis=0)
       zerrspec = np.fft.rfft(zfcstmean - z_truth[ntime+ntstart+1])
       zerrspec_mag = (zerrspec*np.conjugate(zerrspec)).real
       if zspec_errmean is None:
           zspec_errmean = zerrspec_mag
       else:
           zspec_errmean = zspec_errmean + zerrspec_mag
       for nanal in range(nanals):
           zpertspec = np.fft.rfft(zens[nanal] - zfcstmean)
           zpertspec_mag = (zpertspec*np.conjugate(zpertspec)).real/(nanals-1)
           if zspec_sprdmean is None:
               zspec_sprdmean = zpertspec_mag
           else:
               zspec_sprdmean = zspec_sprdmean+zpertspec_mag
       ncount += 1

#if savedata is not None: nc.close()

if ncount:
    zspec_sprdmean = zspec_sprdmean/ncount
    zspec_errmean = zspec_errmean/ncount
    print('# mean error/spread',zspec_errmean.sum(), zspec_sprdmean.sum())
    plt.figure()
    for n in range(kmax):
        print('# ',wavenums[n],zspec_errmean[n],zspec_sprdmean[n])
    plt.loglog(wavenums,zspec_errmean,color='r')
    plt.loglog(wavenums,zspec_sprdmean,color='b')
    plt.title('error (red) and spread (blue) %s' % exptname)
    plt.savefig('errorspread_spectra_cv_%s.png' % exptname)
