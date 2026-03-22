import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time
from lorenz04 import Lorenz04, cartdist, getkf_bloc, gaspcohn
from scipy.linalg import eigh

if len(sys.argv) == 1:
   msg="""
python lgetkf_cv_bloc.py hcovlocal_scale
   hcovlocal_scale = horizontal localization scale
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])

profile = False # turn on profiling?

# if savedata not None, netcdf data will be saved with filename 'savedata'
savedata = None
#savedata = 'lgetkfcv_local%s.nc' % hcovlocal_scale
nassim = 1320  # assimilation times to run
nassim_spinup = 120

nanals = 8 # ensemble members
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

print("# hcovlocal=%g nanals=%s ngroups=%s" %\
        (hcovlocal_scale,nanals,ngroups))

# if fixednetwork=True, every nx//nobs grid point is observed (network moves 1 point every assim step).
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

# full model-space horizontal localization matrix (and it's square root)
covlocal_model = np.empty((nx,nx),np.float64)
for n in range(nx):
    dist = cartdist(x[n],x,nc_climo.model_size)
    covlocal_model[n,:] = gaspcohn(dist/hcovlocal_scale)
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
   nc.hcovlocal_scale = hcovlocal_scale
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
        indxob = np.arange(nx)[ntime%nskip::nskip]
    else:
        indxob = np.sort(rsobs.choice(nx,nobs,replace=False))
    zob = z_truth[ntime+ntstart,indxob]
    zob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x[indxob]

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
    zensmean_b = zens.mean(axis=0).copy()
    zerr_b = (zensmean_b-z_truth[ntime+ntstart])**2
    zsprd_b = ((zensmean_b-zens)**2).sum(axis=0)/(nanals-1)

    if savedata is not None:
        z_t[ntime] = z_truth[ntime+ntstart]
        z_b[ntime,:,:] = zens
        z_obs[ntime] = zob
        x_obs[ntime] = xob

    # GETKF update with b localization.
    zens = getkf_bloc(zens, zob-hxensmean_b, oberrvar, sqrtcovlocal_model, indxob, ngroups=ngroups)

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
    plt.title('error (red) and spread (blue) l=%s' % hcovlocal_scale)
    plt.savefig('errorspread_spectra_cv_local%s.png' % hcovlocal_scale)
