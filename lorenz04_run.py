from lorenz04 import Lorenz04
import numpy as np
from netCDF4 import Dataset

# model definition
model = Lorenz04(K=32,forcing=14,space_time_scale=1,coupling=0.4,smooth_steps=12)

# random initial condition
model.z = np.full(model.model_size, model.forcing) + np.random.uniform(-1,1,size=model.model_size)

outputinterval = 24
tmin = 25.
tmax = 125.
# set number of timesteps to integrate for each call to model.advance
ntimesteps = int(outputinterval/model.dt)

nc = Dataset('lorenz04_truth.nc', mode='w', format='NETCDF4_CLASSIC')
nc.model_size = model.model_size
nc.forcing = model.forcing
nc.dt = model.dt
nc.space_time_scale = model.space_time_scale
nc.coupling = model.coupling
nc.K = model.K
nc.smooth_steps = model.smooth_steps

x = nc.createDimension('x',model.model_size)
t = nc.createDimension('t',None)
zvar =\
nc.createVariable('z',np.float64,('t','x'),zlib=True)
# pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
xvar = nc.createVariable('x',np.float64,('x',))
tvar = nc.createVariable('t',np.float64,('t',))
xvar[:] = np.arange(model.model_size)

nout = 0 
while model.t < tmax-model.dt:
    z = model.advance(timesteps=outputinterval)
    print('%g %5.2f %5.2f'%(model.t,z.min(),z.max()))
    if model.t >= tmin:
        print('saving data at t = %g' % model.t)
        zvar[nout,:] = z
        tvar[nout] = model.t
        nout += 1
nc.close()
