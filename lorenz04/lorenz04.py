import numpy as np

# model III from Lorenz 2005 (https://doi.org/10.1175/JAS3430.1)
# used by Harty et al 2021 (https://doi.org/10.1080/16000870.2021.1903692)
# translated from Fortran code in DART

# Parameter	Nz	K	I	F	b	c
# Harty et al	960	32	12	14	1	0.37
# Lorenz 	960	32	12	15	10	2.5
# where b-> space_time_scale, c-> coupling, F-> forcing, I-> smooth_steps

class Lorenz04:

# The model equations are given by
# 
# Model 3 (III)
#      dZ_i
#      ---- = [X,X]_{K,i} + b^2 (-Y_{i-2}Y_{i-1} + Y_{i-1}Y_{i+1})
#       dt                +  c  (-Y_{i-2}X_{i-1} + Y_{i-1}X_{i+1})
#                         -  X_i - b Y_i + F,
#
# where
#
#     [X,X]_{K,i} = -W_{i-2K}W_{i-K} 
#                 +  sumprime_{j=-(K/2)}^{K/2} W_{i-K+j}X_{i+K+j}/K,
#
#      W_i =  sumprime_{j=-(K/2)}^{K/2} X_{i-j}/K,
#
# and sumprime denotes a special kind of summation where the first
# and last terms are divided by 2.
#
# NOTE: The equations above are only valid for K even.  If K is odd,
# then sumprime is replaced by the traditional sum, and the K/2 limits
# of summation are replaced by (K-1)/2. THIS CODE ONLY IMPLEMENTS THE
# K EVEN SOLUTION!!!
#
# The variable that is integrated is Z,
# but the integration of Z requires
# the variables X and Y.  For model III they are obtained by
#
#      X_i = sumprime_{j= -J}^{J} a_j Z_{i+j}
#      Y_i = Z_i - X_i.
#
# The "a" coefficients are given by
#
#      a_j = alpha - beta |j|,
# 
# where
#
#      alpha = (3J^2 + 3)/(2J^3 + 4J)
#      beta  = (2J^2 + 1)/(1J^4 + 2J^2).
#
# This choice of alpha and beta ensures that X_i will equal Z_i
# when Z_i varies quadratically over the interval 2J.   This choice
# of alpha and beta means that sumprime a_j = 1 and 
# sumprime (j^2) a_j = 0.
#
# Note that the impact of this filtering is to put large-scale
# variations into the X variable, and small-scale variations into
# the Y variable.

    def __init__(
        self,
        z = None,
        model_size: int = 960,
        forcing: float = 14.00,
        dt: float = 0.05/24,
        space_time_scale: float = 1.00,
        coupling: float = 0.4,
        K: int = 32,
        smooth_steps: int = 12
    ):

        if (K + 1) // 2 != K // 2:
            raise ValueError("Model only handles even values of K")
        self.K = K
        self.model_size = model_size
        self.forcing = forcing
        self.dt = dt
        self.space_time_scale = space_time_scale
        self.coupling = coupling
        self.smooth_steps = smooth_steps
        self.z = z
        self.t = 0

        self._static_init()

    def _static_init(self):
        K  = self.K
        ss = self.smooth_steps
        N  = self.model_size

        alpha = (3.0*ss**2 + 3.0) / (2.0*ss**3 + 4.0*ss)
        beta  = (2.0*ss**2 + 1.0) / (ss**4  + 2.0*ss**2)

        a = np.array([alpha - beta*abs(i) for i in range(-ss, ss+1)])

        # sumprime weights baked into 'a' once — halve endpoints
        self._aw = a.copy()
        self._aw[0]  *= 0.5
        self._aw[-1] *= 0.5

        self.H   = K // 2
        self.K2  = 2 * K
        self.K4  = 4 * K
        self.ss2 = 2 * ss
        self.sts2 = self.space_time_scale ** 2

        # Precompute sumprime weights for the W / xx loops (length 2H+1)
        H = self.H
        wp = np.ones(2*H + 1)
        wp[0] = wp[-1] = 0.5
        self._wp = wp / K          # divide by K once here

        # Precompute all cyclic index arrays (done once, reused every step)
        i = np.arange(N)

        # z2xy:  for each output i, tap offsets are -(+ss) .. +(+ss) -> i+ss .. i-ss
        # zwrap = _wrap(z, ss); column k accesses zwrap[i + k], k=0..2ss
        # zwrap(i - j) with j=-ss..ss  <=>  k = ss - j, so k runs ss+ss..ss-ss = 2ss..0
        # which is just the reversed a order — already handled by how we built aw.
        self._z2x_idx = (i[:, None] + np.arange(2*ss+1)[None, :]) % N  # (N, 2ss+1) — pure modular, no buffer needed

        # For z2xy we'll use np.take with mode='wrap' — precompute offsets
        self._z2x_offsets = np.arange(ss, -ss-1, -1)   # ss, ss-1, ..., -ss  (= -j for j=-ss..ss)

        # gettend index arrays (all mod N, no explicit buffer)
        # W calculation: xw(i - j) for j = -H..H  => offset = +H..−H
        self._w_offsets = np.arange(H, -H-1, -1)        # shape (2H+1,)
        self._w_idx = (i[:, None] + self._w_offsets[None, :]) % N   # (N, 2H+1)

        # xx bracket: wx(i-K+j) and xw(i+K+j) for j=-H..H
        j = np.arange(-H, H+1)
        self._wx_off = (-K + j)   # offsets into wx  (shape 2H+1)
        self._xw_off = ( K + j)   # offsets into xw  (shape 2H+1)
        self._wx_bracket_idx = (i[:, None] + self._wx_off[None, :]) % N  # (N,2H+1)
        self._xw_bracket_idx = (i[:, None] + self._xw_off[None, :]) % N

        # scalar index arrays for the remaining terms in dzdt
        self._im2 = (i - 2) % N
        self._im1 = (i - 1) % N
        self._ip1 = (i + 1) % N
        self._imK  = (i - K)  % N
        self._imK2 = (i - self.K2) % N

    # ------------------------------------------------------------------
    def z2xy(self, z: np.ndarray):
        """Decompose z into large-scale x and small-scale y."""
        # Use np.take with wrap mode — no explicit buffer allocation
        # Shape: (N, 2*ss+1); column k = z[(i + offset_k) % N]
        Z = np.take(z, (np.arange(self.model_size)[:, None] + self._z2x_offsets[None, :]) % self.model_size)
        x = Z @ self._aw
        return x, z - x

    # ------------------------------------------------------------------
    def gettend(self, z: np.ndarray) -> np.ndarray:
        x, y = self.z2xy(z)

        # ---- W: sumprime_{j=-H}^{H} x(i-j) / K  (weights pre-divided by K) ----
        wx = x[self._w_idx] @ self._wp          # (N,)

        # ---- [X,X]_K bracket ----
        WX = wx[self._wx_bracket_idx]           # (N, 2H+1)
        XW =  x[self._xw_bracket_idx]           # (N, 2H+1)

        prod = WX * XW
        prod[:, 0]  *= 0.5
        prod[:, -1] *= 0.5

        xx = prod.sum(axis=1) / self.K - wx[self._imK2] * wx[self._imK]

        # ---- dz/dt ----
        dzdt = (
            xx
            + self.sts2   * (-y[self._im2]*y[self._im1] + y[self._im1]*y[self._ip1])
            + self.coupling * (-y[self._im2]*x[self._im1] + y[self._im1]*x[self._ip1])
            - x
            - self.space_time_scale * y
            + self.forcing
        )
        return dzdt

    def timestep(self):

        x = self.z.copy()
        dt = self.dt
        k1 = dt * self.gettend(x)
        k2 = dt * self.gettend(x + 0.5*k1)
        k3 = dt * self.gettend(x + 0.5*k2)
        k4 = dt * self.gettend(x + k3)

        self.z = x + (k1 + 2.0*k2 + 2.0*k3 + k4) * (1.0/6.0)
        self.x, self.y = self.z2xy(self.z)
        self.t += self.dt
        return self.z

    def advance(self, timesteps=1, z=None):
        # advance forward specified number of timesteps
        if z is not None:
            self.z = z
        for _ in range(timesteps):
            z = self.timestep()
        return z

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    #model = Lorenz04(space_time_scale=10,forcing=15,coupling=2.5) # original settings from Lorenz paper
    model = Lorenz04()

    # random initial condition
    rs = np.random.RandomState(42)
    model.z = np.full(model.model_size, model.forcing) + rs.uniform(-1,1,size=model.model_size)
    # spinup
    nspinup = 12000 # 25 time units if dt=0.05/24
    import time
    t1 = time.time()
    for n in range(nspinup):
        z = model.timestep()
    t2 = time.time()
    print(t2-t1,'seconds to run spinup')
    zsave = np.empty((nspinup,model.model_size),np.float64)
    for n in range(nspinup):
        zsave[n] = model.timestep()
    zmean = zsave.mean(axis=0)
    zprime = zsave - zmean
    for n in range(nspinup):
        zspec = np.fft.rfft(zprime[n])
        zspec_mag = (zspec*np.conjugate(zspec)).real
        if not n:
            zspec_mean = zspec_mag/nspinup
        else:
            zspec_mean += zspec_mag/nspinup
    print('mean = ',zmean.mean())
    print('stdev = ',np.sqrt((zprime**2).sum(axis=0)/(nspinup-1)).mean())

    import matplotlib
    matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    plt.figure()
    nx = model.model_size
    wavenums = np.abs((nx*np.fft.fftfreq(nx))[0:(nx//2)+1])
    plt.loglog(wavenums,zspec_mean,color='k')
    plt.title('Z wavenumber spectra')
    plt.savefig('zspec.png')

    windowsteps = 24 # plot every windowsteps time steps
    # plot animation
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    x = np.arange(model.model_size)
    line = ax.plot(x,z)[0]
    ax.set_ylim(-20,20)

    def updatefig(*args):
        z = model.advance(timesteps=windowsteps)
        line.set_ydata(z)
        return line
    
    # interval=0 means draw as fast as possible
    ani = animation.FuncAnimation(fig, updatefig, frames=500, interval=10, repeat=False)
    plt.show()
