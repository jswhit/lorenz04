import numpy as np

# Parameter	Nz	K	I	F	b	c
#This work	960	32	12	14	1	0.37
#Lorenz 	960	32	12	15	1	2.5
# b-> space_time_scale

class Lorenz04:
    def __init__(
        self,
        z = None,
        model_size: int = 960,
        forcing: float = 14.00,
        dt: float = 0.05/24,
        space_time_scale: float = 1.00,
        coupling: float = 0.37,
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
    model = Lorenz04()

    # random initial condition
    model.z = np.full(model.model_size, model.forcing) + np.random.uniform(-1,1,size=model.model_size)
    # spinup
    nspinup = 12000 # 25 time units if dt=0.05/24
    import time
    t1 = time.time()
    for _ in range(nspinup):
        z = model.timestep()
    t2 = time.time()
    print(t2-t1,'seconds to run spinup')
    raise SystemExit

    windowsteps = 24 # plot every windowsteps time steps

    # plot animation
    import matplotlib
    matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
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
