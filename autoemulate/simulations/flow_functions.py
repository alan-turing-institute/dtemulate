import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


class FlowProblem:
    def __init__(
        self,
        T=1.0,
        td=0.2,
        amp=900.0,
        dt=0.001,
        ncycles=10,
        ncomp=10,
        C=38.0,
        R=0.06,
        L=0.0017,
        R_o=0.025,
        p_o=10.0,
    ) -> None:
        """
        Inputs
        -----
        T (float): cycle length (default 1.0)
        td (float): pulse duration, make sure to make this less than T (default 0.2)
        amp (float): inflow amplitude (default 1.0)
        dt (float): temporal discretisation resolution (default 0.001)
        C (float): tube average compliance (default 38.)
        R (float): tube average impedance (default 0.06)
        L (float): hydraulic impedance, inertia (default 0.0017)
        R_o (float) : outflow resistance
        p_o (float) : outflow pressure
        """

        assert td < T, f"td should be smaller than T but {td} >= {T}."

        self._td = td
        self._T = T
        self._dt = dt
        self._amp = amp

        self._ncomp = ncomp
        self._ncycles = ncycles

        self._C = C
        self._R = R
        self._L = L

        self._R_o = R_o
        self._p_o = p_o

        self.res = None

    @property
    def td(self):
        return self._td

    @property
    def T(self):
        return self._T

    @property
    def dt(self):
        return self._dt

    @property
    def amp(self):
        return self._amp

    @property
    def ncomp(self):
        return self._ncomp

    @property
    def ncycles(self):
        return self._ncycles

    @property
    def C(self):
        return self._C

    @property
    def L(self):
        return self._L

    @property
    def R(self):
        return self._R

    @property
    def R_o(self):
        return self._R_o

    @property
    def p_o(self):
        return self._p_o

    def generate_pulse_function(self):
        self.Q_mi_lambda = (
            lambda t: np.sin(np.pi / self.td * t) ** 2.0
            * np.heaviside(self.td - t, 0.0)
            * self.amp
        )

    def dfdt_fd(self, t: float, y: np.ndarray, Q_in):
        Cn = self.C / self.ncomp
        Rn = self.R / self.ncomp
        Ln = self.L / self.ncomp

        out = np.zeros((self.ncomp, 2))
        y_temp = y.reshape((-1, 2))

        for i in range(self.ncomp):
            if i > 0:
                out[i, 0] = (y_temp[i - 1, 1] - y_temp[i, 1]) / Cn
            else:
                out[i, 0] = (Q_in(t % self.T) - y_temp[i, 1]) / Cn
            if i < self.ncomp - 1:
                out[i, 1] = (-y_temp[i + 1, 0] + y_temp[i, 0] - Rn * y_temp[i, 1]) / Ln
                pass
            else:
                out[i, 1] = (
                    -self.p_o + y_temp[i, 0] - (Rn + self.R_o) * y_temp[i, 1]
                ) / Ln
        return out.reshape((-1,))

    def solve(self):
        dfdt_fd_spec = lambda t, y: self.dfdt_fd(t=t, y=y, Q_in=self.Q_mi_lambda)
        self.res = sp.integrate.solve_ivp(
            dfdt_fd_spec,
            [0.0, self.T * self.ncycles],
            y0=np.zeros(self.ncomp * 2),
            method="BDF",
            max_step=self.dt,
        )
        self.res.y = self.res.y[:, self.res.t >= self.T * (self.ncycles - 1)]
        self.res.t = self.res.t[self.res.t >= self.T * (self.ncycles - 1)]

    def plot_res(self):
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        for i in range(self.ncomp):
            ax[0].plot(
                self.res.t,
                self.res.y[2 * i, :],
                "r",
                alpha=0.1 + (1.0 - i / self.ncomp) * 0.9,
            )
            ax[1].plot(
                self.res.t,
                self.res.y[2 * i + 1, :],
                "r",
                alpha=0.1 + (1.0 - i / self.ncomp) * 0.9,
            )

        ax[0].set_title("Pressure")
        ax[1].set_title("Flow rate")
        ax[0].set_xlabel("Time (s)")
        ax[1].set_xlabel("Time (s)")
        ax[0].set_ylabel("mmHg")
        ax[1].set_ylabel("$ml\cdot s^{-1}$")

        return (fig, ax)
