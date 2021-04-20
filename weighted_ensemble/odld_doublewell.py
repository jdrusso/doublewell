from west.propagators import WESTPropagator
from west.systems import WESTSystem
from westpa.binning import RectilinearBinMapper

from numpy import linspace, float32, array, empty, zeros, int_, any
from numpy.random import Generator, PCG64
from numpy import sqrt as np_sqrt

from sympy import symbols, lambdify
from sympy import diff as symdiff

pcoord_len = 21
pcoord_dtype = float32


class ODLDPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(ODLDPropagator, self).__init__(rc)

        self.coord_len = pcoord_len
        self.coord_dtype = pcoord_dtype
        self.coord_ndim = 1

        # Initialize at the surface of this state
        self.initial_pcoord = array([2.1e-8], dtype=self.coord_dtype)

        self.dt = 3e-12

        self.mass = 1.55e-25  # 100 Daltons
        # Friction coefficient. This is a collision frequency, but is
        #  usually used as the inverse collision frequency
        self.lambd = 2.494e13  # 24.95 ps^-1
        self.kT = 4.142e-21  # Corresponds to 300K

        self.sigma = np_sqrt(self.kT * 2)
        self.barrier_height = 10 * self.kT

        self.rng = Generator(PCG64())

    def force(self):
        # This gives the force from the doublewell potential at a point.
        # Returns an object that can be referenced to get the potential at a point x as dV(x)

        x = symbols("x")

        expr = (5e7 * x) ** 10 - (3.5e8 * x) ** 2

        # Rescale the potential to the desired barrier height
        vmin = min(lambdify(x, expr)(linspace(0, 4e-8, 10000)))
        vmax = max(lambdify(x, expr)(linspace(-5e-9, 5e-9, 10000)))
        vrange = vmax - vmin

        normalized_expr = expr / vrange

        expr = -1 * normalized_expr * self.barrier_height

        _varx = symbols("x")

        # Derivative of the potential is a force
        dVdx = lambdify(_varx, symdiff(expr, _varx), modules=["math"])

        return dVdx

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""
        state.pcoord = self.initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    def propagate(self, segments):

        n_segs = len(segments)

        coords = empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )
        velocities = empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        # Set the zeroth index to the starting positions
        for iseg, segment in enumerate(segments):
            coords[iseg, 0] = segment.pcoord[0]
            velocities[iseg, 0] = 0.0

        coord_len = self.coord_len
        all_displacements = zeros(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        dVdx = self.force()

        _variance = np_sqrt(2 * self.kT * self.dt / (self.mass * self.lambd))
        for istep in range(1, coord_len):

            _etas = self.rng.normal(0, _variance, n_segs)

            old_xs = coords[:, istep - 1, 0]
            old_vs = velocities[:, istep - 1, 0]

            # Update position using previous velocity
            new_xs = old_xs + old_vs * self.dt

            # Update the velocity according to Langevin dynamics
            # Compute the forces, and multiply by timestep
            # Not adding to the previous timestep means it's noninertial
            new_vs = (-dVdx(new_xs) + _etas) / (self.lambd * self.mass)

            if any(new_xs > 5e-8):
                raise Exception("Simulation blowing up -- timestep likely too large!")

            coords[:, istep, 0] = new_xs
            velocities[:, istep, 0] = new_vs

        for iseg, segment in enumerate(segments):
            segment.pcoord[...] = coords[iseg, :]
            segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments


class ODLDSystem(WESTSystem):
    def initialize(self):
        self.pcoord_ndim = 1
        self.pcoord_dtype = pcoord_dtype
        self.pcoord_len = pcoord_len

        self.bin_mapper = RectilinearBinMapper(
            # [[-float("inf")] + list(linspace(-2e-8, 2e-8, 21)) + [float("inf")]]
            [[-float("inf")] + list(linspace(-1.99e-8, 2e-8, 20)) + [float("inf")]]
        )
        self.bin_target_counts = empty((self.bin_mapper.nbins,), int_)
        self.bin_target_counts[...] = 10
