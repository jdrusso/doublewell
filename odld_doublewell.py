from west.propagators import WESTPropagator
from west.systems import WESTSystem
from westpa.binning import RectilinearBinMapper

from numpy import arange, float32, array, empty, zeros, int_
from numpy.random import normal as random_normal
from numpy import sqrt as np_sqrt

from sympy import symbols, init_printing, lambdify
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
        self.initial_pcoord = array([10.0], dtype=self.coord_dtype)

        self.dt = 0.0005

        # Friction coefficient. This is a collision frequency, but is
        #  usually used as the inverse collision frequency
        self.lambd = 0.01

        self.kT = 1.0

        self.sigma = np_sqrt(self.kT * 2)
        self.barrier_height = 10 * self.kT

    def force(self):
        # This gives the force from the doublewell potential at a point.
        # Returns an object that can be referenced to get the potential at a point x as dV(x)

        x = symbols("x")

        expr = (0.1 * x) ** 10 - (0.7 * x) ** 2

        vmin = min(lambdify(x, expr)(arange(-17, 17, 0.001)))
        vmax = max(lambdify(x, expr)(arange(-5, 5, 0.001)))
        vrange = vmax - vmin

        # Rescale the potential to the desired barrier height
        normalized_expr = expr / vrange
        expr = normalized_expr * self.barrier_height

        _varx = symbols("x")

        # Derivative of the potential is a force
        dV = lambdify(_varx, symdiff(expr, _varx), modules=["math"])

        return dV

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

        for istep in range(1, coord_len):

            _etas = random_normal(0, self.sigma, n_segs)

            old_xs = coords[:, istep - 1, 0]
            old_vs = velocities[:, istep - 1, 0]

            # Update position using previous velocity
            new_xs = old_xs + old_vs * self.dt

            # Update the velocity according to Langevin dynamics
            # Compute the forces, and multiply by timestep
            # v = v0 + a dT
            # Not adding to the previous timestep means it's noninertial
            _noise = _etas
            correction = np_sqrt(self.lambd / self.dt)
            new_vs = (-self.force()(new_xs) + correction * _noise) / self.lambd

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
            [[-float("inf")] + list(arange(-10.0, 10.0, 1.0)) + [float("inf")]]
        )
        self.bin_target_counts = empty((self.bin_mapper.nbins,), int_)
        self.bin_target_counts[...] = 10
