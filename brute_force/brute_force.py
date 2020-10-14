import numpy as np
from numpy.random import uniform

from sympy import symbols, init_printing, lambdify
from sympy import diff as symdiff
from numpy.random import Generator, PCG64
import ray

from rich.progress import Progress


# Set up the potential. Barrier height should be provided in units of kT
def doublewell_potential(barrier_height, kT):

    x = symbols("x")
    init_printing(use_unicode=True)

    expr = (0.1 * x) ** 10 - (0.7 * x) ** 2

    vmin = min(lambdify(x, expr)(np.arange(-17, 17, 0.001)))
    vmax = max(lambdify(x, expr)(np.arange(-5, 5, 0.001)))
    vrange = vmax - vmin

    normalized_expr = expr / vrange

    barrier_height_kT = barrier_height * kT

    expr = normalized_expr * barrier_height_kT

    return expr


# Starting from a position start_x, run a trajectory with timestep dt for
#    at most max_t steps.
# If the trajectory reaches either basin return the index of the basin it
#    went to and the list of x positions.
# (The _l is for _local instead of distributed via Ray)
def generate_trajectory_l(
    start_x,
    sigma,
    lambd,
    kT,
    max_t=100,
    dt=0.001,
    output=True,
    _barrier_height=5,
    BASINS=[[-np.inf, -8], [8, np.inf]],
    subsample_result=1,
):

    potential = doublewell_potential(barrier_height=_barrier_height, kT=kT)

    # Use the math module here, since it'll be faster
    _varx = symbols("x")
    dVdx = lambdify(_varx, symdiff(potential, _varx), modules=["math"])

    np.seterr(over="raise", invalid="raise")

    print(
        f"max: {max_t:.2e} | dt: {dt:.2e} | sigma: {sigma:.2e} | Barrier: {_barrier_height} kT | basins: {BASINS} "
    )

    # How many random etas to generate at once
    ETA_CHUNK = 1000

    # Maximum number of steps a trajectory should run for
    n_steps = int(max_t // dt) + 1
    print(f"Running for {n_steps:.2e} steps, storing {n_steps//subsample_result:.2e}")

    # This array creation is pretty slow, but it beats appending!
    _xs = np.empty(n_steps // subsample_result)
    _vs = np.empty(n_steps // subsample_result)

    # Precompute these, much faster than doing them one by one!
    # However, computing more than necessary ends up being a MAJOR
    #     bottleneck for this function.
    # So, compute them in chunks instead. 1000 is a pretty reasonable size,
    #     but this could be tuned depending on general trajectory lengths
    rg = Generator(PCG64())
    _etas = rg.normal(0, sigma, ETA_CHUNK)
    used_eta = 0

    correction = np.sqrt(lambd / dt)

    previous_x = None
    current_x = None
    previous_v = None

    for i in range(int(max_t / dt)):

        used_eta += 1
        # If you've used up all your random numbers, make a new set
        if used_eta % (ETA_CHUNK - 1) == 0:
            _etas = np.random.normal(0, sigma, ETA_CHUNK)

        # If you're just starting, set the first position
        if i == 0:
            current_x = start_x
        # Update position using previous velocity
        else:
            current_x = previous_x + previous_v * dt

        # Update the velocity according to Langevin dynamics
        # Compute the forces, and multiply by timestep
        # v = v0 + a dT
        # Not adding to the previous timestep means it's noninertial
        try:
            _noise = _etas[i % ETA_CHUNK]
            _v = (-dVdx(current_x) + correction * _noise) / lambd
        except FloatingPointError as e:
            print("Errored out")
            print(
                f"X: {_xs[i-5:i]} | v-1: {_vs[i-6:i-1]} | l: {lambd} | e: {_etas[i%ETA_CHUNK-5:i%ETA_CHUNK]}"
            )
            raise e

        # Check if you've reached any of the basins and return its index if so
        for j, _bin in enumerate(BASINS):
            if current_x > _bin[0] and current_x < _bin[1]:
                return j, _xs[: i + 1]

        previous_x = current_x
        previous_v = _v

        # If you're generating long trajectories, you may want to subsample the result so you're not unnecessarily using
        #    tons of memory.
        # It's more efficient to subsample when storing, rather than storing the full-resolution trajectory and then
        #    subsampling for that reason, to minimize the total memory footprint at any time.
        if i % subsample_result == 0:
            _vs[i // subsample_result] = _v
            _xs[i // subsample_result] = current_x

    return _xs[: i + 1]


# This is a wrapper function for generate_trajectory that allows it to be called through Ray.
# I've done it this way rather than one function so you can still call generate_trajectory_l
#   without needing Ray at all.
@ray.remote
def generate_trajectory(*args, **kwargs):

    return generate_trajectory_l(*args, **kwargs)


def generate_start_positions(n_starts, barrier_height, kT):

    potential = doublewell_potential(barrier_height, kT)

    _varx = symbols("x")
    V = lambdify(_varx, potential, modules=["math"])

    _range = np.arange(-12, 12, 0.01)
    pdf = np.exp(-V(_range) / kT)
    cdf = np.cumsum(pdf)
    cdf /= max(cdf)

    uniformDraw = uniform(size=n_starts)

    densityChoices = [min(cdf[cdf >= draw]) for draw in uniformDraw]
    sampleIndex = [np.where(cdf == dc)[0][0] for dc in densityChoices]

    start_positions = _range[sampleIndex]

    return start_positions


def find_bin(x_pos, bins):

    for i, _bin in enumerate(bins):

        if x_pos > _bin[0] and x_pos <= _bin[1]:
            return i


# Returns lists of the AB and BA MFPTs
def compute_mfpt(trajectories, statesA, statesB, lags=[1, 10], stride=1):

    trajectory_to_analyze = trajectories

    source_sink_states = statesA + statesB

    all_AB, all_BA = [], []

    with Progress() as progress:

        lag_task = progress.add_task(description="Lags", total=len(lags))
        for j, lag in enumerate(lags):

            all_AB.append([])
            all_BA.append([])

            trajectories_task = progress.add_task(
                description="Trajectories", total=trajectory_to_analyze.shape[0]
            )
            for i, trajectory in enumerate(trajectory_to_analyze):
                AB, BA = [], []

                trajectory = trajectory[::stride]

                # This does a "sliding scale" so all points with the interval 'lag' are used
                trajectory_task = progress.add_task(
                    description="Trajectory", total=len(trajectory)
                )
                for start in range(lag):

                    # This is the state you were last in
                    last_in = -1
                    last_in_idx = 0

                    for idx, point in enumerate(trajectory[start::lag]):

                        if point in source_sink_states:

                            if last_in_idx > idx:
                                print(last_in_idx)
                                print(idx)
                                raise Exception

                            # If you just finished an A to B transit
                            if last_in in statesA and point in statesB:
                                AB.append(idx - last_in_idx)
                                last_in_idx = idx
                                last_in = point

                            # If this is the first time you're reaching a state
                            elif last_in == -1:
                                last_in = point
                                last_in_idx = idx

                            # If you just finished a B to A transit
                            elif last_in in statesB and point in statesA:
                                BA.append(idx - last_in_idx)
                                last_in_idx = idx
                                last_in = point

                        if idx % 100 == 0:
                            progress.update(trajectory_task, advance=100)

                progress.update(trajectory_task, visible=False)

                # The list comprehension deep-copies the list
                all_AB[j].append([x * lag * stride for x in AB])
                all_BA[j].append([x * lag * stride for x in BA])
                progress.update(trajectories_task, advance=1)

            progress.update(trajectories_task, visible=False)
            progress.update(lag_task, advance=1)

    return all_AB, all_BA


if __name__ == "__main__":

    # Define some physical simulation parameters
    kT = 1.0
    lambd = 0.01
    sigma = np.sqrt(kT * 2)
    barrier_height = 5

    # Generate a set of start points corresponding to the distribution
    n_starts = 4
    start_positions = generate_start_positions(
        n_starts=n_starts, barrier_height=barrier_height, kT=kT
    )

    # Define simulation length
    _dt = 0.0005
    max_time = 1000

    # This makes 130 bins, 2 from -16 to -inf and 16 to inf, and 128 of uniform width from
    n_fine_bins = 20
    bin_spacing = 32 / (n_fine_bins - 2)
    bin_coords = np.concatenate(
        [
            [[-np.inf, -16.0]],
            list(
                zip(
                    np.linspace(-16, 16 - bin_spacing, (n_fine_bins - 2)),
                    np.linspace(-16 + bin_spacing, 16, (n_fine_bins - 2)),
                )
            ),
            [[16.0, np.inf]],
        ]
    )

    # Ray is used for parallelizing the simulations.
    # It should automatically initiate with a reasonable amount of system resources.
    # If Ray fails, or you don't want to use it, just call generate_trajectory_l directly
    #    instead of generate_trajectory.remote, and remove the surrounding Ray code.
    ray.init()

    # The basins are set to infinity here, because we don't want to run with absorbing boundary conditions.
    continuous_trajectories = ray.get(
        [
            generate_trajectory.remote(
                _startx,
                max_t=max_time,
                sigma=sigma,
                lambd=lambd,
                kT=kT,
                dt=_dt,
                _barrier_height=barrier_height,
                subsample_result=1,
                BASINS=[[-np.inf, -np.inf], [np.inf, np.inf]],
            )
            for _startx in start_positions
        ]
    )

    # Discretize the continuous trajectories according to bin definitions
    discretized_trajectories = np.ndarray(shape=(n_starts, int(max_time / _dt)))
    discretized_trajectories[:] = [
        np.array([find_bin(_x, bin_coords) for _x in trajectory])
        for trajectory in continuous_trajectories
    ]

    lags = [1, 100]
    mfpts = compute_mfpt(
        discretized_trajectories,
        statesA=[0],
        statesB=[n_fine_bins - 1],
        lags=lags,
        stride=1,
    )

    for i, lag in enumerate(lags):
        print(f"Lag {lag}: ")

        AB_means = []
        BA_means = []
        for trajectory in range(n_starts):
            traj_AB_mean = np.mean(mfpts[0][i][trajectory])
            traj_BA_mean = np.mean(mfpts[1][i][trajectory])

            AB_means.append(traj_AB_mean)
            BA_means.append(traj_BA_mean)

        print(f"\t AB MFPT: {np.mean(AB_means):.2e} steps")
        print(f"\t BA MFPT: {np.mean(BA_means):.2e} steps")

    print("Saving generated trajectories...")
    np.savez(
        "generated_trajectories.npz",
        continuous_trajectories=continuous_trajectories,
        discretized_trajectories=discretized_trajectories,
        mfpts=mfpts,
        lags=lags,
        bin_coords=bin_coords,
        start_positions=start_positions,
        parameters=(_dt, max_time, sigma, lambd, kT, barrier_height),
    )
