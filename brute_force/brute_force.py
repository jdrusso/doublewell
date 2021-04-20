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

    expr = (5e7 * x) ** 10 - (3.5e8 * x) ** 2

    vmin = min(lambdify(x, expr)(np.linspace(0, 4e-8, 10000)))
    vmax = max(lambdify(x, expr)(np.linspace(-5e-9, 5e-9, 10000)))
    vrange = vmax - vmin

    normalized_expr = expr / vrange

    expr = normalized_expr * barrier_height

    return expr


# Starting from a position start_x, run a trajectory with timestep dt for
#    at most max_t steps.
# If the trajectory reaches either basin return the index of the basin it
#    went to and the list of x positions.
# (The _l is for _local instead of distributed via Ray)
def generate_trajectory_l(
    start_x,
    mass,
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

    variance = np.sqrt(2 * kT * dt / (mass * lambd))
    print(
        f"max: {max_t:.2e} | dt: {dt:.2e} | variance: {variance:.2e} | Barrier: {_barrier_height} ({_barrier_height/kT} kT) | basins: {BASINS} "
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
    _etas = rg.normal(0, variance, ETA_CHUNK)
    used_eta = 0

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
            _v = (-dVdx(current_x) + _noise) / (lambd * mass)
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

    _range = np.arange(-12, 12, 0.01) * 1e-9
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
def compute_mfpt(trajectories, statesA, statesB, stride=1):

    lags = [1]

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
    mass = 1.55e-25  # 100 Daltons
    kT = 4.142e-21  # Corresponds to 300K
    lambd = 2.494e13  # 24.95 ps^-1
    sigma = np.sqrt(kT * 2)
    barrier_height = 10*kT

    # Generate a set of start points corresponding to the distribution
    n_starts = 10
    start_positions = generate_start_positions(
        n_starts=n_starts, barrier_height=barrier_height, kT=kT
    )

    # Define simulation length
    _dt = 3e-12
    max_time = _dt*10000000000
    subsample = 1000
    # max_time = _dt*1e7
    # subsample = 1

    n_fine_bins = 20
    bin_spacing = 4e-8 / (n_fine_bins - 2)

    boundary = 2e-8
    bin_coords = np.concatenate(
        [
            [[-np.inf, -boundary]],
            list(
                zip(
                    np.linspace(-boundary, boundary - bin_spacing, (n_fine_bins - 2)),
                    np.linspace(-boundary + bin_spacing, boundary, (n_fine_bins - 2)),
                )
            ),
            [[boundary, np.inf]],
        ]
    )

    # Ray is used for parallelizing the simulations.
    # It should automatically initiate with a reasonable amount of system resources.
    # If Ray fails, or you don't want to use it, just call generate_trajectory_l directly
    #    instead of generate_trajectory.remote, and remove the surrounding Ray code.
    ray.init()
    # For running ray with a cluster (surprisingly easy, try it!)
    # ray.init(address='auto', _redis_password='set some password here', dashboard_host='127.0.0.1')

    # The basins are set to infinity here, because we don't want to run with absorbing boundary conditions.
    continuous_trajectories = ray.get(
        [
            generate_trajectory.remote(
                _startx,
                max_t=max_time,
                mass=mass,
                sigma=sigma,
                lambd=lambd,
                kT=kT,
                dt=_dt,
                _barrier_height=barrier_height,
                subsample_result=subsample,
                BASINS=[[-np.inf, -np.inf], [np.inf, np.inf]],
            )
            for _startx in start_positions
        ]
    )

    # Discretize the continuous trajectories according to bin definitions
    discretized_trajectories = np.ndarray(shape=(n_starts, int(max_time / _dt) // subsample))
    discretized_trajectories[:] = [
        np.array([find_bin(_x, bin_coords) for _x in trajectory])
        for trajectory in continuous_trajectories
    ]

    lags = [1]
    mfpts = compute_mfpt(
        discretized_trajectories,
        statesA=[0],
        statesB=[n_fine_bins - 1],
        stride=1,
    )

    print(mfpts)
    for i, lag in enumerate(lags):
        print(f"Lag {lag}: ")

        AB_means = []
        BA_means = []
        for trajectory in range(n_starts):
            print(f"Trajectory {trajectory}")
            traj_AB_mean = np.nanmean(mfpts[0][i][trajectory])
            traj_BA_mean = np.nanmean(mfpts[1][i][trajectory])

            print(traj_AB_mean)
            print(traj_BA_mean)

            AB_means.append(traj_AB_mean)
            BA_means.append(traj_BA_mean)

        print(AB_means)
        print(BA_means)
        print(f"\t AB MFPT: {np.nanmean(AB_means):.2e} steps ")
        print(f"\t BA MFPT: {np.nanmean(BA_means):.2e} steps ")

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
