import numpy as np
from astropy import constants as c, units as u
import csv
import os
from nbodykit.lab import cosmology, LogNormalCatalog
from nbodykit.utils import GatherArray
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pandas import Timestamp
import libPJC as my

start_time = Timestamp.now()

# Enable LaTeX text rendering
plt.rc("text", usetex=True)


def mock(
    L_box,
    N_grid,
    axionFrac,
    redshift,
    start_time,
    dirName_plot,
    mpi_on=False,
    comm=None,
    rank=None,
):
    """
    Generates mock density and velocity fields using a log-normal catalog simulation.

    Parameters
    ----------
    L_box : astropy.Quantity
        Box size (e.g., 100*u.Mpc)
    N_grid : int
        Number of grid points per dimension
    axionFrac : str
        Axion fraction identifier (e.g., "010" for 10%)
    redshift : float
        Simulation redshift
    start_time : pandas.Timestamp
        Timestamp for tracking runtime
    dirName_plot : str
        Directory for saving plots
    mpi_on : bool, optional
        Enable MPI parallelism (default: False)
    comm : MPI.Comm, optional
        MPI communicator (required if mpi_on=True)
    rank : int, optional
        MPI rank (required if mpi_on=True)

    Notes
    -----
    - MPI Dependency: If mpi_on=True, comm and rank must be provided
    - Unit Handling: All inputs with units must be astropy.Quantity objects
    - File Overwrite: Overwrites existing parameter files in output directory

    Output Files
    ------------
    - Density field: mocks.npy
    - Velocity fields: vel_x.npy, vel_y.npy, vel_z.npy
    - Parameters: my_params.csv
    Saved in directory: forestsNBK{axionFrac}{int(L_box.value)}
    """
    if not mpi_on or rank == 0:
        print(f"Starting at {start_time}")

    #################################################################################### FILE NAMES
    dirName = f"forestsNBK{axionFrac}{int(L_box.value)}"
    file_param = f"{dirName}/my_params.csv"  # f"{dirName}/my_params.csv"
    file_mocks = {
        "dens": f"{dirName}/mocks.npy",
        "vel_x": f"{dirName}/vel_x.npy",
        "vel_y": f"{dirName}/vel_y.npy",
        "vel_z": f"{dirName}/vel_z.npy",
    }

    #################################################################################### PARAMETERS OF SIMULATION

    # Cosmo Parameters
    cosmo = cosmology.Planck15
    H_0 = 100.0 * cosmo.h * u.km / u.s / u.Mpc  # in km * s^-1 * Mpc^-1 * h
    H_z = H_0 * np.sqrt(
        cosmo.Om0 * (1 + redshift) ** 3 + cosmo.Ode0
    )  # in km * s^-1 * Mpc^-1 * h (Bird_2023 p.4)
    h_z = H_z / (100.0 * u.km / u.s / u.Mpc)  # unitless
    rho_crit = (3.0 * H_z**2.0 / (8.0 * np.pi * c.G)).to(
        u.M_sun / (u.kpc) ** 3
    )  # in M_sun/h / (kpc/h)^3

    # Parameters for mock
    b1 = 2  # lognormal transformation: delta(x) = exp(-sigma^2 + b_L * delta_L(x)) - 1     with b_L = b1-1 --> b1=2!!
    nbar = 3 * (N_grid / L_box.value) ** 3  # number density

    if not mpi_on or rank == 0:
        print(f"L_box: {L_box}, N_grid: {N_grid}")
    titlePlots = f"Redshift: {redshift}, box size: {L_box:.2f}, Grid points: {N_grid}, tracer density: {nbar:.2f}"

    ########## CONSIDER taking only the baryonic fraction
    mean_density = (
        cosmo.Ob0 * rho_crit
    )  # Mean matter density                                       # in M_sun/h / (kpc/h)^3
    total_mass = (
        mean_density * (L_box.to(u.kpc)) ** 3
    )  # in M_sun/h ???????????????????????
    # if not mpi_on or rank == 0: print(f"\nmean density: {mean_density:.2f}\n")
    params_axionCAMB = my.getParams(
        f"Archive/PJC_{axionFrac}_params.ini", foresting=True
    )
    if not mpi_on or rank == 0:
        print("my H_0: ", H_0, "axiCAMB's: ", params_axionCAMB["hubble"])
    axfrac = params_axionCAMB["axfrac"]
    m_ax = params_axionCAMB["m_ax"]

    #################################################################################### WRITING PARAMETERS TO FILE
    if os.path.dirname(file_param):
        os.makedirs(os.path.dirname(file_param), exist_ok=True)
    with open(file_param, "w", newline="") as p:
        writer = csv.writer(p, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["N_grid", N_grid])
        writer.writerow(["L_box", L_box.value])  # Mpc/h
        writer.writerow(["nbar", np.round(nbar, 2)])
        writer.writerow(["redshift", redshift])
        writer.writerow(["H_z", (H_z).value])  # in km * s^-1 * Mpc^-1 * h
        writer.writerow(
            ["mean_density", (mean_density).value]
        )  # in M_sun/h / (kpc/h)^3
        writer.writerow(["b1", b1])
        writer.writerow(["axfrac", axfrac])
        writer.writerow(["m_ax", m_ax])

    #################################################################################### GENERATE CATALOG
    if not mpi_on or rank == 0:
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: done importing, generating catalog"
        )
    # Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    # NBK_Tk = cosmology.power.transfers.EisensteinHu(cosmo, redshift)

    data_axionCAMB = np.loadtxt(f"Archive/PJC_{axionFrac}_matterpower.dat")
    k_axion = data_axionCAMB[:, 0]
    Pk_axion = data_axionCAMB[:, 1]

    # for PJC_010_transfer_out.dat:
    # Tkc = data_axionCAMB[:, 1]
    # Tkb = data_axionCAMB[:, 2]
    # Tkax = data_axionCAMB[:, 6]
    # Tktot = data_axionCAMB[:, 8]
    # Pk_axion = Plin(k)/NBK_Tk(k)**2 * Tktot**2
    def Pk_callable(X_theirs):
        X_ours = k_axion
        Y_ours = Pk_axion
        interp = interp1d(X_ours, Y_ours, bounds_error=False)
        Y_theirs = interp(X_theirs)
        return Y_theirs

    mock = LogNormalCatalog(
        Plin=Pk_callable,
        nbar=nbar,  # needed number density
        BoxSize=L_box.value,
        Nmesh=N_grid,
        bias=b1,  # lognormal transformation: delta(x) = exp(-sigma^2 + b_L * delta_L(x)) - 1     with b_L = b1-1 --> b1=2!!
        seed=42,
        cosmo=cosmo,
        redshift=redshift,
        comm=comm,
    )

    #################################################################################### CREATE DENSITY MESH
    if not mpi_on or rank == 0:
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: done with catalog, generating mesh"
        )
    compensated = False
    delta_mesh = mock.to_mesh(
        # Nmesh&BoxSize are already stored in the mock meta
        resampler="cic",  # https://nbodykit.readthedocs.io/en/latest/cookbook/interlacing.html
        interlaced=False,  # ibid
        compensated=compensated,  # whether to apply a Fourier-space transfer function to account for the effects of the gridding + aliasing
        position="Position",
        dtype="f4",
    )

    one_plus_delta = delta_mesh.preview(Nmesh=N_grid, root=0)  # .paint(mode='real')
    if not mpi_on or rank == 0:
        print(
            f"1+delta: mean {one_plus_delta.mean():.2f} min {one_plus_delta.min():.5f}, max {one_plus_delta.max():.2f}"
        )
    print("rank: ", rank, "mesh: ", one_plus_delta.shape)

    if compensated:
        one_plus_delta_temp = one_plus_delta - np.min(
            [0.0, np.min(one_plus_delta.value)]
        )
        one_plus_delta = one_plus_delta_temp / np.mean(one_plus_delta_temp)
        # for compensated = True to work, we need to shift and renormalize
        if not mpi_on or rank == 0:
            print(
                f"1+delta: mean {one_plus_delta.mean():.2f} min {one_plus_delta.min():.5f}, max {one_plus_delta.max():.2f}"
            )

    den_field = one_plus_delta * mean_density
    # den_field = np.clip(np.log(1+one_plus_delta), 0, None) * mean_density

    #################################################################################### CREATE VELOCITY MESH
    mock["Velocity_x"] = mock["Velocity"][:, 0]
    mock["Velocity_y"] = mock["Velocity"][:, 1]
    mock["Velocity_z"] = mock["Velocity"][:, 2]
    vel_x_mesh = mock.to_mesh(
        position="Position",
        value="Velocity_x",
        resampler="tsc",
        compensated=False,
        dtype="f4",
    )
    vel_y_mesh = mock.to_mesh(
        position="Position",
        value="Velocity_y",
        resampler="tsc",
        compensated=False,
        dtype="f4",
    )
    vel_z_mesh = mock.to_mesh(
        position="Position",
        value="Velocity_z",
        resampler="tsc",
        compensated=False,
        dtype="f4",
    )
    vel_x = (
        vel_x_mesh.preview(Nmesh=N_grid, root=0) * u.km / u.s
    )  # .compute().value * u.km / u.s # compute() turns to realF<ield, .value to np.array
    vel_y = (
        vel_y_mesh.preview(Nmesh=N_grid, root=0) * u.km / u.s
    )  # .compute().value * u.km / u.s
    vel_z = (
        vel_z_mesh.preview(Nmesh=N_grid, root=0) * u.km / u.s
    )  # .compute().value * u.km / u.s

    # vel_field = np.stack([vel_x, vel_y, vel_z], axis=0) # shape: (3, N_grid, N_grid, N_grid)

    if not mpi_on or rank == 0:
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: saving"
        )
    if mpi_on:
        # Only rank 0 saves
        if rank == 0:
            # Verify final shape matches expectations
            if den_field.shape != (N_grid, N_grid, N_grid):
                print(
                    f"Warning: Gathered shape {den_field.shape} differs from expected {(N_grid, N_grid, N_grid)}"
                )
            np.save(file_mocks["dens"], den_field.value)
            np.save(file_mocks["vel_x"], vel_x.value)
            np.save(file_mocks["vel_y"], vel_y.value)
            np.save(file_mocks["vel_z"], vel_z.value)
    else:
        # Serial mode
        np.save(file_mocks["dens"], den_field.value)
        np.save(file_mocks["vel_x"], vel_x.value)
        np.save(file_mocks["vel_y"], vel_y.value)
        np.save(file_mocks["vel_z"], vel_z.value)


def foresting(
    mpi_on,
    mocked,
    axionFrac,
    L_box,
    N_los,
    HI,
    start_time,
    dirName_plot,
    extraEnding="",
    comm=None,
    rank=None,
    size=None,
    plotose=False,
):
    """
    Generates synthetic Lyman-alpha forest spectra from density/velocity fields.

    Parameters
    ----------
    mpi_on : bool
        Enable MPI parallelism
    mocked : bool
        Use mock data (True) or AxioNyx simulation data (False)
    axionFrac : str
        Axion fraction identifier
    L_box : astropy.Quantity
        Box size (must match mock if mocked=True)
    N_los : int
        Number of lines of sight (per dimension)
    HI : bool
        Include neutral hydrogen physics (reserved for future use)
    start_time : pandas.Timestamp
        Runtime tracker
    dirName_plot : str
        Plot output directory
    extraEnding : str, optional
        Suffix for output directory (e.g., "_test")
    comm : MPI.Comm, optional
        MPI communicator (required if mpi_on=True)
    rank : int, optional
        MPI rank (required if mpi_on=True)
    size : int, optional
        Number of MPI processes (required if mpi_on=True)
    plotose : bool, optional
        Generate test plots (default: False)

    Notes
    -----
    - MPI Requirement: If mpi_on=True, comm, rank and size are mandatory
    - Data Consistency: L_box and axionFrac must match pre-generated data
    - Directory Structure: Assumes input files exist in dirName/

    Output Files
    ------------
    Saved in subdirectory download/:
    - Spectra: lambda_range.csv, forest_{x,y,z}.csv
    - Density: x.csv, den_{x,y,z}.csv
    Input directory depends on mocked flag:
    - Mock data: forestsNBK{axionFrac}{int(L_box.value)}
    - AxioNyx data: fromAxioNyx0108/forestsAxioNyx{axionFrac}
    """
    # overall or only HI??? (cf Viel_2004 p.5)
    if mpi_on:
        from mpi4py import MPI

        if not isinstance(comm, MPI.Intracomm):
            raise ValueError(f"Invalid communicator of type {type(comm)}")

        try:
            if not MPI.Is_initialized():
                MPI.Init()
            # Optional: MPI.COMM_WORLD.Barrier()  # wait for other ranks if needed
        except Exception as e:
            print("MPI init workaround failed:", e)

    #################################################################################### FILE NAMES
    if mocked:
        dirName = f"forestsNBK{axionFrac}{int(L_box.value)}{extraEnding}"
        file_mocks = {
            "dens": f"{dirName}/mocks.npy",
            "vel_x": f"{dirName}/vel_x.npy",
            "vel_y": f"{dirName}/vel_y.npy",
            "vel_z": f"{dirName}/vel_z.npy",
        }
    else:
        dirName = f"fromAxioNyx0108/forestsAxioNyx{axionFrac}"  # f"512 Snapshots{axionFrac}{int(L_box.value)}" #"forestsAxioNyx"
        file_mocks = {
            "dens": f"{dirName}/mock.npy",
            "vel_x": f"{dirName}/vel_x.npy",
            "vel_y": f"{dirName}/vel_y.npy",
            "vel_z": f"{dirName}/vel_z.npy",
        }

    file_param = f"{dirName}/my_params.csv"  # f"{dirName}/my_params.csv"
    files = {
        "specs": [
            f"{dirName}/download/lambda_range.csv",
            f"{dirName}/download/forest_x.csv",
            f"{dirName}/download/forest_y.csv",
            f"{dirName}/download/forest_z.csv",
        ],
        "dens": [
            f"{dirName}/download/x.csv",
            f"{dirName}/download/den_x.csv",
            f"{dirName}/download/den_y.csv",
            f"{dirName}/download/den_z.csv",
        ],
    }

    #################################################################################### GET MOCKING PARAMS
    if not mpi_on or rank == 0:
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: done importing, reading files"
        )
    params = my.getParams(file_param, foresting=True)
    # if not mpi_on or rank == 0: print(params)

    N_grid = params["N_grid"]
    L_box = params["L_box"] * u.Mpc  # Mpc/h
    redshift = params["redshift"]
    axfrac = params["axfrac"]
    m_ax = params["m_ax"]

    # hard coded FW cosmo params
    FW_OmM = 0.31  # (all CDM)
    FW_OmL = 0.69
    FW_h = 0.675

    if "nbar" in params.keys():
        nbar = params["nbar"]
    else:
        nbar = 3 * (N_grid / L_box.value) ** 3

    if "H_z" in params.keys():
        H_z = params["H_z"] * u.km / u.s / u.Mpc  # in km/s * h/Mpc
    else:
        H_0 = 100.0 * FW_h * u.km / u.s / u.Mpc  # in km * s^-1 * Mpc^-1 * h
        H_z = H_0 * np.sqrt(
            FW_OmM * (1 + redshift) ** 3 + FW_OmL
        )  # in km * s^-1 * Mpc^-1 * h (Bird_2023 p.4)

    if "mean_density" in params.keys():
        mean_density = params["mean_density"] * u.M_sun / (u.kpc) ** 3
    else:
        rho_crit = (3.0 * H_z**2.0 / (8.0 * np.pi * c.G)).to(
            u.M_sun / (u.kpc) ** 3
        )  # in M_sun/h / (kpc/h)^3
        mean_density = FW_OmM * rho_crit

    if "b1" in params.keys():
        b1 = params["b1"]
    else:
        b1 = 2

    #################################################################################### PARAMETERS OF SIMULATION
    # Parameters for spectra
    line = 1215.67 * u.Angstrom  # Angstroms
    einstein_Acoeff = 6.265e8 / u.s  # s^-1
    delta_z = (
        H_z * L_box / c.c.to(u.km / u.s)
    ).value  # delta_z=0.0296 for L_box=11,1Mpc/h and z=3                         # cf Croft_1997 p.7
    line_obs = my.get_lambda_from_d(
        L_box, H_z, line
    )  # line * (delta_z + 1)                   # Angstroms
    lambda_int = line_obs - line  # Angstroms
    lambda_margin = (
        0.0 * u.Angstrom
    )  # 5                                                            # Angstroms

    obs_res = 6 * u.km / u.s  # km/s/pixel
    # 6 *u.km/u.s                                                                                               # cf Boera_2019 HIRES
    # 69 * u.km/u.s                                                                                             # cf Palanque-Delabrouille_2013 p6 for BOSS
    # r_s = 1.5 Mpc/h                         # --> 1.5 * H_z  = 85 km/s                                        # cf Croft_1997 p.11 AT REDSHIFT 3!!!
    # obs_res *= 0.1 # subsampling pixels 10× finer than instrument resolution
    res = (obs_res * line / c.c).to(
        u.Angstrom
    )  # ∆v = c∆λ/λ = 69km/s                         # Angstroms/pixel   # cf Palanque-Delabrouille_2013 p6

    N_spectrum = int(((lambda_int + lambda_margin * 2) / res).value)

    N_spectrum = np.max(
        (N_spectrum - N_spectrum % N_grid + N_grid, 3 * N_grid)
    )  # + N_spectrum//N_grid
    # make N_spectrum a multiple of N_grid, keeping minimal resolution, need the N_spectrum//N_grid addend too, because otherwise the last bin of every set is underfilled
    N_spectrum = N_grid  # !!!!! will be undersampled anyway
    res = (lambda_int + lambda_margin * 2) / N_spectrum

    """N_spectrum = N_grid
    res = (lambda_int+lambda_margin*2)/N_spectrum          # .01 * u.Angstrom                       # Angstroms/pixel
    obs_res = (res * c.c/line).to(u.km/u.s)                                                         # km/s/pixel
    """
    if not mpi_on or rank == 0:
        print(
            f"L_box: {L_box}, N_grid: {N_grid}, N_los: {N_los}, N_spectrum: {N_spectrum}, delta_z: {delta_z:.4f}, lambda_int: {lambda_int:2f}"
        )
    titlePlots = f"Redshift: {redshift}, box size: {L_box:.2f}, Grid points: {N_grid}, tracer density: {nbar:.2f}, spectral resolution: {N_spectrum}, Ax frac: {axfrac}, Ax m: {m_ax}"

    total_mass = (
        mean_density * (L_box.to(u.kpc)) ** 3
    )  # in M_sun/h ???????????????????????
    # if not mpi_on or rank == 0: print(f"\nmean density: {mean_density:.2f}\n")

    # Params for IGM
    # temperature field Bird_2019 p.7
    gamma = 1.6  # T-ρ relation exponent McQuinn_2016 p.16
    Temp_0 = (
        1e4 * u.K * (1 + redshift)
    )  # Temperature at mean density McQuinn_2016 p.16              # in K
    # optical depth approximation Viel_2004 p5
    tau_0 = 0.305
    # optical depth at mean density depends on redshift, baryon density,
    # temperature at mean density, Hubble constant and photoionisation rate
    tau_s = 1.7 * gamma
    # slope of optical depth-density relation 2.7-0.7 * gamma (cf Viel_2004 p5)

    if not mpi_on or rank == 0:
        #################################################################################### WRITING PARAMETERS TO FILE
        if not os.path.exists(file_param):  # rewrite the param file
            raise FileNotFoundError(f"Data file {file_param} not found!")
        with open(file_param, "w") as p:
            writer = csv.writer(
                p, delimiter=" "
            )  # , quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(params.keys())):
                writer.writerow([list(params.keys())[i], list(params.values())[i]])
            writer.writerow(["N_spectrum", N_spectrum])
            writer.writerow(["res", res.value])  # Angstrom/pixel
            writer.writerow(["res_obs", obs_res.value])  # km/s/pixel
            writer.writerow(["N_los", N_los])
            writer.writerow(["lambda_start", (line - lambda_margin).value])  # Angstrom
            writer.writerow(
                ["lambda_end", (line_obs + lambda_margin).value]
            )  # Angstrom
            writer.writerow(["lambda_margin", lambda_margin.value])  # Angstrom
            writer.writerow(["gamma", gamma])
            writer.writerow(["Temp_0", Temp_0.value])  # K
            writer.writerow(["tau_0", tau_0])
            writer.writerow(["tau_s", tau_s])

    #################################################################################### COMPUTING SPECTRA & WRITING SPECTRA FILES
    if not mpi_on or rank == 0:
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing spectra"
        )

    my_dir, start, end = my.organizeAbsorption(mpi_on, size, rank, N_los)

    print("rank: ", rank, "my dir: ", my_dir, "start: ", start, "end: ", end)
    if mpi_on:
        lambda_range, spectra = my.computeAbsorption(
            my_dir,
            start,
            end,
            N_grid,
            N_los,
            N_spectrum,
            L_box,
            res,
            file_mocks,
            HI,
            mocked,
            redshift,
            H_z,
            line,
            line_obs,
            lambda_margin,
            einstein_Acoeff,
            Temp_0,
            gamma,
            tau_0,
            tau_s,
            mpi_on,
            rank,
            dirName_out=f"{dirName}_MPI",
        )

        my.write_spectra_file(
            spectra,
            lambda_range,
            files,
            mpi_on,
            comm,
            size,
            rank,
            N_los,
            N_spectrum,
            end,
            start,
        )

        my.write_den_file(
            my_dir, N_los, N_grid, L_box, H_z, line, file_mocks, files, mpi_on, rank
        )

    else:
        final_spectra = []
        for i in my_dir:
            lambda_range, spectra = my.computeAbsorption(
                i,
                start,
                end,
                N_grid,
                N_los,
                N_spectrum,
                L_box,
                res,
                file_mocks,
                HI,
                mocked,
                redshift,
                H_z,
                line,
                line_obs,
                lambda_margin,
                einstein_Acoeff,
                Temp_0,
                gamma,
                tau_0,
                tau_s,
                mpi_on,
                rank,
                dirName_out=dirName,
            )
            my.write_den_file(
                i, N_los, N_grid, L_box, H_z, line, file_mocks, files, mpi_on, rank
            )
            final_spectra.append(spectra)
        my.write_spectra_file(
            np.asarray(final_spectra),
            lambda_range,
            files,
            mpi_on,
            comm,
            size,
            rank,
            N_los,
            N_spectrum,
            end,
            start,
        )

    if (not mpi_on or rank == 0) and (plotose):
        #################################################################################### TESTING
        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: testing spectra"
        )
        my.testSpectra(
            file_param, files, titlePlots, dirName_plot, foresting=True, tot=True
        )

        print(
            f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: testing slices"
        )
        my.plotMock(
            L_box,
            N_grid,
            files,
            mocked,
            Temp_0,
            gamma,
            dirName_plot,
            titlePlots,
            savefig=True,
        )


def analyze(
    axionFrac,
    boxSize,
    mocked,
    J,
    Q,
    order,
    log_binning,
    fit,
    savefigs,
    start_time,
    dirName_plot,
    extraEnding="",
    dirName_in=None,
    dirName_out=None,
    csv_on=True,
    filter_on=False,
    window_on=False,
):
    """
    Computes statistical metrics (PDF, power spectra, WST) from spectra/density fields.

    Parameters
    ----------
    axionFrac : str
        Axion fraction identifier
    boxSize : int
        Box size (e.g., 100 for 100 Mpc/h)
    mocked : bool
        Use mock data (True) or AxioNyx data (False)
    J : int
        WST scale parameter (number of octaves)
    Q : int
        WST quality factor (wavelets per octave)
    order : int
        WST order (1 or 2)
    log_binning : bool
        Use logarithmic k-binning for power spectra
    fit : bool
        Fit power spectra with a model
    savefigs : list[bool]
        Flags to save plots: [PS1D, PS3D, WST, WST_summary]
    start_time : pandas.Timestamp
        Runtime tracker
    dirName_plot : str
        Plot output directory
    extraEnding : str, optional
        Suffix for output directory (e.g., "_highres")
    dirName_in : str, optional
        Override input directory
    dirName_out : str, optional
        Override output directory
    csv_on : bool, optional
        Input data as CSV (default: True)
    filter_on : bool, optional
        Apply Gaussian filter to spectra (default: False)
    window_on : bool, optional
        Apply window function to power spectra (default: False)

    Notes
    -----
    - Input Directory: Defaults to forestsNBK... or forestsAxioNyx... if dirName_in=None
    - Output Files: Overwrites existing statistical files
    - Binning: nbins = N_grid//2 for power spectra

    Output Files
    ------------
    Saved in output directory (default: statToolNBK{extraEnding} or statToolAxioNYX{extraEnding}):
    - Power spectra: kPS_{axionFrac}{boxSize}.npy, powSpectrum{axionFrac}{boxSize}.npy
    - WST: kWST_{axionFrac}{boxSize}.npy, WST{axionFrac}{boxSize}.npy
    - PDF: pdf{axionFrac}{boxSize}.npy
    """
    if dirName_out == None:
        if mocked:
            dirName_out = f"statToolNBK{extraEnding}"
        else:
            dirName_out = f"statToolAxioNYX{extraEnding}"
    os.makedirs(dirName_out, exist_ok=True)

    # VARY FILENAMES FOR FISHER MATRIX
    # fiducial cosmology
    if dirName_in == None:
        if mocked:
            dirName_in = f"forestsNBK{axionFrac}{extraEnding}"
        else:
            dirName_in = f"forestsAxioNyx{axionFrac}{extraEnding}"

    file_param = f"{dirName_in}/my_params.csv"

    files = {
        "specs": [
            f"{dirName_in}/lambda_range",
            f"{dirName_in}/forest_x",
            f"{dirName_in}/forest_y",
            f"{dirName_in}/forest_z",
        ],
        "dens": [
            f"{dirName_in}/x",
            f"{dirName_in}/den_x",
            f"{dirName_in}/den_y",
            f"{dirName_in}/den_z",
        ],
    }

    extension = ".csv" if csv_on else ".npy"
    for key in files:
        files[key] = [f"{path}{extension}" for path in files[key]]

    k_PS_file = f"{dirName_out}/kPS_{axionFrac}{boxSize}"
    k_WST_file = f"{dirName_out}/kWST_{axionFrac}{boxSize}"
    powSpectr_file = f"{dirName_out}/powSpectrum{axionFrac}{boxSize}"
    WST_file = f"{dirName_out}/WST{axionFrac}{boxSize}"
    pdf_file = f"{dirName_out}/pdf{axionFrac}{boxSize}"
    """
    # variation in Axion mass m_a
    dirName_in = "forests_m" 
    powSpectr_file = f"{dirName_out}/powSpectrum_m" 
    WST_file = f"{dirName_out}/WST_m.csv" 

    #variation in Axion fraction x_a
    dirName_in = "forests_x" 
    powSpectr_file = f"{dirName_out}/powSpectrum_x" 
    WST_file = f"{dirName_out}/WST_x.csv"
    """

    #################################################################################### GETTING PARAMS
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: done importing, reading files"
    )
    params = my.getParams(file_param, foresting=False)
    # print(params)

    N_grid = params["N_grid"]
    N_spectrum = params["N_spectrum"]
    N_los = params["N_los"]
    L_box = params["L_box"]  # Mpc/h
    redshift = params["redshift"]
    mean_density = params["mean_density"]
    H_z = params["H_z"]  # in km/s * h/Mpc
    res = params["res"]  # in Angstroms/pixel
    line = params["lambda_start"] + params["lambda_margin"]  # in Angstroms
    line_obs = params["lambda_end"] - params["lambda_margin"]  # in Angstroms
    delta_lambda = (
        params["lambda_end"] - params["lambda_start"] - params["lambda_margin"] * 2
    )
    res_obs = params["res_obs"]  # in km/s/pixel
    res_obs = params["res_obs"]  # in km/s/pixel
    res_obs = params["res_obs"]  # in km/s/pixel

    #################################################################################### BINNING&SAMPLING
    # Wavelet parameters

    nbins = N_grid // 2  # divided by 2 because k sampling for real FFT will divide by 2
    nsamples = int(N_spectrum / (2**J))  # ????

    k_min = (
        2 * np.pi / (L_box)
    )  # * H_z)                              # cf. Arinyio-i-Prats p.12
    # and https://nbodykit.readthedocs.io/en/latest/cookbook/interlacing.html
    # np.pi * params["lambda_start"]/(c.c.to(u.km/u.s).value * delta_lambda/N_spectrum)
    # 0.06                          # h/Mpc         # cf Nakashima_2025
    # .1e-3                         # s/km          # cf Palanque-Delabrouille_2013 p.6
    k_max = (
        nbins * np.pi / (L_box)
    )  # *H_z)                              # Nyquist frequency cf. Arinyio-i-Prats p.12
    # and https://nbodykit.readthedocs.io/en/latest/cookbook/interlacing.html
    # np.pi * params["lambda_start"]/(c.c.to(u.km/u.s).value * delta_lambda/N_spectrum) # Palanque_Delabrouille_2013 p.6
    # N_spectrum * L / (4*np.pi*(N_grid-1))
    # 32. # h/Mpc cf Nakashima_2025
    # .02 # s/km cf Palanque-Delabrouille_2013 p.6
    # print("kmin: ", k_min, "kmax: ", k_max)

    R = delta_lambda / res  # remember: res = delta_lambda/N_spectrum

    # Gaussian filters (given in index/pixel units)
    sigma_filter_ind_spec_den = None  # .5 * nbins/N_spectrum # N_spectrum/nbins                  # remember! we are oversampling the den along the LOS for the 1D powSpectr...

    if not filter_on:
        sigma_filter_ind_spec_forests = None
    else:
        sigma_filter_ind_spec_forests = nbins / N_spectrum
    # res / delta_lambda * N_spectrum # ==1             # ---> add this only if testing resilience towards gaussian noise
    # (res_obs * N_spectrum / c.c.to(u.km/u.s).value) # or 2*np.pi/(...) ??
    # L_box/N_grid * H_z was on 11.06, if you think about it, the dependency is similar... worried about N_grid antiprop...

    # Window functions (given in km/s) Niemeyer p.70, Dodelson p.345 (for factor 1 / np.sqrt(8*np.log(2)))
    # Weinberg_2003, Yeche_2017, Viel_2013, Tohfa_2024, Rogers_2021, Peirani_2022, Kaiser&Peacock_1991, Palanque_Delabrouille_2013, Gaikwad_2021

    if not window_on:
        sigma_Window_PS = None
    else:
        sigma_Window_PS = (
            2
            * np.pi
            / my.convert_k_to_velocity_space(2 * np.pi * nbins / L_box, redshift, H_z)
        ) / np.sqrt(
            8 * np.log(2)
        )  # FWHM-->sigma
    # N_spectrum/nbins # nbins/N_spectrum                       # this should make sense? I compute more spectral points than I have grid resolution and filter excesses out after

    sigma_Window_PS3D = None  # 2*np.pi/((k_max-k_min)/(2*nbins))
    sigma_Window_WST = None  # 2*np.pi / (my.convert_k_to_velocity_space(k_min, redshift, H_z)) / nsamples / np.sqrt(8*np.log(2))
    # nsamples/nbins # nbins/nsamples
    # !!!!!! i have no use for a window function on the WST

    print(
        "sigma spec: ",
        sigma_filter_ind_spec_forests,
        "sigma den: ",
        sigma_filter_ind_spec_den,
    )
    print("sigma PS: ", sigma_Window_PS, "sigma WST: ", sigma_Window_WST)

    #################################################################################### PLOTTING STUFF
    colors = ["red", "green", "blue", "orange", "limegreen", "deepskyblue"]
    titleGeneral = f"L_box: {L_box:.2f}, N_grid: {N_grid}, N_los: {N_los}, N_spectrum: {N_spectrum},"
    losToPlot = np.min([N_los, 5])

    titlePlots = {
        "mat/flux": f"{titleGeneral} Filter mat: {my.format_value(sigma_filter_ind_spec_den, '.2f')}, Filter flux: {my.format_value(sigma_filter_ind_spec_forests)}",
        "PS mat": f"{titleGeneral} Filter: {my.format_value(sigma_filter_ind_spec_den, '.2f')}, Filter PS: {my.format_value(sigma_Window_PS)}",
        "PS flux": f"{titleGeneral} Filter: {my.format_value(sigma_filter_ind_spec_forests, '.2f')}, Filter PS: {my.format_value(sigma_Window_PS)}",
        "WST mat": f"{titleGeneral} Filter: {my.format_value(sigma_filter_ind_spec_den, '.2f')}, Filter WST: {my.format_value(sigma_Window_WST)}",
        "WST flux": f"{titleGeneral} Filter: {my.format_value(sigma_filter_ind_spec_forests, '.2f')}, Filter WST: {my.format_value(sigma_Window_WST)}",
    }

    #################################################################################### GETTING DATA
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: getting data"
    )
    forests, lambda_range, densities, x_for_den = my.getData(
        files, N_spectrum, N_grid, N_los, csv_on
    )  # forests

    # Undersampling
    undersampling_mask = len(lambda_range) // (
        len(x_for_den)
    )  # *2)              # divided by 2 because k sampling for real FFT will divide by 2
    forests = forests[:, :, :, ::undersampling_mask]
    lambda_range = lambda_range[::undersampling_mask]
    if len(lambda_range) == (
        len(x_for_den) + 1
    ):  # * 2) + 1:                       # divided by 2 because k sampling for real FFT will divide by 2
        forests = forests[:, :, :, :-1]
        lambda_range = lambda_range[:-1]
    elif len(lambda_range) != (
        len(x_for_den)
    ):  # *2):                            # divided by 2 because k sampling for real FFT will divide by 2
        raise ValueError(
            "len(lambda_range)!=(len(x_for_den) * 2)",
            len(lambda_range),
            (len(x_for_den) * 2),
        )
    N_spectrum = len(lambda_range)
    print(N_spectrum)
    nsamples = int(N_spectrum / (2**J))

    my.plotForestsDens(
        lambda_range,
        forests[:, :30, :30, :],
        x_for_den,
        densities,
        line,
        line_obs,  # forests[:,:30,:30,:] #----------WARNING! :30 slice because the NBK was killed before finish on 3007
        titlePlots["mat/flux"],
        dirName_plot,
        filename=f"spectra",
        tot=True,
        losToPlot=1,
    )

    if sigma_filter_ind_spec_forests != None:
        forests = my.gaussian_filter1d(
            forests, sigma=sigma_filter_ind_spec_forests, axis=-1, truncate=1.0
        )  # shape (3, N_los, N_los, N_spectrum/2)

    if sigma_filter_ind_spec_den != None:
        densities = my.gaussian_filter1d(
            densities, sigma=sigma_filter_ind_spec_den, axis=-1, truncate=1.0
        )  # shape (3, N_los, N_los, N_spectrum/2)

    my.plotForestsDens(
        lambda_range,
        forests[:, :30, :30, :],
        x_for_den,
        densities,
        line,
        line_obs,  # forests[:,:30,:30,:] #----------WARNING! :30 slice because the NBK was killed beofre finish on 3007
        titlePlots["mat/flux"],
        dirName_plot,
        filename=f"spectra_smooth",
        tot=True,
        losToPlot=losToPlot,
    )

    print(
        f"############################----- DENSITY -----##############################"
    )
    #################################################################################### COMPUTING PDF
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing PDF"
    )
    one_plus_delta = densities / np.mean(densities)
    my.plotPDF(
        one_plus_delta,
        nbinsPDF=1000,
        range=(0, 2),
        x_label=r"Matter density $\varrho$",
        title=titlePlots["PS mat"],
        dirName_plot=dirName_plot,
        save_to=pdf_file,
        xlog=False,
        ylog=False,
        filename="densities3",
        savefig=True,
    )
    my.plotPDF(
        one_plus_delta,
        nbinsPDF=1000,
        range=(-4, 2),
        x_label=r"Perturbations $\delta$",
        title=titlePlots["PS mat"],
        dirName_plot=dirName_plot,
        save_to=pdf_file,
        xlog=True,
        ylog=False,
        filename="delta",
        savefig=True,
    )

    """    #################################################################################### COMPUTING POWER SPECTRA
    print(f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing 1D power spectrum")
    # since the density array is of size N_grid and not N_spectrum, when binning, i get empty bins. i therefore need more data...
    densities_hat = np.empty((3, N_los, N_los, N_spectrum))
    for i in range(N_spectrum): # this is in order for the powerSpectrum function to work also on the density distributions...
        densities_hat[:,:,:,i] = densities[:,:,:,i*N_grid//N_spectrum] 

    kz_mat, kbins_edges_mat, kbins_centers_mat = my.sample_k(k_min, k_max, nbins=nbins, N_spectrum=N_spectrum, log_binning=log_binning) # Precompute bins once
    kz_vel_mat = my.convert_k_to_velocity_space(kz_mat, redshift=redshift, H_z=H_z)
    kbins_edges_vel_mat = my.convert_k_to_velocity_space(kbins_edges_mat, redshift=redshift, H_z=H_z)
    kbins_centers_vel_mat = my.convert_k_to_velocity_space(kbins_centers_mat, redshift=redshift, H_z=H_z)

    Pk_1D_vel_mat = my.compute_all_1d_power_spectra(densities_hat, kz_vel_mat, kbins_edges_vel_mat, L_box, N_grid=N_grid, R=R,
                                                    sigma_filter_spec=sigma_filter_ind_spec_den, sigma_filter_PS=sigma_Window_PS) # shape (3, N_los, N_los, N_spectrum/2)

    print(f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: fitting 1D power spectrum")
    if fit:
        Pk_fit_mat, Pk_fit_std_mat, fit_params_mat, fit_cov_mat = my.fitPk(kbins_centers_vel_mat, Pk_1D_vel_mat)
        print(fit_params_mat)
    else: Pk_fit_mat = None

    print(f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: translating to 3D")
    Pk_3D_vel_mat = my.oneD_to_threeD_powSpectr(Pk_1D_vel_mat, kbins_centers_vel_mat, sigma_Window_PS3D) # shape (3, N_los, N_los, N_spectrum/2)
    if Pk_fit_mat != None: Pk_3D_fit_mat = my.oneD_to_threeD_powSpectr(Pk_fit_mat, kbins_centers_vel_mat, sigma_Window_PS3D)
    else: Pk_3D_fit_mat = None

    print(f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: getting means&medians")
    Pk_1D_tot_vel_mat = my.converge(kbins_centers_vel_mat, Pk_1D_vel_mat, Pk_fit_mat, dir=3, shape=(0,1))
    Pk_3D_tot_vel_mat = [my.oneD_to_threeD_powSpectr(temp, kbins_centers_vel_mat, sigma_Window_PS3D) for temp in Pk_1D_tot_vel_mat] #my.converge(kbins_centers_vel_mat, Pk_3D_vel_mat, Pk_3D_fit_mat, dir=3, shape=(0,1))

    my.plotPowSpectr(kbins_centers_vel_mat, kbins_centers_vel_mat*Pk_1D_vel_mat, kbins_centers_vel_mat*Pk_1D_tot_vel_mat,
                    colors=colors, x_label=r"$k\ [\mathrm{s/km}]$", y_label= r"$P_{mat}(k)\ [\mathrm{km/s}]$",
                    filename="1D_vel_MAT", dirName_plot=dirName_plot, title=titlePlots["PS mat"], savefig=True,
                    summary=True, fit=fit, dir=3)

    Delta2_mat = L_box**3 / (2*np.pi)**3 * 4*np.pi * kbins_centers_vel_mat**3 * Pk_3D_vel_mat
    Delta2_tot_mat = L_box**3 / (2*np.pi)**3 * 4*np.pi * kbins_centers_vel_mat**3 * Pk_3D_tot_vel_mat
    my.plotPowSpectr(kbins_centers_vel_mat, Delta2_mat, Delta2_tot_mat, 
                    colors=colors, x_label = r"$k\ [\mathrm{s/km}]$", y_label = r"$\Delta^2_{mat}(k)\ [-]$",
                    filename="3D_vel_MAT", dirName_plot=dirName_plot, title=titlePlots["PS mat"], savefig=True,
                    summary=True, fit=fit, dir=3)

    """

    print(f"############################----- FLUX -----##############################")
    #################################################################################### COMPUTING PDF
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing PDF"
    )
    my.plotPDF(
        forests,
        nbinsPDF=1000,
        range=(0, 100),
        x_label=r"Flux $F$",
        title=titlePlots["PS mat"],
        dirName_plot=dirName_plot,
        save_to=pdf_file,
        xlog=False,
        ylog=True,
        filename="flux",
        savefig=True,
    )
    my.plotPDF(
        forests / np.mean(forests),
        nbinsPDF=1000,
        range=(-9, 0),
        x_label=r"Flux perturbations $\delta F$",
        title=titlePlots["PS mat"],
        dirName_plot=dirName_plot,
        save_to=pdf_file,
        xlog=True,
        ylog=True,
        filename="deltaF",
        savefig=True,
    )

    #################################################################################### COMPUTING POWER SPECTRA
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing 1D power spectrum"
    )
    kz, kbins_edges, kbins_centers = my.sample_k(
        k_min, k_max, nbins, N_spectrum, log_binning
    )  # Precompute bins once
    kz_vel = my.convert_k_to_velocity_space(kz, redshift=redshift, H_z=H_z)
    kbins_edges_vel = my.convert_k_to_velocity_space(
        kbins_edges, redshift=redshift, H_z=H_z
    )
    kbins_centers_vel = my.convert_k_to_velocity_space(
        kbins_centers, redshift=redshift, H_z=H_z
    )

    # Pk_1D = my.compute_all_1d_power_spectra(forests, kz, kbins_edges, L_box, N_grid, R,
    #                                        sigma_filter_spec=sigma_filter_ind_spec_forests, sigma_filter_PS=sigma_Window_PS) # shape (3, N_los, N_los, N_spectrum/2)
    Pk_1D_vel = my.compute_all_1d_power_spectra(
        forests,
        kz_vel,
        kbins_edges_vel,
        L_box,
        N_grid,
        R,
        sigma_filter_spec=sigma_filter_ind_spec_forests,
        sigma_filter_PS=sigma_Window_PS,
    )
    # shape (3, N_los, N_los, N_spectrum/2)
    # Pk_1D.copy() # = 1/my.convert_k_to_velocity_space(1/Pk_1D)
    # print("k shape powSpectr: ", kbins_centers_vel.shape, "\nPk1D shape powSpectr: ", Pk_1D_vel.shape)
    # print("mean intensity as observed and expected: ", np.mean(forests), 100*np.exp(-params["tau_0"]))

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: fitting 1D power spectrum"
    )
    if fit:
        Pk_fit, Pk_fit_std, fit_params, fit_cov = my.fitPk(kbins_centers_vel, Pk_1D_vel)
        print(fit_params)
    else:
        Pk_fit = None

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: translating to 3D"
    )
    # Pk_3D = my.oneD_to_threeD_powSpectr(Pk_1D, kbins_centers, sigma_Window_PS3D)
    Pk_3D_vel = my.oneD_to_threeD_powSpectr(
        Pk_1D_vel, kbins_centers_vel, sigma_Window_PS3D
    )  # shape (3, N_los, N_los, N_spectrum/2)
    # Pk_3D.copy() # 1/my.convert_k_to_velocity_space(1/Pk_3D)
    if Pk_fit != None:
        Pk_3D_fit = my.oneD_to_threeD_powSpectr(
            Pk_fit, kbins_centers_vel, sigma_Window_PS3D
        )
    else:
        Pk_3D_fit = None

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: getting means&medians"
    )
    # Pk_1D_tot = my.converge(kbins_centers, Pk_1D, dir=3, shape=(0,1)) #remember: first element will be cut off
    Pk_1D_tot_vel = my.converge(
        kbins_centers_vel, Pk_1D_vel, Pk_fit, dir=3, shape=(0, 1)
    )
    # Pk_3D_tot = my.converge(kbins_centers, Pk_3D, dir=3, shape=(0,1))
    Pk_3D_tot_vel = my.converge(
        kbins_centers_vel, Pk_3D_vel, Pk_3D_fit, dir=3, shape=(0, 1)
    )

    #################################################################################### COMPUTING WAVELET SCATTERING TRANSFORMS
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: computing 1D WST"
    )
    WST_coeffs = my.compute_all_1d_WST(
        forests,
        J,
        Q,
        order,
        sigma_filter_ind_spec_forests,
        sigma_Window_WST,
        out_type="list",
    )
    # shape WST: (ncoeffs, ["coef",             3*N_los*N_los, nsamples)         nsamples=N_spectrum/2^J
    #                       "n", "j"]           1 or 2)
    ncoeffs = len(WST_coeffs)
    if nsamples != WST_coeffs[0]["coef"].shape[1]:
        raise ValueError(
            f"nsamples != N_spectrum/2^J:         {nsamples}!={WST_coeffs[0]['coef'].shape[1]}"
        )
    print(nsamples)

    # print("ncoeffs WST: ", ncoeffs, "shape coefficients: ", (3*N_los*N_los, nsamples))
    # for i in range(ncoeffs): print(i, " j: ", WST_coeffs[i]['j'], "n: ", WST_coeffs[i]['n'])

    _, _, k_samples = my.sample_k(
        k_min, k_max, nbins=nsamples, N_spectrum=N_spectrum, log_binning=log_binning
    )
    k_samples_vel = my.convert_k_to_velocity_space(
        k_samples, redshift=redshift, H_z=H_z
    )

    WST_tot = my.summerize_WST(WST_coeffs, J, order)

    #################################################################################### WRITING RESULTS TO FILE
    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: saving powSpectrum to file"
    )

    np.save(k_PS_file, kbins_centers_vel)  # shape (N_spectrum/2)
    np.save(k_WST_file, WST_tot["x"])  # shape (n_coeffs)
    np.save(
        powSpectr_file, Pk_1D_vel.reshape(3 * N_los * N_los, -1).T
    )  # shape (N_spectrum/2, n_los)
    # np.save(f"{powSpectr_file}_mean", Pk_1D_tot_vel[0])
    np.save(WST_file, WST_tot["y_means"])  # shape (n_coeffs, n_los)
    # np.save(f"{WST_file}_mean", WST_tot["y_means_tot"])

    """
    os.makedirs(os.path.dirname(powSpectr_file), exist_ok=True)
    with open(powSpectr_file, 'w', newline='') as p:
        writer = csv.writer(p, delimiter= " ") #, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(nbins): #len(kbins_centers)):
            writer.writerow([kbins_centers[i], Pk_1D_tot_vel[0,i]])

    print(f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: saving WST to file")
    os.makedirs(os.path.dirname(WST_file), exist_ok=True)
    with open(WST_file, 'w', newline='') as p:
        writer = csv.writer(p, delimiter=" ", quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(1, ncoeffs): #len(WST_tot["y_means_tot"])): # skip zeroth order
            writer.writerow([WST_tot['x'][i], WST_tot['y_means_tot'][i]])

    if not os.path.exists(file_param):  #rewrite the param file
        raise FileNotFoundError(f"Data file {file_param} not found!")
    with open(file_param,'w') as p:
        writer = csv.writer(p, delimiter=' ') #, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(params.keys())):
            writer.writerow([list(params.keys())[i], list(params.values())[i]])
        writer.writerow(['nbins', nbins])
        writer.writerow(['nsamples', nsamples])
        writer.writerow(['ncoeffs', ncoeffs])
        writer.writerow(['J', J])
        writer.writerow(['Q', Q])
        writer.writerow(['order', order])
    """

    #################################################################################### PLOTTING

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: plotting power spectrum 1D"
    )
    # print("k shape powSpectr: ", kbins_centers_vel.shape, "\nPk1D shape powSpectr: ", Pk_1D_vel.shape, "\nPk1Dtot shape powSpectr: ", Pk_1D_tot_vel.shape)

    my.plotPowSpectr(
        kbins_centers_vel,
        Pk_1D_vel,
        Pk_1D_tot_vel,  # kbins_centers_vel[:100], Pk_1D_vel[:,:,:,:100], Pk_1D_tot_vel[:,:100]
        colors=colors,
        x_label=r"$k\ [\mathrm{s/km}]$",
        y_label=r"$P_{flux}(k)\ [\mathrm{km/s}]$",
        losToPlot=losToPlot,
        filename="1D_vel_FLUX",
        dirName_plot=dirName_plot,
        title=titlePlots["PS flux"],
        savefig=savefigs[0],
        summary=True,
        fit=fit,
        dir=3,
        oneD=True,
    )

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: plotting power spectrum 3D"
    )
    Delta2 = (
        Pk_3D_vel * kbins_centers_vel**3 / (2 * np.pi**2)
    )  # L_box**3 / (2*np.pi)**3 * 4*np.pi * kbins_centers_vel**3 * Pk_3D_vel
    Delta2_tot = (
        Pk_3D_tot_vel * kbins_centers_vel**3 / (2 * np.pi**2)
    )  # L_box**3 / (2*np.pi)**3 * 4*np.pi * kbins_centers_vel**3 * Pk_3D_tot_vel

    my.plotPowSpectr(
        kbins_centers_vel,
        Delta2,
        Delta2_tot,
        colors=colors,
        x_label=r"$k\ [\mathrm{s/km}]$",
        y_label=r"$\Delta^2(k)\ [-]$",
        losToPlot=losToPlot,
        filename="3D_vel_FLUX",
        dirName_plot=dirName_plot,
        title=titlePlots["PS flux"],
        savefig=savefigs[1],
        summary=True,
        fit=fit,
        dir=3,
        oneD=False,
    )

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: plotting WST general"
    )
    my.plotWS_all(
        k_samples_vel,
        WST_coeffs,
        J,
        order,
        colors=colors,
        xlabels=None,
        ylabels=None,
        losToPlot=losToPlot**2,
        filename="wst",
        dirName_plot=dirName_plot,
        title=titlePlots["WST flux"],
        savefig=savefigs[2],
    )

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: plotting WST summarized"
    )
    my.plotWS_summary(
        WST_tot,
        J,
        order,
        colors=colors,
        xlabels=None,
        ylabels=None,
        losToPlot=losToPlot**2,
        filename="wst_summary",
        dirName_plot=dirName_plot,
        title=titlePlots["WST flux"],
        savefig=savefigs[3],
        stat=True,
    )  ########### True: means, False: medians

    print(
        f"----------------- sec {(Timestamp.now()-start_time).total_seconds():.2f}: DONE"
    )
