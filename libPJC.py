import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, generic_filter1d
from scipy import stats
from kymatio.numpy import Scattering1D
from nbodykit.lab import cosmology
from numpy.linalg import inv
from astropy import constants as c, units as u
from scipy.special import wofz
from scipy.optimize import curve_fit
import gc  # garbage collector

# Enable LaTeX text rendering
plt.rc("text", usetex=True)


def plotSave(filename, dirName_plot, title, savefig=True):
    """
    Saves or displays the current matplotlib figure.

    Parameters:
        filename (str): Name for the output file (without extension)
        dirName_plot (str): Directory to save the plot
        title (str): Title for the plot
        savefig (bool): If True, saves as PDF; if False, displays plot
    """
    os.makedirs(dirName_plot, exist_ok=True)
    save_path = os.path.join(dirName_plot, f"{filename}.pdf")
    plt.suptitle(title + f"\nsave to {save_path}")
    print(f"Save to: {save_path}")
    if savefig:
        plt.savefig(
            save_path, format="pdf"
        )  # , dpi=200) #, bbox_inches='tight') # Save with full path
    else:
        plt.show()
    # plt.show()
    plt.close()


# ====================================================================================================================== FORESTING
#################################################################################### FUNCTIONS FOR THE IGM PROPERTIES
def temp(rho, Temp_0, gamma, mean_density):  # T-ρ relation Viel_2004, garzilli_2015
    """
    Computes temperature from density using a power-law relation.

    Inputs:
        rho (Quantity): Gas density array [M_sun/kpc^3]
        Temp_0 (Quantity): Temperature normalization [K]
        gamma (float): Polytropic index
        mean_density (Quantity): Mean density of the field [M_sun/kpc^3]

    Returns:
        T (Quantity): Temperature array [K], clipped to 1e-10-1e10 K
    """
    T = Temp_0 * (rho / mean_density) ** (gamma - 1)
    T = np.clip(T, 1e-10 * u.K, 1e10 * u.K)
    return T


def neutral_frac(rho, T, z, mean_density):  # check Garzilli_2015 !
    """
    Computes neutral hydrogen fraction based on density and temperature.

    Inputs:
        rho (Quantity): Gas density array [M_sun/kpc^3]
        T (Quantity): Temperature array [K]
        z (float): Redshift
        mean_density (Quantity): Mean density of the field [M_sun/kpc^3]

    Returns:
        xHI (Quantity): Neutral hydrogen density array [M_sun/kpc^3]
    """
    # for 1 < z < 5.5 (McQuinn_2016 p.7f)
    # Gunn-Peterson troughs in the Hi Lyα forest at z ∼ 6, place the limit xHI & 10−4 (McQuinn_2016 p.24)
    # ALTERNATIVE: 2e-5 * (rho/mean_density)**0.7 * (temp/1e4)**-0.7 ----------------------------reference???????????????????
    photoionization_time = 3.0e4  # in yr, 1e12 in s                           # yr
    delta_b = rho / mean_density
    xHI = (
        photoionization_time
        * (delta_b * ((1 + z) / 4) ** 3)
        / 1e10
        * pow(T.value, -0.7)
    )  # McQuinn_2016 p.16, WeinbergDH_1998 p.2
    # xHI = np.clip(xHI, 1e-10, 1.0)
    # xHI = 9.6e-6 * delta_b * photoionization_time * 1e-12 * (T.value/1e4)**-0.72 * cosmo.Ob0*cosmo.h**2/0.022/0.72 * ((1+z)/5)**3 # Bolton&Becker_2015 p.5
    # xHI[delta_b > 100.] = 1.0 # Neutral DLAs (self-shielding) (Bird_2015 p.3)
    return xHI * rho


#################################################################################### FUNCTIONS FOR Ly-ALPHA ABSORPTION
def get_gamma_natural(lambda_0, A_21):
    """
    Calculates the natural line width (FWHM) in wavelength units.

    Inputs:
        lambda_0 (Quantity): Rest wavelength [m]
        A_21 (Quantity): Einstein A-coefficient [s^-1]

    Returns:
        gamma_natural (Quantity): FWHM in [Angstrom]
    """
    gamma_natural = ((lambda_0**2) * A_21 / (2 * np.pi * c.c)).to(u.Angstrom)
    return gamma_natural


def find_absorbed_lambda(
    wavelength, vel_hubble_los, vel_pec_los, sigma, line, N_spectrum
):
    """
    Finds wavelength range affected by absorption.

    Inputs:
        wavelength (Quantity): Wavelength array [Angstrom]
        vel_hubble_los (Quantity): Hubble flow velocity [km/s]
        vel_pec_los (Quantity): Peculiar velocity [km/s]
        sigma (Quantity): Absorption width [Angstrom]
        line (Quantity): Rest wavelength [Angstrom]
        N_spectrum (int): Size of wavelength array

    Returns:
        tuple: (start_index, end_index) of affected range
    """
    v_tot = (
        vel_hubble_los + vel_pec_los
    )  # Bird_2015 p.6 including hubble flow AND peculiar parallel velocity
    beta_rel = (v_tot.to(u.m / u.s) / c.c.to(u.m / u.s)).value
    if 1 - beta_rel < 1e-10:
        beta_rel = 1 - 1e-10
    lambda_received = line * np.sqrt((1 + beta_rel) / (1 - beta_rel))
    lambda_low = lambda_received - sigma
    lambda_high = lambda_received + sigma

    # Clip to wavelength array bounds
    lambda_low = max(lambda_low.value, wavelength[0].value)
    lambda_high = min(lambda_high.value, wavelength[-1].value)

    # Find indices
    i = 0
    while i < N_spectrum - 1 and wavelength[i].value < lambda_low:
        i += 1
    start = i
    while i < N_spectrum and wavelength[i].value < lambda_high:
        i += (
            1  # here i < N_spectrum, since end value is not reached in range(start,end)
        )
    end = i
    return (start, end)


def get_hubble_vel_los(gridpoint_ind_LOS, com_distance_LOS, n_points_LOS, H_z):
    """
    Computes Hubble flow velocity along line of sight.

    Inputs:
        gridpoint_ind_LOS (int): Grid index along LOS
        com_distance_LOS (Quantity): Comoving distance [Mpc]
        n_points_LOS (int): Number of grid points
        H_z (Quantity): Hubble parameter at redshift z [km/s/Mpc]

    Returns:
        vel_hubble_flow (Quantity): Hubble velocity [km/s]
    """
    # gridpoint_ind_LOS: index of the gridpoint along the line of sight (z for now, maybe later weird LOS)
    # com_distance_LOS: total comiving distance thorugh the box along the LOS (L_box for now, maybe later weird LOS)
    # interpolating to oblique LOS will possibly have non linear gridpoint distributions
    delta_gripoints = com_distance_LOS / n_points_LOS * gridpoint_ind_LOS
    vel_hubble_flow = H_z * delta_gripoints
    return vel_hubble_flow


def get_lambda_from_d(d, H_z, line):
    """
    Converts comoving distance to observed wavelength.

    Inputs:
        d (Quantity): Comoving distance [Mpc]
        H_z (Quantity): Hubble parameter [km/s/Mpc]
        line (Quantity): Rest wavelength [Angstrom]

    Returns:
        lambda_out (Quantity): Observed wavelength [Angstrom]
    """
    vel_hubble_flow = H_z * d
    beta_rel = (vel_hubble_flow.to(u.m / u.s) / c.c.to(u.m / u.s)).value
    lambda_out = line * np.sqrt((1 + beta_rel) / (1 - beta_rel))
    return lambda_out


############################ GAUSS ################################################# ABSORPTION
def gaussian_profile(x, mu, sigma, amplitude):
    return amplitude * np.exp(-(x - mu).value ** 2 / (2 * sigma**2).value)


def absorb_gauss(
    intensity,
    wavelength,
    density,
    temperature,
    velocity,
    N_grid,
    L_box,
    line,
    tau_0,
    tau_s,
    mean_density,
):  # should consider the density redshift relation during the delta-z = 0.0057
    """
    Applies Gaussian absorption profiles to a spectrum along a line of sight.

    Inputs:
        intensity (array): Initial spectrum flux values
        wavelength (Quantity): Wavelength array [Angstrom]
        density (Quantity): Gas density along LOS [M_sun/kpc^3]
        temperature (Quantity): Temperature along LOS [K]
        velocity (Quantity): Velocity along LOS [km/s]
        N_grid (int): Number of grid points
        L_box (Quantity): Simulation box size [Mpc]
        line (Quantity): Rest wavelength [Angstrom]
        tau_0 (float): Optical depth normalization
        tau_s (float): Optical depth exponent
        mean_density (Quantity): Mean density [M_sun/kpc^3]

    Returns:
        intensity (array): Spectrum with applied absorption
    """
    mean_density = np.mean(density)
    for i in range(N_grid):
        print(".", end="")
        sigma_doppler = (
            np.sqrt(((c.k_B * temperature[i]).to(u.J) / (c.m_p * c.c**2).to(u.J)).value)
            * line
        )
        tau_peak = tau_0 * pow(
            (density[i] / mean_density).value, tau_s
        )  # Optical depth at peak
        start, end = find_absorbed_lambda(
            wavelength,
            get_hubble_vel_los(i, L_box, N_grid),
            velocity[i],
            4 * sigma_doppler,
        )
        # look for the 3 sigma interval
        index_mid = (start + end) // 2
        for j in range(start, end):
            # print('|', end='')
            tau_projected = gaussian_profile(
                wavelength[j], wavelength[index_mid], sigma_doppler, tau_peak
            )
            intensity[j] = intensity[j] * np.exp(-tau_projected)
    print("\n")
    return intensity


############################ VOIGT ################################################# ABSORPTION
def voigt_profile(x, mu, sigma, gamma, amplitude):
    """
    Computes the Voigt profile using the Faddeeva function.
    Parameters:
        x (array): Wavelength grid
        mu (float): Central wavelength
        sigma (float): Gaussian standard deviation (thermal broadening)
        gamma (float): Lorentzian FWHM (natural/pressure broadening)
        amplitude (float): Peak optical depth (τ₀)
    Returns:
        Voigt profile (array)
    """
    # Convert to dimensionless parameters
    z = (x - mu + 1j * gamma / 2) / (sigma * np.sqrt(2))
    voigt = wofz(z).real / (sigma * np.sqrt(2 * np.pi))
    return amplitude * voigt.value


def absorb_voigt(
    intensity,
    wavelength,
    density,
    temperature,
    velocity,
    N_spectrum,
    N_grid,
    L_box,
    H_z,
    line,
    A_21,
    tau_0,
    tau_s,
    mean_density,
):
    """
    Applies Voigt absorption profiles (combining Gaussian and Lorentzian) to a spectrum.

    Inputs:
        intensity (array): Initial spectrum flux values
        wavelength (Quantity): Wavelength array [Angstrom]
        density (Quantity): Gas density along LOS [M_sun/kpc^3]
        temperature (Quantity): Temperature along LOS [K]
        velocity (Quantity): Velocity along LOS [km/s]
        N_spectrum (int): Size of wavelength array
        N_grid (int): Number of grid points
        L_box (Quantity): Simulation box size [Mpc]
        H_z (Quantity): Hubble parameter [km/s/Mpc]
        line (Quantity): Rest wavelength [Angstrom]
        A_21 (Quantity): Einstein A coefficient [s^-1]
        tau_0 (float): Optical depth normalization
        tau_s (float): Optical depth exponent
        mean_density (Quantity): Mean density [M_sun/kpc^3]

    Returns:
        intensity (array): Spectrum with applied absorption
    """
    for i in range(N_grid):
        sigma_doppler = (
            np.sqrt(c.k_B * temperature[i] / (c.m_p * c.c**2)) * line
        )  # Thermal Doppler broadening
        tau_peak = tau_0 * pow(
            (density[i] / mean_density).value, tau_s
        )  # Optical depth at peak Viel_2004 p5
        FWHM_G = 2 * sigma_doppler * np.sqrt(2 * np.log(2))
        gamma_natural = get_gamma_natural(line, A_21)
        FWHM_L = 2 * gamma_natural
        FWHM_V = FWHM_L / 2 + np.sqrt(FWHM_L * FWHM_L / 4 + FWHM_G * FWHM_G)
        start, end = find_absorbed_lambda(
            wavelength,
            get_hubble_vel_los(i, L_box, N_grid, H_z),
            velocity[i],
            2 * FWHM_V,
            line,
            N_spectrum,
        )  # Find affected wavelength range
        # look for the 2 full width half maximum interval
        index_mid = (end + start) // 2

        # print(tau_peak, density[i], mean_density)

        for j in range(start, end):
            # Calculate Voigt profile
            # print('|', end='')
            tau_projected = voigt_profile(
                wavelength[j],
                wavelength[index_mid],
                sigma_doppler,
                gamma_natural,
                tau_peak,
            )
            try:
                if isinstance(tau_projected, u.Quantity):
                    exponent = tau_projected.value
                else:
                    exponent = tau_projected
                intensity[j] = intensity[j] * np.exp(-exponent)
            except ValueError:
                print(intensity[j])
    return intensity


def organizeAbsorption(mpi_on, size, rank, N_los):
    """
    Organizes MPI work distribution for absorption calculation.

    Inputs:
        mpi_on (bool): MPI flag
        size (int): Number of MPI processes
        rank (int): Current process rank
        N_los (int): Number of lines of sight per dimension

    Returns:
        tuple: (direction, start_index, end_index) for current process
    """
    if mpi_on:
        my_dir = rank % 3  # 0=x, 1=y, 2=z
        ranks_per_dir = size // 3
        rank_in_dir = rank // 3  # Position in direction group

        total_los = N_los * N_los
        chunk_size = total_los // ranks_per_dir
        start = rank_in_dir * chunk_size
        end = (
            (rank_in_dir + 1) * chunk_size
            if rank_in_dir != (ranks_per_dir - 1)
            else total_los
        )

        return my_dir, start, end
    else:
        # Serial case - process all directions
        return [0, 1, 2], 0, N_los * N_los


def computeAbsorption(
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
    dirName_out,
):
    """
    Computes absorption spectra for assigned lines of sight.

    Inputs:
        my_dir (int): Direction index (0=x,1=y,2=z)
        start/end (int): LOS range indices
        N_grid (int): Number of grid points per dimension
        N_los (int): Number of lines of sight per dimension
        N_spectrum (int): Wavelength array size
        L_box (Quantity): Box size [Mpc]
        res (Quantity): Spectral resolution [Angstrom]
        file_mocks (dict): Paths to input files
        HI (bool): Neutral hydrogen flag
        mocked (bool): Mock data flag
        redshift (float): Redshift
        H_z (Quantity): Hubble parameter [km/s/Mpc]
        line (Quantity): Rest wavelength [Angstrom]
        line_obs (Quantity): Observed wavelength [Angstrom]
        lambda_margin (Quantity): Wavelength margin [Angstrom]
        einstein_Acoeff (Quantity): Einstein A coefficient [s^-1]
        Temp_0 (Quantity): Temperature normalization [K]
        gamma (float): Polytropic index
        tau_0 (float): Optical depth normalization
        tau_s (float): Optical depth exponent
        mpi_on (bool): MPI flag
        rank (int): MPI rank

    Returns:
        tuple: (wavelength_array, spectra_array)
    """
    # Precompute wavelength range once
    lambda_range = (
        np.arange(
            (line - lambda_margin).value, (line_obs + lambda_margin).value, res.value
        )
        * u.Angstrom
    )

    # Load density field ONCE
    den_field = np.load(file_mocks["dens"]).reshape((N_grid, N_grid, N_grid), order="C")
    if not mocked:
        den_field *= 1e-9
    den_field = den_field * (u.M_sun / (u.kpc) ** 3)
    mean_density = np.mean(den_field)

    # Load velocity field ONLY for needed direction
    file_vel = [file_mocks["vel_x"], file_mocks["vel_y"], file_mocks["vel_z"]][my_dir]
    vel_field = np.load(file_vel).reshape((N_grid, N_grid, N_grid), order="C")
    vel_field *= u.km / u.s

    # Initialize spectra array
    chunk_size = end - start
    spectra = np.zeros((chunk_size, N_spectrum)) + 100.0

    for idx_local, idx_global in enumerate(range(start, end)):
        j = idx_global // N_los
        k = idx_global % N_los

        # Print progress (only from rank 0 in MPI mode)
        if (not mpi_on) or (rank == 0):
            if k == 0:
                print(f"\n  {j}:", end=" ", flush=True)
            print(k, end=" ", flush=True)

        # Get LOS slices
        if my_dir == 0:
            den_slice = den_field[:, j, k]
            vel_slice = vel_field[:, j, k]
        elif my_dir == 1:
            den_slice = den_field[j, :, k]
            vel_slice = vel_field[j, :, k]
        else:  # my_dir == 2
            den_slice = den_field[j, k, :]
            vel_slice = vel_field[j, k, :]

        # Compute temperature and neutral fraction
        temp_slice = temp(den_slice, Temp_0, gamma, mean_density)
        if HI:
            absorber_slice = neutral_frac(den_slice, temp_slice, redshift, mean_density)
            mean_density = neutral_frac(
                mean_density,
                temp(mean_density, Temp_0, gamma, mean_density),
                redshift,
                mean_density,
            )
        else:
            absorber_slice = den_slice

        # Compute absorption
        spectra[idx_local] = absorb_voigt(
            intensity=np.full(N_spectrum, 100.0),
            wavelength=lambda_range,
            density=absorber_slice,
            temperature=temp_slice,
            velocity=vel_slice,
            N_spectrum=N_spectrum,
            N_grid=N_grid,
            L_box=L_box,
            H_z=H_z,
            line=line,
            A_21=einstein_Acoeff,
            tau_0=tau_0,
            tau_s=tau_s,
            mean_density=mean_density,
        )

    dirName_allRanks = os.path.join(
        dirName_out, f"AllRanks_{N_grid}_{N_los}_{L_box.value}"
    )
    os.makedirs(dirName_allRanks, exist_ok=True)
    filename = f"R{rank}_D{my_dir}_E{end}_S{start}"
    filepath = os.path.join(dirName_allRanks, filename)
    np.save(filepath, spectra)

    # Clean up large arrays
    del den_field, vel_field
    gc.collect()

    if not mpi_on or rank == 0:
        print("\n\n", spectra.shape, spectra[0])
    if np.all(spectra == 0):
        print(f"\nWarning: Rank {rank} has all zero spectra!")
    return lambda_range, spectra


def write_spectra_file(
    spectra,
    lambda_range,
    file_names,
    mpi_on,
    comm,
    size,
    rank,
    N_los,
    N_spectrum,
    start,
    end,
):
    """
    Writes absorption spectra to files.

    Inputs:
        spectra (array): Absorption spectra
        lambda_range (Quantity): Wavelength array
        file_names (dict): Output file paths
        mpi_on (bool): MPI flag
        comm: MPI communicator
        size (int): Number of MPI processes
        rank (int): Current process rank
        N_los (int): Number of lines of sight
        N_spectrum (int): Wavelength array size
        start/end (int): LOS range indices
    """
    if mpi_on:
        from mpi4py import MPI

        if not isinstance(comm, MPI.Intracomm):
            raise ValueError(f"Invalid communicator of type {type(comm)}")

        gathered_spectra = comm.gather(spectra, root=0)

        if rank == 0:
            # Initialize full array
            final_spectra = np.zeros((3, N_los, N_los, N_spectrum))
            total_los = N_los * N_los
            ranks_per_dir = size // 3
            chunk_size = total_los // ranks_per_dir

            # Reconstruct full array
            for r in range(size):
                d = r % 3  # Direction (0=x,1=y,2=z)
                rank_in_dir = r // 3
                chunk_start = rank_in_dir * chunk_size
                chunk_end = (
                    (rank_in_dir + 1) * chunk_size
                    if rank_in_dir != (ranks_per_dir - 1)
                    else total_los
                )

                # Convert 1D indices back to 2D (j,k)
                for idx in range(chunk_start, chunk_end):
                    j = idx // N_los
                    k = idx % N_los
                    final_spectra[d, j, k, :] = gathered_spectra[r][
                        idx - chunk_start, :
                    ]
    else:
        # Serial case
        final_spectra = spectra.reshape((3, N_los, N_los, N_spectrum))

    # Write to file (only root in MPI mode)
    if not mpi_on or rank == 0:
        # Debug: Verify we have data
        print(f"Debug - Final spectra shape: {final_spectra.shape}")
        print(f"Debug - Non-zero values: {np.count_nonzero(final_spectra)}")

        for d in range(3):
            output_file = f"{file_names['specs'][d+1]}"
            print(f"Saving forests for direction {d} to {output_file}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, "w", newline="") as f:
                writer = csv.writer(
                    f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                for j in range(N_los):
                    for k in range(N_los):
                        if np.all(final_spectra[d, j, k, :] == 0):
                            print(f"Warning: Zero spectra at d={d}, j={j}, k={k}")
                        writer.writerow(final_spectra[d, j, k, :].round(6))

        # Write wavelength data
        wavelength_file = file_names["specs"][0]
        with open(wavelength_file, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(lambda_range[:N_spectrum].value)


def write_den_file(
    d, N_los, N_grid, L_box, H_z, line, files_in, files_out, mpi_on, rank
):
    """
    Writes density data to files.

    Inputs:
        d (int): direction to save
        N_los (int): Number of lines of sight
        N_grid (int): Number of grid points
        L_box (Quantity): Box size [Mpc]
        H_z (Quantity): Hubble parameter [km/s/Mpc]
        line (Quantity): Rest wavelength [Angstrom]
        files_in (dict): Input file paths
        files_out (dict): Output file paths
    """
    if not mpi_on or rank == 0:
        den_field = np.load(files_in["dens"]).reshape(
            (N_grid, N_grid, N_grid), order="C"
        )

        print(f"Saving densities for direction {d}")
        os.makedirs(os.path.dirname(files_out["dens"][d + 1]), exist_ok=True)
        with open(f"{files_out['dens'][d+1]}", "w", newline="") as f:
            writer = csv.writer(
                f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for j in range(N_los):
                for k in range(N_los):
                    if d == 0:
                        den_slice = den_field[:, j, k]
                    elif d == 1:
                        den_slice = den_field[j, :, k]
                    else:  # d == 2
                        den_slice = den_field[j, k, :]
                    writer.writerow(den_slice.round(6))

        print("Saving position data")
        os.makedirs(os.path.dirname(files_out["dens"][0]), exist_ok=True)
        with open(files_out["dens"][0], "w", newline="") as f:
            writer = csv.writer(
                f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            x_pos = np.arange(0, L_box.value, L_box.value / N_grid)
            lambda_pos = get_lambda_from_d(x_pos * u.Mpc, H_z, line).value
            writer.writerow(lambda_pos)


# ====================================================================================================================== READ
def getParams(file_param, foresting):
    """
    Reads simulation parameters from file.

    Inputs:
        file_param (str): Parameter file path
        foresting (bool): Flag for forest analysis

    Returns:
        dict: Parameter dictionary with converted values
    """
    params = {}
    if not os.path.exists(file_param):
        raise FileNotFoundError(f"Parameters file {file_param} not found!")
    with open(file_param, "r", newline="") as p:
        reader = csv.reader(p, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            try:
                # Try for int
                if foresting and row[0] == "nbins":
                    break  # nbins is not a foresting parameter and will be written by analyze. this line avoids doubles
                params[row[0]] = int(row[1])
            except ValueError:
                try:
                    # If that fails, try float
                    params[row[0]] = float(row[1])
                except ValueError:
                    # If both fail, keep as string
                    params[row[0]] = row[1]
    return params


def getData(files, N_spectrum, N_grid, N_los, csv_on=True):
    """
    Reads spectral and density data from files.

    Inputs:
        files (dict): Data file paths
        N_spectrum (int): Expected wavelength array size
        N_grid (int): Expected grid size
        N_los (int): Expected number of lines of sight
        csv_on (bool): Whether input file will be ".csv" or ".npy"
    Returns:
        tuple: (forests_array, lambda_range, densities_array, x_positions)
    """

    forests = np.empty((3, N_los, N_los, N_spectrum))
    for i, file_path in enumerate(files["specs"][1:]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found!")
        if csv_on:
            with open(file_path, "r", newline="") as p:
                # Read all data at once (more efficient for grid data)
                try:
                    # data = np.loadtxt(p, delimiter=",")
                    data = np.loadtxt(p, delimiter=" ")
                    if data.shape != (N_los**2, N_spectrum):
                        raise ValueError(
                            f"File {file_path} has shape {data.shape}, expected (N_los², N_spectrum) = ({N_los**2}, {N_spectrum})"
                        )
                    forests[i] = data.reshape(N_los, N_los, N_spectrum)
                except ValueError as e:
                    raise ValueError(f"Non-numeric data in {file_path}: {str(e)}")
        else:
            data = np.load(file_path)
            forests[i] = data.reshape(N_los, N_los, N_spectrum)

    if np.array(forests).shape != (3, N_los, N_los, N_spectrum):
        raise ValueError(
            f"Final shape {np.array(forests).shape} doesn't match expected (3, N_spectrum, N_los²) = (3, {N_spectrum}, {N_los**2})"
        )

    if csv_on:
        p = open(files["specs"][0], "r", newline="")
        reader = csv.reader(p, delimiter=" ", quotechar="|")
        row = next(reader)
        try:
            lambda_range = [float(x) for x in row]  ########################## my data
        except ValueError:
            print(
                f"ValueError: the file {files['specs'][0]} contains non float characters in row: {row}"
            )
    else:
        lambda_range = np.load(files["specs"][0])
    N_los = 30  # ----------- WARNING!!! this is only because the NBK was killed before finish
    densities = np.empty((3, N_los, N_los, N_grid))
    for i, file_path in enumerate(files["dens"][1:]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found!")
        if csv_on:
            with open(file_path, "r", newline="") as p:
                # Read all data at once (more efficient for grid data)
                try:
                    data = np.loadtxt(p, delimiter=" ")
                    if data.shape != (N_los**2, N_grid):
                        raise ValueError(
                            f"File {file_path} has shape {data.shape}, expected (N_los², N_grid) = ({N_los**2}, {N_grid})"
                        )
                    densities[i] = data.reshape(N_los, N_los, N_grid)
                except ValueError as e:
                    raise ValueError(f"Non-numeric data in {file_path}: {str(e)}")
        else:
            data = np.load(file_path)
            densities[i] = data.reshape(N_los, N_los, N_grid)

    if np.array(densities).shape != (3, N_los, N_los, N_grid):
        raise ValueError(
            f"Final shape {np.array(densities).shape} doesn't match expected (3, N_grid, N_los²) = (3, {N_grid}, {N_los**2})"
        )

    if csv_on:
        p = open(files["dens"][0], "r", newline="")
        reader = csv.reader(p, delimiter=" ", quotechar="|")
        row = next(reader)
        try:
            x_for_den = [float(x) for x in row]  ######################### my data
        except ValueError:
            print(
                f"ValueError: the file {files['dens'][0]} contains non float characters in row: {row}"
            )
    else:
        x_for_den = np.load(files["dens"][0])

    return forests, lambda_range, densities, x_for_den


def plotForestsDens(
    lambda_range,
    forests,
    x_for_den,
    densities,
    line,
    line_obs,
    title,
    dirName_plot,
    filename,
    tot,
    losToPlot,
):
    """
    Plots absorption spectra and density profiles.

    Inputs:
        lambda_range (array): Wavelength values
        forests (array): Absorption spectra (3D)
        x_for_den (array): Position values
        densities (array): Density profiles (3D)
        line (float): Rest wavelength
        line_obs (float): Observed wavelength
        title (str): Plot title
        dirName_plot (str): Output directory
        filename (str): Output filename
        tot (bool): Combined plot flag
        losToPlot (int): Number of lines to plot
    """
    N_los = forests.shape[1]
    if (
        forests.shape[2] != N_los
        or densities.shape[1] != N_los
        or densities.shape[2] != N_los
        or densities.shape[-1] != len(x_for_den)
        or forests.shape[-1] != len(lambda_range)
    ):
        raise ValueError(
            "forests and densities have wrong shapes:", forests.shape, densities.shape
        )

    plt.figure(figsize=(14, 6))
    N_dir = np.min([losToPlot, 3])
    if losToPlot == 1:
        alpha = 1
        map_size = 2
    else:
        alpha = np.max([0.005, 1 / losToPlot])  # *.7
        map_size = losToPlot * losToPlot * N_dir
    cmap_dens = get_cmap("winter", lut=map_size)
    cmap_specs = get_cmap("autumn", lut=map_size)
    colors_dens = cmap_dens(
        np.linspace(0, 1, map_size)
    )  # double the number, to make it fit only half the map
    colors_specs = cmap_specs(np.linspace(0, 1, map_size))

    if tot:
        plt.xlabel(r"$\lambda$ [Angstrom]")
        for i in range(losToPlot):  # N_los):
            for j in range(losToPlot):  # N_los):
                for k in range(N_dir):  # 3):
                    plt.plot(
                        lambda_range,
                        forests[k, i, j, :],
                        color=colors_specs[-(i * losToPlot * N_dir + j * N_dir + k)],
                        alpha=alpha,
                    )
        for i in range(losToPlot):  # N_los):
            for j in range(losToPlot):  # N_los):
                for k in range(N_dir):  # 3):
                    plt.plot(
                        x_for_den,
                        densities[k, i, j, :]
                        / np.max(densities[:N_dir, :losToPlot, :losToPlot, :])
                        * 100,
                        color=colors_dens[-(i * losToPlot * N_dir + j * N_dir + k)],
                        alpha=alpha,  # /0.7 if alpha != 1 else alpha,
                    )
        #
        # plt.xticks(ticks=[line, line_obs], labels=[r"Ly-$\alpha$", "obs"])

    else:
        gs_spectra = plt.GridSpec(
            1, 3, width_ratios=[1, 1, 1]
        )  # Create grid for subplots

        # spectra_x (left) setup
        ax_x = plt.subplot(gs_spectra[:, 0])  # Takes all rows of first column
        ax_x.set_xlabel(r"$\lambda$ [Angstrom]")
        ax_x.set_ylabel(r"$\mathcal{F}$")
        for i in range(losToPlot):
            for j in range(losToPlot):
                ax_x.plot(
                    lambda_range,
                    forests[0, i, j, :],
                    color=colors_specs[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
                ax_x.plot(
                    x_for_den,
                    densities[0, i, j, :]
                    / np.max(densities[0, :losToPlot, :losToPlot, :])
                    * 100,
                    color=colors_dens[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
        ax_x.set_xticks(ticks=[line, line_obs], labels=[r"Ly-$\alpha$", "obs"])

        # spectra_y (middle) setup
        ax_y = plt.subplot(gs_spectra[:, 1])
        ax_y.set_xlabel(r"$\lambda$ [Angstrom]")
        for i in range(losToPlot):
            for j in range(losToPlot):
                ax_y.plot(
                    lambda_range,
                    forests[1, i, j, :],
                    color=colors_specs[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
                ax_y.plot(
                    x_for_den,
                    densities[1, i, j, :]
                    / np.max(densities[1, :losToPlot, :losToPlot, :])
                    * 100,
                    color=colors_dens[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
        ax_y.set_xticks(ticks=[line, line_obs], labels=[r"Ly-$\alpha$", "obs"])

        # spectra_z (right) setup
        ax_z = plt.subplot(gs_spectra[:, 2])
        ax_z.set_xlabel(r"$\lambda$ [Angstrom]")
        for i in range(losToPlot):
            for j in range(losToPlot):
                ax_z.plot(
                    lambda_range,
                    forests[2, i, j, :],
                    color=colors_specs[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
                ax_z.plot(
                    x_for_den,
                    densities[2, i, j, :]
                    / np.max(densities[2, :losToPlot, :losToPlot, :])
                    * 100,
                    color=colors_dens[-(i * losToPlot * 3 + j * 3)],
                    alpha=alpha,
                )
        ax_z.set_xticks(ticks=[line, line_obs], labels=[r"Ly-$\alpha$", "obs"])
        # f"Redshift: {redshift}, box size: {L_box:.2f}, Grid points: {N_grid}, tracer density: {nbar:.2f}, spectral resolution: {N_spectrum}"

    plotSave(filename, dirName_plot=dirName_plot, title=title, savefig=True)


def testSpectra(file_param, files, titlePlots, dirName_plot, foresting, tot):
    params = getParams(file_param, foresting)
    N_grid = params["N_grid"]
    N_spectrum = params["N_spectrum"]
    N_los = params["N_los"]
    L_box = params["L_box"]  # Mpc/h
    redshift = params["redshift"]
    nbar = params["nbar"]
    line = params["lambda_start"]  # in Angstroms
    line_obs = params["lambda_start"]  # in Angstroms
    mean_density = params["mean_density"]

    forests, lambda_range, densities, x_for_den = getData(
        files, N_spectrum, N_grid, N_los
    )
    densities = densities / np.mean(densities)

    plotForestsDens(
        lambda_range,
        forests,
        x_for_den,
        densities,
        line,
        line_obs,
        titlePlots,
        dirName_plot,
        filename=f"spectra",
        tot=True,
        losToPlot=1,
    )


def plotMock(
    L_box, N_grid, files, mocked, T_0, gamma, dirName_plot, title, savefig=True
):
    """
    Plots 2D slices of mock simulation data.

    Inputs:
        L_box (float): Box size [Mpc/h]
        den_field (array): Density field
        vel_field (array): Velocity field
        temp_field (array): Temperature field
        dirName_plot (str): Output directory
        title (str): Plot title
        savefig (bool): Save figure flag
    """
    ## DENSITY SLICE
    den_field = np.load(files["dens"]).reshape((N_grid, N_grid, N_grid), order="C")
    if not mocked:
        den_field *= 1e-9
    den_field = den_field * (u.M_sun / (u.kpc) ** 3)
    mean_density = np.mean(den_field)
    slice_index = N_grid // 2
    den_slice = den_field[
        slice_index, :, :
    ]  # HI_den_field ######################################
    del den_field
    gc.collect()

    ## TEMPERATURE SLICE
    temp_slice = temp(den_slice, T_0, gamma, mean_density)
    # print("shape temp_field: ", temp_field.shape, "temp_slice: ", temp_slice.shape)

    ## VELOCITY SLICE
    plt.figure(figsize=(14, 6))
    gs_vels = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])  # Create grid for subplots
    file_vel = [files["vel_x"], files["vel_y"], files["vel_z"]]
    vel_mag_slice = 0
    for i, dir in enumerate(["x", "y", "z"]):
        vel_field = np.load(file_vel[i]).reshape((N_grid, N_grid, N_grid), order="C")
        vel_field *= u.km / u.s
        vel_slice = vel_field[slice_index, :, :]  # Shape (3, N_grid, N_grid)
        vel_mag_slice += vel_slice**2
        ax = plt.subplot(gs_vels[:, i])  # Takes all rows of first column
        if mocked:
            clipLow = 1e-3
        else:
            clipLow = 1e1
        ocean_cmap = plt.cm.ocean
        half_ocean = mcolors.LinearSegmentedColormap.from_list(
            "half_ocean", ocean_cmap(np.linspace(0.4, 0.75, 256))
        )
        # im = ax.imshow(np.log10(np.clip(np.abs(vel_slice.value),clipLow,None)), origin='lower', cmap=half_ocean, extent=[0, L_box, 0, L_box])
        # im = ax.imshow(np.log10(np.clip(np.abs(vel_slice.value),clipLow,None)), origin='lower', cmap=half_ocean, extent=[0, L_box, 0, L_box])
        # im = ax.imshow(np.log10(np.clip(np.abs(vel_slice.value),clipLow,None)), origin='lower', cmap="cool", extent=[0, L_box, 0, L_box])
        im = ax.imshow(
            np.log10(np.clip(np.abs(vel_slice.value), clipLow, None)),
            origin="lower",
            cmap="plasma",
            extent=[0, L_box, 0, L_box],
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        not_dir = ["x", "y", "z"][:i] + ["x", "y", "z"][i + 1 :]
        ax.set_xlabel(f"{not_dir[0]} [Mpc/h]")
        ax.set_ylabel(f"{not_dir[1]} [Mpc/h]")
        unit = r"$\mathrm{log}_{10}(\mathrm{km/s})$"
        ax.set_title(f"Velocity Slice {dir} [{unit}]")
    plotSave("mock_velComps", dirName_plot, title, savefig=savefig)

    vel_mag_slice = np.sqrt(vel_mag_slice)  # Take slice and calculate magnitude
    # print("shape vel_field: ", vel_field.shape, "vel_mag_slice: ", vel_mag_slice.shape)

    plt.figure(figsize=(14, 6))
    gs_mock = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])  # Create grid for subplots

    # Density plot (left)
    ax_rho = plt.subplot(gs_mock[:, 0])  # Takes all rows of first column
    im_rho = ax_rho.imshow(
        np.log10(np.clip(den_slice.value, 1e-2, None)),
        origin="lower",
        cmap="viridis",
        extent=[0, L_box, 0, L_box],
    )
    plt.colorbar(im_rho, ax=ax_rho, shrink=0.8)
    ax_rho.set_xlabel("y [Mpc/h]")
    ax_rho.set_ylabel("z [Mpc/h]")
    unit = r"$\mathrm{log}_{10}(M_{\mathrm{sun}} \mathrm{Mpc}^{-3} h^3)$"
    ax_rho.set_title(f"Density Slice [{unit}]")

    # Velocity magnitude (middle)
    ax_vel = plt.subplot(gs_mock[:, 1])
    im_v = ax_vel.imshow(
        np.log10(np.clip(vel_mag_slice.value, 1e1, None)),
        origin="lower",
        cmap="plasma",
        extent=[0, L_box, 0, L_box],
    )
    plt.colorbar(im_v, ax=ax_vel, shrink=0.8)
    ax_vel.set_xlabel("y [Mpc/h]")
    unit = r"$\mathrm{log}_{10}(\mathrm{km/s})$"
    ax_vel.set_title("Velocity magnitude [{unit}]")

    # Temperature (right)
    ax_temp = plt.subplot(gs_mock[:, 2])
    im_temp = ax_temp.imshow(
        np.log10(np.clip(temp_slice.value, 1e3, None)),
        origin="lower",
        cmap="inferno",
        extent=[0, L_box, 0, L_box],
    )
    plt.colorbar(im_temp, ax=ax_temp, shrink=0.8)
    ax_temp.set_xlabel("y [Mpc/h]")
    unit = r"$\mathrm{log}_{10}(\mathrm{K})$"
    ax_temp.set_title("Temperature [{unit}]")

    plotSave("mock_DenVelTemp", dirName_plot, title, savefig=savefig)


#################################################################################### HOW TO COMPUTE PROBABILITY DENSITY FUNCTION


def plotPDF(
    data,
    nbinsPDF,
    range,
    x_label,
    title,
    dirName_plot,
    save_to,
    filename,
    xlog=True,
    ylog=False,
    savefig=True,
):
    """
    Calculates and plots probability density function.

    Inputs:
        data (array): Input data
        nbinsPDF (int): Number of bins
        range (tuple): Data range
        x_label (str): X-axis label
        title (str): Plot title
        dirName_plot (str): Output directory
        filename (str): Output filename
        ylog (bool): Log y-scale flag
        savefig (bool): Save figure flag
    """
    all_flux = data.flatten()

    print("densities zero?", np.any(all_flux == 0))

    if xlog:
        bins = np.logspace(range[0], range[1], nbinsPDF)
    else:
        bins = np.linspace(range[0], range[1], nbinsPDF)
    counts, bin_edges = np.histogram(all_flux, bins=bins)
    pdf = counts / counts.sum()  # Normalize to probability density
    np.save(f"{save_to}_{filename}", {"x": bin_edges, "y": pdf}, allow_pickle=True)

    plt.figure(figsize=(6, 4))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.scatter(bin_centers, pdf, marker="x", s=3, c="red")

    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(r"$N_\mathrm{cells}$")

    plotSave(f"pdf_{filename}", dirName_plot, title, savefig)


# ====================================================================================================================== ANALYZE


def format_value(value, format_spec=""):
    return (
        ""
        if value is None
        else (format(value, format_spec) if format_spec else str(value))
    )


def get_ticks(total_length, step):
    n_ticks = total_length // step
    return np.linspace(0, total_length - 1, n_ticks, dtype=int)


#################################################################################### HOW TO BIN AND SAMPLE
def sample_k(k_min, k_max, nbins, N_spectrum, log_binning=True):
    """
    Compute 1D k-bins along the spectral dimension
    Args:
        k_min: min k value
        k_max: max k value --> Nyquist frequency,
        nbins: nb of k bins
        N_spectrum: nb of values in spectrum
        log_binning: whether to bin in logspace or not
    Returns:
        k_mag: Array of wavenumbers
        kbins_edges: Bin edges
        kbins_centers: Bin centers
    """
    # relating to box (and Nyquist frequency Numerical recipes p.500)
    if log_binning:
        k_mag = np.logspace(
            np.log10(k_min), np.log10(k_max), N_spectrum // 2 + 1
        )  # logarithmic binning might mess up with the gaussian filtering...
    else:
        k_mag = np.linspace(k_min, k_max, N_spectrum // 2 + 1)

    # dx = L/N_grid # only L, not 2*L, because fftfreq handles Nyquist already)
    # k_mag = np.abs(2*np.pi * np.fft.fftfreq(N_spectrum, d=dx)) # absolute value because P(k) will be real?

    if log_binning:
        kbins_edges = np.logspace(
            np.log10(k_min), np.log10(k_max), nbins + 1
        )  # logarithmic binning might mess up with the gaussian filtering...
    else:
        kbins_edges = np.linspace(
            k_min, k_max, nbins + 1
        )  # relating to box (and Nyquist frequency Numerical recipes p.500)

    kbins_centers = 0.5 * (kbins_edges[1:] + kbins_edges[:-1])
    return k_mag, kbins_edges, kbins_centers


#################################################################################### HOW TO COMPUTE POWER SPECTRA
# Niemeyer Skript Würzburg p.69, Peebles p.167, Ryden p.237, Dodelson p.16
def _window_function(k, *, R, dv):
    """The window function corresponding to the spectra response of the spectrograph.
    R is the spectrograph resolution.
    dv is the pixel width of the spectrograph.
    Default values for BOSS are:
        dv = 69, R = 60 at 5000 A and R = 80 at 4300 A."""
    # FWHM of a Gaussian is 2 \sqrt(2 ln 2) sigma
    sigma = R / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * (k * sigma) ** 2)  # * np.sinc(k * dv/2/np.pi)


def gauss_W(k, R):
    Wk = np.exp(-((k * R) ** 2))
    return Wk


def tophat_W(k, R):
    Wk = 3 * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R) ** 3
    return Wk


def fftPower_1d(delta, kz, kbins_edges, L, nbins, sigma_filter):
    """
    Compute 1D power spectrum for a single spectrum
    Args:
        delta: 1D perturbation spectrum (length N_spectrum)
        kz: Wavenumbers from get_k_bins_1d()
        kbins_edges: Bin edges
    Returns:
        Pk_1d: Binned 1D power spectrum
    """
    delta_k = np.fft.rfft(delta)  #  np.fft.fft(delta)
    Pk = np.abs(delta_k) ** 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ not sure
    Pk *= (
        L / nbins**2
    )  # Pk * L*2 * kz /(2*np.pi)                  ----------------------- NORMALIZATION (cf Kaiser&Peacock_1991 p.8)
    # dx = L / N_spectrum
    # Pk = np.abs(delta_k)**2 * dx / N_spectrum**2
    # Pk = np.abs(delta_k)**2 * dx / L
    # for 3D Box it would be: * L_box**3 / (N_grid ** 6) / (2 * np.pi) ** 3

    # Only use first half (real FFT is symmetric)
    # n_half = N_spectrum // 2
    # Pk = Pk[:n_half]
    # kz = kz[:n_half]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # sigma_filter=None
    if sigma_filter != None:
        # Pk = gaussian_filter1d(Pk, sigma=sigma_filter, axis=-1, truncate=1.)      # shape (3, N_los, N_los, N_spectrum/2)
        # sigma as resolution (cf Tohfa_2024 p.2, Croft_1997 p.11, Viel_2004 p.10)
        # Pk = uniform_filter1d(Pk, size=int(sigma_filter), axis=-1)

        # Window function is applied as a product in fourier space, not a convolution as in real space!
        Pk *= gauss_W(kz, sigma_filter) ** 2
        # Pk *= tophat_W(kz, sigma_filter)**2

    # obsolete version for lin binning
    # Pk_binned, _, _ = binned_statistic(kz, Pk, statistic='mean', bins=kbins_edges)
    # Pk_binned *= np.diff(kbins_edges)          # wrong!

    if len(kz) > len(kbins_edges):
        # general version for both lin and log binning
        counts, _, _ = binned_statistic(kz, Pk, statistic="count", bins=kbins_edges)
        Pk_sum, _, _ = binned_statistic(kz, Pk, statistic="sum", bins=kbins_edges)
        # if np.where(counts<=0,1,0).any(): raise ValueError("there are empty bins, increase number of values!")
        Pk_binned = np.where(counts > 0, Pk_sum / counts, 0.0)
    else:
        Pk_binned = Pk[:-1]
    return Pk_binned


def compute_all_1d_power_spectra(
    forests, kz, kbins_edges, L_box, N_grid, R, sigma_filter_spec, sigma_filter_PS
):
    """
    Compute 1D power spectra for all lines of sight
    Args:
        forests: Array of shape (3, N_los, N_los, N_spectrum)
        kz: the spectral bins
        kbins_edges: the wavenumbers of the edges of the bins
        L_box: normalization factor and needed for Window function. Often corresponding to vmax rather than L_box, depending on space of computation.
        R: spectral resolution Delta_lambda/d_lambda
        sigma_filter: sigma of gaussian filter used to smoothe out the power spectrum, in index units. If None, no smoothing applied.
    Returns:
        all_Pk: Array of power spectra (3, N_los, N_los, nbins)
    """
    N_los = len(forests[0, :, 0, 0])

    # Initialize output array
    all_Pk = np.empty(
        (3, N_los, N_los, N_grid // 2)
    )  # divided by 2 because k sampling for real FFT will divide by 2 (undersampling)

    mean_spectrum = np.mean(forests)
    # Compute for each line of sight
    for los in range(3):
        for i in range(N_los):
            for j in range(N_los):
                spectrum = forests[los, i, j, :]
                if spectrum.ndim != 1:
                    spectrum = spectrum.flatten()
                if np.mean(forests[los, i, j, :]) == 0 or not np.isfinite(
                    np.mean(forests[los, i, j, :])
                ):
                    raise ValueError("mean along LOS is zero or not finite!")
                else:
                    delta = (
                        spectrum / mean_spectrum
                    ) - 1.0  # dimensionless overdensity (considering only this los)

                # sigma_filter_spec = None
                # if sigma_filter_spec!=None:
                #    delta = gaussian_filter1d(delta, sigma=sigma_filter_spec, axis=-1, truncate=1.)      # shape (3, N_los, N_los, N_spectrum/2)

                all_Pk[los, i, j] = fftPower_1d(
                    delta,
                    kz,
                    kbins_edges,
                    L=L_box,
                    nbins=N_grid,
                    sigma_filter=sigma_filter_PS,
                )
                # all_Pk[los, i, j] /= _window_function(kbins_edges[:-1], R=R, dv=L_box/N_grid)**2

    return all_Pk


def oneD_to_threeD_powSpectr(P_1D, kbins_centers, sigma_filter):
    """
    We recover P(k) from P_1D(k) using Formula P(k) = -2π/k d(P1D(k))/dk, as in Croft_1997 p.8, Kaiser&Peacock_1991 p.6, Peacock p.517
    When plotting we will need to consider multiplying k^3 to P(k) in order to get the variance per ln(k), as in WeinbergDH_2003 p.5
    """
    # binSizes = [kbins_edges[i]-kbins_edges[i-1] for i in range(1,kbins_edges.shape[0])] # ------ OLD
    # P_3D = (-2*np.pi/k) * np.gradient(P_1D, binSizes, axis = 3)  # shape of P_1D: (3, N_los, N_los, N_spectrum/2) # ------ OLD
    # P_1D = gaussian_filter1d(P_1D, sigma=30, axis=-1, truncate=1.) # doesnt seem to affect anything... weird
    if sigma_filter != None:
        P_1D = gaussian_filter1d(
            P_1D, sigma=sigma_filter, axis=-1, truncate=1.0
        )  # *= gauss_W(kbins_centers, sigma_filter) ** 2
    log_k = np.log(np.clip(kbins_centers, 1e-30, None))  # Avoid log(0)
    log_P1D = np.log(np.clip(P_1D, 1e-30, None))  # Avoid log(0)
    print("shapes Pk, k: ", log_P1D.shape, log_k.shape)
    dlogP_dlogk = np.gradient(log_P1D, log_k, axis=-1, edge_order=2)
    P_3D = (
        dlogP_dlogk * P_1D / (kbins_centers)
    )  # [..., None]  # shape-safe division -----> need to divide by k^2! not only k
    P_3D *= -2 * np.pi / kbins_centers
    # P_3D = gauss_filter1d(P_3D, kbins_centers, 14.)
    return P_3D


def convert_k_to_velocity_space(k_h_per_Mpc, redshift, H_z):
    """
    Convert wavenumbers from comoving h/Mpc to velocity space (s/km) as in McDonald_2000 p.18.
    Args:
        k_h_per_Mpc: Array of k values in h/Mpc
        redshift: The redshift of the simulation
        H_z: Hubble parameter at given redshift in km/s h/Mpc
    Returns:
        k_s_per_km: k values in s/km
    """
    # Convert k from h/Mpc → s/km
    k_s_per_km = (
        k_h_per_Mpc * (1 + redshift) / H_z
    )  # Arinyio-i-Prats_2015, McDonald_2000 p.18
    return k_s_per_km


def converge(k, fk, fit, dir, shape):
    fk_tot = np.empty((8, len(k)))
    mask = (~np.isnan(fk)) & (fk > 0)
    # print("shape fk:", fk.shape, "shape mask:", mask.shape, fk[mask].shape)

    for i in range(fk.shape[-1]):  # Loop through spectrum
        for a in range(dir):
            # print("shape fk[a,:,:i]:", fk[a,:,:,i].shape, "shape mask[a,:,:i]:", mask[a,:,:,i].shape)
            # print(fk[a,:,:,i][mask[a,:,:,i]].shape)
            fk_tot[a + 1, i] = np.mean(fk[a, :, :, i][mask[a, :, :, i]])
            fk_tot[a + dir + 1, i] = np.median(fk[a, :, :, i][mask[a, :, :, i]])
        fk_tot[0, i] = np.mean(fk[:, :, :, i][mask[:, :, :, i]])  # total mean
    fk_tot[7] = fit  # include fit
    return fk_tot


def fitFunc(x, a, b, c, d, e):
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=2.75, transfer="NoWiggleEisensteinHu")
    Pk_lin = Plin(x)
    out = Pk_lin * (1 + a * x**b) ** c / (1 + d * x**e)
    return out


def fitPk(x, y):
    """
    fit Pk to data considering deviation
    Params:
        x:          x values of data set. Assume shape (N_spectrum/2)
        y:          y values of data set. Assume shape ((3, N_los, N_los, N_spectrum/2)
    Returns:
        y_fit:      y values fitted. Shape (N_spectrum/2)
        y_std:    per-point standard deviations used in weighting
        params:     optimized params for fitFunc
        cov:        covariance of the fit parameters
    """
    # mean and standard deviation across all LOS
    y_mean = np.mean(y, axis=(0, 1, 2))  # shape: (N_spectrum/2,)
    y_std = np.std(y, axis=(0, 1, 2))  # shape: (N_spectrum/2,)

    # exclude zeros in std
    # y_std = np.where(y_std < 1e-10, np.mean(y_std[y_std > 0]), y_std)

    # Perform weighted fit
    params, cov = curve_fit(
        fitFunc,
        x,
        y_mean,
        p0=None,  # initial guesses
        sigma=y_std,  # standard deviation as weights
        absolute_sigma=True,  # y_std are physical quantities
    )

    y_fit = fitFunc(x, *params)  # Unpack params with *

    return y_fit, y_std, params, cov


#################################################################################### HOW TO COMPUTE WAVELET SCATTERING TRANSFORMS
def compute_all_1d_WST(
    forests, J, Q, order, sigma_filter_spec, sigma_filter_WST, out_type="list"
):
    """
    Compute 1D WST for all lines of sight using KYMATIO
    Args:
        forests: Array of shape (3, N_los, N_los, N_spectrum)
        J: maximum log-scale of scattering transform. maximum scale is given by 2^J
        Q: Q = (Q1, Q2=1) are number of wavelets per octave for 1st and 2nd order
        order: max order of scattering coefficients (either 1 or 2)
    Returns:
        all_WST: Array of WST coefficients (B, C, N1),
            out_type='array':
                B = 3 * N_los * N_los: batch size
                C: number of scattering coefficients
                N1: signal length after subsampling to scale 2^J (with appropriate oversampling factor to reduce aliasing)
            out_type='list':
                list of dictionaries, each dictionary corresponding to a scattering coefficient and its associated meta information.
                'coef': coefficient
                'j': scale of the filter used
                'n'; the filter index
    """
    N_los = len(forests[0, :, 0, 0])
    N_spectrum = len(forests[0, 0, 0, :])
    delta = np.empty((3, N_los, N_los, N_spectrum))

    # Compute for each line of sight
    for los in range(3):
        for i in range(N_los):
            for j in range(N_los):
                spectrum = forests[los, i, j]
                if spectrum.ndim != 1:
                    spectrum = spectrum.flatten()
                delta[los, i, j] = (
                    spectrum / np.mean(forests[los])
                ) - 1.0  # dimensionless overdensity (considering only this los)
    delta = delta.reshape((3 * N_los * N_los, N_spectrum))

    # if sigma_filter_spec!=None:
    #    delta = gaussian_filter1d(delta, sigma=sigma_filter_spec, axis=-1, truncate=1.)      # shape (3, N_los, N_los, N_spectrum/2)

    S = Scattering1D(  # Tohfa_2024
        J=J,  # maximum log-scale of scattering transform. maximum scale is given by 2^J
        shape=N_spectrum,  # length of input signals
        Q=Q,  # Q = (Q1, Q2=1) are number of wavelets per octave for 1st and 2nd order
        T=sigma_filter_WST,  # temporal support of low-pass filter: imposed time-shift invariance and maximum subsampling
        max_order=order,  # max order of scattering coefficients (either 1 or 2)
        # average=None,  # whether output is averaged in time (default) or not
        oversampling=0,  # WST reduces high-frequency content of signal -> we subsample. If not wanted, set oversampling to large value
        out_type=out_type,  # ‘list’: outputs list with individual coeffs and meta information. 'array': outputs concatenated array
        backend="numpy",  #
    )

    all_WST = S.scattering(
        delta
    )  # shape WST: (ncoeffs, ["coef",             3*N_los*N_los, nsamples)         nsamples=N_spectrum/2^J
    #                       "n", "j"]           1 or 2)

    N_coeffs = len(all_WST)
    """
    for i in range(N_coeffs):
        if sigma_filter_WST!=None:
            all_WST[i]['coef'] = gaussian_filter1d(all_WST[i]['coef'], sigma=sigma_filter_WST, axis=-1, truncate=1.) # try along samples, not los...
                # sigma as resolution (cf Tohfa_2024 p.2, Croft_1997 p.11, Viel_2004 p.10)
    """
    return all_WST


def summerize_WST(WST, J, max_order):
    coeff_lengths = np.empty(max_order + 1, dtype=np.int8)
    for i in range(max_order + 1):
        coeff_lengths[i] = i * (J + 1)  # [1,6,12] for J=5, max_order=2
    coeff_lengths[0] = 1
    a = 0

    # WST has n_coeff dict elements, shape of "coef" is (3*N_los*N_los, N_spectrum/2^J)
    n_los = len(WST[0]["coef"][:, 0])
    n_coeffs = len(WST)
    # if n_coeffs!= np.sum(coeff_lengths): raise TypeError("number of coefficients not as expected")

    x_vals = np.empty(n_coeffs, dtype="<U9")
    y_means = np.empty(shape=(n_coeffs, n_los))
    y_medians = np.empty(shape=(n_coeffs, n_los))
    y_means_tot = np.empty(n_coeffs)
    y_medians_tot = np.empty(n_coeffs)

    for i in range(
        n_coeffs
    ):  # get the summarized arrays ordered after the coefficient index
        x_vals[i] = f"{WST[i]['n']}"
        y_means[i] = np.mean(
            WST[i]["coef"], axis=1
        )  # axis=0 is mean over los, 1 is over k
        y_medians[i] = np.median(WST[i]["coef"], axis=1)

    for order in range(max_order):  # do the different plots for each order separately
        xlength = coeff_lengths[order + 1]
        a += coeff_lengths[order]
        b = a + xlength
        y_means_tot[a:b] = np.mean(y_means[a:b, :], axis=1)
        y_medians_tot[a:b] = np.median(y_medians[a:b, :], axis=1)

    out = {
        "x": x_vals,
        "y_means": y_means,
        "y_medians": y_medians,
        "y_means_tot": y_means_tot,
        "y_medians_tot": y_medians_tot,
    }
    return out


#################################################################################### HOW TO PLOT
def setup_plot(ax, xlabel, ylabel, a=None):
    # forced_ticks = np.linspace(5.11e-3, 5.15e-3, num=5) #5.11-5.15-5.19e-3
    ax.set_xlabel(xlabel)
    if a == 0:
        ax.set_ylabel(ylabel)
    # ax.set_yscale("log")
    # ax.set_xscale('linear')
    # ax.set_xticks(forced_ticks)#, minor=False)
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
    ax.legend(loc="lower left")


def plotPowSpectr(
    x,
    y,
    y_tot,
    colors,
    x_label,
    y_label,
    losToPlot,
    filename,
    dirName_plot,
    title,
    savefig=False,
    summary=True,
    fit=False,
    log_binning=True,
    dir=3,
    oneD=True,
):
    """
    Plot the power spectra along x/y/z lines of sight and their means/medians.
    Args:
        x (array): k values
        y (array): Power spectra, shape (3, N_los, N_los, N_bins)
        y_tot (array): Averaged power spectra, shape (7, N_bins)
        filename (str): Label for filename
        x_units (str): "comoving" for h/Mpc, "velocity" for s/km
    """
    # N_los = len(y[0, :, 0])lotf
    if log_binning:
        x = x  # np.log10(x) #10**x
    mask = (~np.isnan(y_tot[0])) & (y_tot[0] > 0)

    if summary:
        plt.figure(num=1, figsize=(9, 12), clear=True)

        if oneD:
            y_masked = y[:, :, :, mask]  # Shape: (3, N_los, N_los, N_spectrum_masked)
            mean_per_bin = np.mean(
                y_masked, axis=(0, 1, 2)
            )  # Shape: (N_spectrum_masked,)
            std_above = np.zeros_like(mean_per_bin)
            std_below = np.zeros_like(mean_per_bin)
            for i in range(len(mean_per_bin)):
                los_values = y_masked[:, :, :, i].flatten()
                above_mean = los_values[los_values > mean_per_bin[i]]
                below_mean = los_values[los_values < mean_per_bin[i]]
                std_above[i] = (
                    np.sqrt(
                        np.mean(
                            np.clip((above_mean - mean_per_bin[i]) ** 2, 0, 1e-2 - 1e-5)
                        )
                    )
                    if len(above_mean) > 0
                    else 0
                )
                std_below[i] = (
                    np.sqrt(
                        np.mean(
                            np.clip((below_mean - mean_per_bin[i]) ** 2, 0, 1e-2 - 1e-5)
                        )
                    )
                    if len(below_mean) > 0
                    else 0
                )
                if (np.isnan(std_below[i])) or (std_below[i] < 0):
                    std_below[i] = 0
                if (np.isnan(std_above[i])) or (std_above[i] < 0):
                    std_above[i] = 0
            plt.errorbar(
                x[mask],
                y_tot[0][mask],
                yerr=(std_below, std_above),
                label="Mean overall",
                c="darkviolet",
                alpha=0.2,
            )
        for a, dir in enumerate(["x", "y", "z"]):
            for b in np.random.randint(low=0, high=len(y[0, :, 0]), size=losToPlot):
                for c in np.random.randint(low=0, high=len(y[0, 0, :]), size=losToPlot):
                    plt.scatter(x, y[a, b, c], alpha=0.3, c="grey", s=4)
            plt.loglog(
                x[(~np.isnan(y_tot[a + 1])) & (y_tot[a + 1] > 0)],
                y_tot[a + 1][(~np.isnan(y_tot[a + 1])) & (y_tot[a + 1] > 0)],
                label=f"mean in LOS {dir}",
                c=colors[a],
                alpha=0.6,
                lw=2,
            )
            """plt.loglog(
                x[(~np.isnan(y_tot[a + dir + 1])) & (y_tot[a + dir + 1] > 0)],
                y_tot[a + dir + 1][(~np.isnan(y_tot[a + dir + 1])) & (y_tot[a + dir + 1] > 0)],
                label=f"median in LOS {a}", c=colors[a + dir]
            )"""

        plt.loglog(x[mask], y_tot[0][mask], label="Mean overall", c="black", lw=2)

        if fit:
            plt.loglog(
                x[(~np.isnan(y_tot[7])) & (y_tot[7] > 0)],
                y_tot[7][(~np.isnan(y_tot[7])) & (y_tot[7] > 0)],
                label=f"fit overall",
                c="black",
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(bottom=1e-6)
        plt.legend(loc="lower left")

    else:
        _, ax = plt.subplots(1, 3, sharey=True, num=1, figsize=(14, 6), clear=True)
        for a in range(3):
            for b in np.random.randint(low=0, high=len(y[0, :, 0]), size=losToPlot):
                for c in np.random.randint(low=0, high=len(y[0, 0, :]), size=losToPlot):
                    ax[a].scatter(
                        x, y[a, b, c], alpha=0.3, c="grey", linewidth=0.5, s=4
                    )
            ax[a].loglog(
                x[(~np.isnan(y_tot[a + 1])) & (y_tot[a + 1] > 0)],
                y_tot[a + 1][(~np.isnan(y_tot[a + 1])) & (y_tot[a + 1] > 0)],
                label=f"mean in LOS {a}",
                c=colors[a],
            )
            ax[a].loglog(
                x[(~np.isnan(y_tot[a + dir + 1])) & (y_tot[a + dir + 1] > 0)],
                y_tot[a + dir + 1][
                    (~np.isnan(y_tot[a + dir + 1])) & (y_tot[a + dir + 1] > 0)
                ],
                label=f"median in LOS {a}",
                c=colors[a + dir],
            )

            ax[a].loglog(x[mask], y_tot[0][mask], label=f"mean overall", c="black")
            if fit:
                ax[a].loglog(
                    x[(~np.isnan(y_tot[7])) & (y_tot[7] > 0)],
                    y_tot[7][(~np.isnan(y_tot[7])) & (y_tot[7] > 0)],
                    label=f"fit overall",
                    c=colors[a + dir],
                )
            setup_plot(ax[a], xlabel=x_label, ylabel=y_label, a=a)

    plotSave(filename, dirName_plot, title, savefig)


def plotWS_all(
    k,
    WST,
    J,
    max_order,
    colors,
    xlabels,
    ylabels,
    losToPlot,
    filename,
    dirName_plot,
    title,
    savefig=True,
):
    """
    Plot the power spectra along x/y/z lines of sight and their means/medians.
    Args:
        x: k samples
        y: Array of WST coefficients (B, C, N1)
        B = 3 * N_los * N_los: batch size
        C: number of scattering coefficients
        N1: signal length after subsampling to scale 2^J (with appropriate oversampling factor to reduce aliasing)
    """
    coeff_lengths = np.empty(max_order + 1, dtype=np.int8)
    for i in range(max_order + 1):
        coeff_lengths[i] = i * (J + 1)  # [1,6,12] for J=5, max_order=2
    coeff_lengths[0] = 1
    offset = 0

    # WST has n_coeff dict elements, shape of "coef" is (3*N_los*N_los, N_spectrum/2^J)
    # N_los = len(WST[0]["coef"][:, 0])

    _, ax = plt.subplots(
        1, max_order + 1, sharey=False, num=1, figsize=(17, 6), clear=True
    )
    if ylabels == None:
        ylabels = [r"$S_0$", r"$S_1$", r"$S_2$"]
    if xlabels == None:
        xlabels = [
            r"$k\ [\mathrm{s/km}]$",
            r"$k\ [\mathrm{s/km}]$",
            r"$k\ [\mathrm{s/km}]$",
        ]

    for order in range(max_order + 1):
        xlength = coeff_lengths[order]
        if order >= 1:
            offset += coeff_lengths[order - 1]
        for ind in range(offset, offset + xlength):
            for los in range(losToPlot):
                ax[order].plot(
                    k,
                    WST[ind]["coef"][los, :],
                    alpha=0.3,
                    c=colors[ind % 6],
                    linewidth=0.5,
                )
        print(
            f"len WST at los {los}: ",
            WST[ind]["coef"][los, :].shape,
            "len k:",
            k.shape,
            "first WST value: ",
            WST[ind]["coef"][los, 0],
        )
        for ind in range(offset, offset + xlength):
            ax[order].plot(
                k,
                np.mean(WST[ind]["coef"], axis=0),
                label=f"coef {WST[ind]['j']}",
                c=colors[ind % 6],
                linewidth=2,
                ls="dotted",
            )
            """
            y_tot_error = [np.var(WST[ind]["coef"][:, i]) for i in range(len(WST[ind]["coef"][0, :]))]
            ax[order].errorbar(k, np.mean(WST[ind]["coef"], axis=0), y_tot_error) """

        ax[order].set_yscale("log")
        ax[order].set_ylabel(ylabels[order])
        ax[order].set_xlabel(xlabels[order])
        # ax[order].legend(loc="lower left")

    plotSave(filename, dirName_plot, title, savefig)


def plotWS_summary(
    dictToPlot,
    J,
    max_order,
    colors,
    xlabels,
    ylabels,
    losToPlot,
    filename,
    dirName_plot,
    title,
    savefig=False,
    stat=True,
):
    """
    Plot the power spectra along x/y/z lines of sight and their means/medians.
    Args:
        x: k samples
        y: Array of WST coefficients (B, C, N1)
        B = 3 * N_los * N_los: batch size
        C: number of scattering coefficients
        N1: signal length after subsampling to scale 2^J (with appropriate oversampling factor to reduce aliasing)
    """
    coeff_lengths = np.empty(max_order + 1, dtype=np.int8)
    for i in range(max_order + 1):
        coeff_lengths[i] = i * (J + 1)  # [1,6,12] for J=5, max_order=2
    coeff_lengths[0] = 1
    a = 0

    _, ax = plt.subplots(1, max_order, sharey=False, num=1, figsize=(17, 6), clear=True)
    if ylabels == None:
        ylabels = [r"$S_1$", r"$S_2$"]
    if xlabels == None:
        xlabels = [r"$j_1$", r"$(j_1, j_2)$"]

    x = dictToPlot["x"]
    if stat:  # True means means, False means medians
        y = dictToPlot["y_means"]
        y_tot = dictToPlot["y_means_tot"]
    else:
        y = dictToPlot["y_medians"]
        y_tot = dictToPlot["y_medians_tot"]

    std_above = np.zeros_like(y_tot)
    std_below = np.zeros_like(y_tot)
    for i in range(len(y_tot)):
        above = y[i, :][y[i, :] > y_tot[i]]  # y[:, i][y[:, i] > y_tot[i]]
        below = y[i, :][y[i, :] < y_tot[i]]
        std_above[i] = (
            np.sqrt(np.mean((above - y_tot[i]) ** 2)) if len(above) > 0 else 0
        )
        std_below[i] = (
            np.sqrt(np.mean((below - y_tot[i]) ** 2)) if len(below) > 0 else 0
        )

    for order in range(max_order):  # do the different plots for each order separately
        xlength = coeff_lengths[order + 1]
        a += coeff_lengths[order]
        b = a + xlength
        # print("a: ", a, "b: ", b)
        ax[order].errorbar(
            x[a:b],
            y_tot[a:b],
            yerr=(std_below[a:b], std_above[a:b]),
            label=r"Standard deviation",
            c="darkviolet",
            alpha=0.3,
            fmt="none",
        )
        for i in np.random.randint(low=0, high=len(y[0, :]), size=losToPlot):
            # y_means summarized all LOS, because the shape of WST is (3*N_los*N_los, n_coeffs, N_spectrum/2^J)
            # I want to plot the n_coeff y_means array for every different "k" value from the spectrum
            # but i want 1st order coefficients and 2nd order ones on separate plots --> xlength&a
            # print(x.shape, y_tot.shape, std_above.shape)
            ax[order].scatter(
                x[a:b], y[a:b, i], alpha=0.3, c=colors[i % 6], s=6
            )  # , label=f"mean sample {i}")
        ax[order].scatter(
            x[a:b], y_tot[a:b], c="black", marker="x", label=r"Mean over los"
        )  # plot mean wrt "k" of means wrt LOS
        ax[order].set_ylabel(ylabels[order])
        ax[order].set_xlabel(xlabels[order])
    ax[1].tick_params(axis="x", labelrotation=90)

    # ax[order].set_yscale("log")
    # ax[order].legend(loc="lower left")

    plotSave(filename, dirName_plot, title, savefig)


# ====================================================================================================================== FISHING
#################################################################################### HOW TO CONSTRAIN
def plotTogether(
    x,
    y,
    filename,
    dirName_plot,
    labels,
    x_label,
    y_label,
    title,
    loglog,
    ylim,
    savefig=False,
):
    plt.figure()
    print(len(y))
    for i in range(len(y)):
        mask = True  # (~np.isnan(y[i])) & (y[i] > 0)
        plt.scatter(
            x[i][mask],
            y[i][mask],
            c=["green", "blueviolet", "red", "dodgerblue"][i],
            label=labels[i],
            alpha=0.7,
            lw=1,
            marker=[".", "*", "+", "x"][i],
        )  # , s=2)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend(loc="lower left")
    plt.ylim(ylim)

    plotSave(filename, dirName_plot, title, savefig)


def plotDeviations(
    PSx, PSdevs, WSTx, WSTdevs, filename, dirName_plot, title, savefig=False
):
    _, ax = plt.subplots(1, 2, sharey=False, num=1, figsize=(22, 8), clear=True)
    ylabels = [
        r"$S_{1/2}/S_{1/2}^{\mathrm{fid}} - 1$",
        r"$P(k)/P^{\mathrm{fid}}(k) - 1$",
    ]
    xlabels = [r"$(j_1, j_2)$", r"$k$ [s/km]"]

    if len(WSTdevs.shape) > 1:
        for i in range(WSTdevs.shape[1]):
            ax[0].scatter(
                WSTx, WSTdevs[:, i], c="green", marker="x"
            )  # plot mean wrt "k" of means wrt LOS
    else:
        ax[0].scatter(
            WSTx, WSTdevs, c="green", marker="x"
        )  # plot mean wrt "k" of means wrt LOS

    ax[0].set_ylabel(ylabels[0])
    ax[0].set_xlabel(xlabels[0])
    ticks = get_ticks(len(WSTx), 2)
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels([str(WSTx[i]) for i in ticks], rotation=90)
    # ax[0].set_yscale("log")
    # ax[0].legend(loc="lower left")

    if len(PSdevs.shape) > 1:
        for i in range(PSdevs.shape[1]):
            ax[1].scatter(
                PSx, PSdevs[:, i], c="red", marker="x"
            )  # plot mean wrt "k" of means wrt LOS
            slope, offset, _, _, _ = stats.linregress(PSx, PSdevs[:, i])
            linApprox = list(slope * PSx + offset)
            ax[1].plot(PSx, linApprox, c="black")
    else:
        ax[1].scatter(
            PSx, PSdevs, c="red", marker="x"
        )  # plot mean wrt "k" of means wrt LOS
        slope, offset, _, _, _ = stats.linregress(PSx, PSdevs)
        linApprox = list((slope * PSx + offset))
        ax[1].plot(PSx, linApprox, c="black")

    ax[1].set_ylabel(ylabels[1])
    ax[1].set_xlabel(xlabels[1])
    # ax[1].set_yscale("log")
    # ax[1].legend(loc="lower left")

    plotSave(filename, dirName_plot, title, savefig)


def dS_dTheta(S, S_prime, deltaTheta):
    delS_delTheta = (S_prime - S) / deltaTheta
    return delS_delTheta


def Fisher_ab(dS_dTheta_a, dS_dTheta_b, Cov, H_factor):
    Cov_inv = np.linalg.inv(Cov) * H_factor
    term = (
        Cov_inv @ dS_dTheta_b
    )  # (n_bins, n_bins) @ (n_bins, n_params) -> (n_bins, n_params)
    fisher_ab = (
        dS_dTheta_a.T @ term
    )  # (n_params, n_bins) @ (n_bins, n_params) -> (n_params, n_params)

    print(
        "covinv: ", Cov_inv.shape, "term: ", term.shape, "fisher_ab: ", fisher_ab.shape
    )

    return fisher_ab


"""
def Fisher_ab(dS_dTheta_a, dS_dTheta_b, Cov, H_factor):
    Cov_inv = inv(Cov) * H_factor
    firstProd = np.matmul(Cov_inv, dS_dTheta_b.T)
    fisher_ab = np.matmul(dS_dTheta_a, firstProd)
    return fisher_ab
"""


def getConstrainingPower(stat, stat_prime, param_var, cov_matrix):
    delS_delTheta = (stat_prime - stat) / param_var

    H_fact = 1
    fish = Fisher_ab(delS_delTheta, delS_delTheta, cov_matrix, H_fact)

    # Regularize Fisher matrix to avoid singularity
    # reg = 1e-6 * np.trace(fish) / fish.shape[0]
    fish_reg = fish  # + reg * np.eye(fish.shape[0])

    print(fish_reg.shape)
    if fish_reg.ndim > 1:
        try:
            fish_inv = np.linalg.inv(fish_reg)
        except np.linalg.LinAlgError:
            fish_inv = np.linalg.pinv(fish_reg)
        sigma = np.sqrt(np.diag(fish_inv))
    else:
        print("I'm a 1D fish ", fish_reg)
        fish_inv = 1 / fish_reg
        sigma = np.sqrt(fish_inv)

    print("dS_dTheta: ", delS_delTheta.shape)
    return sigma


'''
def getConstrainingPower(stat, stat_prime, param_var):
    """
    A function to get the constraint of a parameter, being given the variation of the same and the caused increment for a statistical method
    Computes the covariance matrix of the observable (statistics) and from that, the Fisher matrix.

    Args:
        stat            the unvaried observable (S)
        stat_prime      the varied observable (S_prime)
        param_var       the applied parameter variation (delta_Theta)
    Returns:
        constraint      the parameter variation that the considered observable is capable of detecting (??)
    """
    delS_delTheta = dS_dTheta(stat, stat_prime, param_var)
    cov_stat = np.cov(stat, rowvar=False)
    cov_stat_prime = np.cov(stat_prime, rowvar=False)
    H_fact = 1 # (N_noise-N_bins-2)/(N_noise-1) ??? Nina
    fish = Fisher_ab(delS_delTheta, delS_delTheta, cov_stat, H_fact)

    sigma = np.sqrt(np.diag(inv(fish))) # - dS_dTheta(cov_stat, cov_stat_prime, param_var)

    return sigma'''


def plotCovCorr(kPk, Pk, kWST, WST, J, dirName_plot, start_time):
    # Precompute covariance matrices from multiple realizations
    # Pk_ensemble shape: (n_realizations, 128)
    # WST_ensemble shape: (n_realizations, 19)
    cov_Pk = np.cov(np.log(Pk[0]).T, rowvar=False)
    cov_WST = np.cov(WST[0].T, rowvar=False)

    CorrPk = np.zeros(cov_Pk.shape)
    for i in range(cov_Pk.shape[0]):
        for j in range(cov_Pk.shape[1]):
            CorrPk[i, j] = cov_Pk[i, j] / np.sqrt(cov_Pk[i, i] * cov_Pk[j, j])

    CorrWST = np.zeros(cov_WST.shape)
    for i in range(cov_WST.shape[0]):
        for j in range(cov_WST.shape[1]):
            CorrWST[i, j] = cov_WST[i, j] / np.sqrt(cov_WST[i, i] * cov_WST[j, j])

    fig, axs = plt.subplots(2, 3, figsize=(18, 13))
    axs = axs.ravel()

    im1 = axs[0].imshow(cov_Pk, cmap="seismic")
    axs[0].set_title(r"Cov $P(k)$")
    ticks = get_ticks(len(kPk), 64)
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels([f"{kPk[i]:.2f}" for i in ticks], rotation=90)
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([f"{kPk[i]:.2f}" for i in ticks])
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(cov_WST[:J, :J], cmap="seismic")
    axs[1].set_title(r"Cov $S_1(j_1)$")
    ticks = get_ticks(J, 2)
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels([str(kWST[i]) for i in ticks], rotation=90)
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels([str(kWST[i]) for i in ticks])
    fig.colorbar(im2, ax=axs[1])

    s2_len = len(kWST[J:])
    im3 = axs[2].imshow(cov_WST[J:, J:], cmap="seismic")
    axs[2].set_title(r"Cov $S_2(j_1,j_2)$")
    ticks = get_ticks(s2_len, 4)
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels([str(kWST[J + i]) for i in ticks], rotation=90)
    axs[2].set_yticks(ticks)
    axs[2].set_yticklabels([str(kWST[J + i]) for i in ticks])
    fig.colorbar(im3, ax=axs[2])

    im4 = axs[3].imshow(CorrPk, cmap="seismic", vmin=-1, vmax=1)
    axs[3].set_title(r"Corr $P(k)$")
    ticks = get_ticks(len(kPk), 64)
    axs[3].set_xticks(ticks)
    axs[3].set_xticklabels([f"{kPk[i]:.2f}" for i in ticks], rotation=90)
    axs[3].set_yticks(ticks)
    axs[3].set_yticklabels([f"{kPk[i]:.2f}" for i in ticks])
    fig.colorbar(im4, ax=axs[3])

    im5 = axs[4].imshow(CorrWST[:J, :J], cmap="seismic", vmin=-1, vmax=1)
    axs[4].set_title(r"Corr $S_1(j_1)$")
    ticks = get_ticks(J, 2)
    axs[4].set_xticks(ticks)
    axs[4].set_xticklabels([str(kWST[i]) for i in ticks], rotation=90)
    axs[4].set_yticks(ticks)
    axs[4].set_yticklabels([str(kWST[i]) for i in ticks])
    fig.colorbar(im5, ax=axs[4])

    im6 = axs[5].imshow(CorrWST[J:, J:], cmap="seismic", vmin=-1, vmax=1)
    axs[5].set_title(r"Corr $S_2(j_1,j_2)$")
    ticks = get_ticks(s2_len, 4)
    axs[5].set_xticks(ticks)
    axs[5].set_xticklabels([str(kWST[J + i]) for i in ticks], rotation=90)
    axs[5].set_yticks(ticks)
    axs[5].set_yticklabels([str(kWST[J + i]) for i in ticks])
    fig.colorbar(im6, ax=axs[5])

    # plt.tight_layout()  # Adjust layout to prevent overlap
    plotSave("covCorr", dirName_plot, "title", savefig=True)


def getFish(stat, param_var, n_params):
    """
    A plotter function for the Fisher matrix
    Args:
        stat            A 2D array containing multiple variables and observations.
                        Each row represents a variable, and each column a single observation of all those variables.
        param_var       the applied parameter variation (delta_Theta)
        n_params        number of varied parameters
    Returns:
        fish            A 2D array containing the Fisher information matrix for the different variables
    """
    cov_stat = np.cov(stat, rowvar=False)
    H_fact = 1  # (N_noise-N_bins-2)/(N_noise-1) ??? Nina
    fish = np.empty((n_params, n_params))
    for i in range(1, n_params):
        for j in range(1, n_params):
            dS_dTheta_a_array = dS_dTheta(
                stat[0, :], stat[i, :], param_var[i]
            )  # 0 is fid, i is varied
            dS_dTheta_a = np.mean(dS_dTheta_a_array)
            print(
                "PARAM ALPHA: mean: ",
                dS_dTheta_a,
                "mean absolute variance: ",
                np.mean(np.abs(np.var(dS_dTheta_a_array))),
                "max absolute variance: ",
                np.max(np.abs(np.var(dS_dTheta_a_array))),
            )

            dS_dTheta_b_array = dS_dTheta(
                stat[0, :], stat[j, :], param_var[j]
            )  # 0 is fid, i is varied
            dS_dTheta_b = np.mean(dS_dTheta_b_array)
            print(
                "PARAM BETA: mean: ",
                dS_dTheta_b,
                "mean absolute variance: ",
                np.mean(np.abs(np.var(dS_dTheta_b_array))),
                "max absolute variance: ",
                np.max(np.abs(np.var(dS_dTheta_b_array))),
            )

            fish[i, j] = Fisher_ab(dS_dTheta_a, dS_dTheta_b, cov_stat, H_fact)
    return fish


def plotFish(fish, filename, dirName_plot, title, savefig=False):
    plt.figure(num=1, figsize=(14, 6), clear=True)
    plt.imshow(fish)
    plotSave(filename, dirName_plot, title, savefig)
