# HOW DOES foresting.py WORK?

# HOW IS MY CODE STRUCTURED
## Density field (using nbodykit)
- Build box with regularly ordered k-values, give every point on the grid a magnitude (abs value of its position in fourier space) and a uniform random phase (0 to 2 $\pi$)
- Transform the magnitudes according to the given power spectrum (which will be affected by the Eisenstein&Hu Transfer function $T(k)$ and a $\Lambda$CDM growth function $D(z)$ first, but might be replaced by ULA transferfunctions as in Marsh_2016 p.40) as in $P(|k|)$ (Ryden p.236)
- Inverse Fourier Transform to get perturbations in real space (Ryden p.236)
- Apply lognormal transform according to Jones&Coles 1991 to get the density field in real space
    - this is a procedure that provenly (see nbodykit_compare folder) yields fields as in nbodykit (lognormal catalog, than .to_mesh() usual procedure to get contrasts)
    - this is also a provenly functioning procedure, while the one including the box-muller algorithm didn't yield the looked for scale dependency (aka power spectrum, cf same folder)

## IGM properties
- temperature field is approximated using the temperature-density relation $T(\rho) = T_0 (\rho/\langle \rho \rangle)^{\gamma-1}$
as in Tohfa_2024, Garzilli_2021, Viel_2004, Gaikwad_2020 (with the according variation) etc. with $T_0 \approx 4 \cdot 10^4 K$ and $\gamma \approx 1.6$.
    - the temperature at mean density is redshift corrected $T_0(z) = T_0 (1+z)$ as in McQuinn_2016 p.16
- the neutral hydrogen fractions can be computed with different levels of approximation, I implemented 2 of them for 1 < z < 5.5:
    - $x_{\text{HI}} = \frac{\alpha_A n_e}{\Gamma} = \frac{\rho}{\langle \rho \rangle} \frac{((1+z)/4)^3}{3^\cdot 10^4 \cdot 10^{10}}$ as in McQuinn_2016 p.7f, 16, 24, as well as $x_{\text{HI}} = 9.6\times 10^{-6}\, \Delta \frac{(1+\chi_{\text{He}})}{\Gamma_{-12}}\left(\frac{T}{10^{4}\text{K}}\right)^{-0.72}\left(\frac{\Omega_{\text{b}} h^2}{0.022}\right) \left(\frac{1-Y}{0.76}\right)\left(\frac{1+z}{5}\right)^{3}$ as in Bolton&Becker_2015 p.5 (assuming hydrogen ionization only $Y, \chi_{\text{He}}=0$, which is silly). Moreover Bird_2015 p.3 states; that neutral DLA's self shielding can be successfully modelled as a "Heaviside" function at a given density of $n_{\text{HI}} \approx 10^{−2} cm^{−3}$, which could be applied just as easily
    - since fake_spectra didn't read temperature but instead the internal energy of the tracers, which again depends on the neutral fraction, as well as on the metallicity and the ionization state of helium etc., using it became very intransperent and i decided to go for my own implementation as described here
- the peculiar velocity field is easily derived from Agrawal_2017 p.9 $\vec{v}_{\text{pec}}(\vec{k})=i H f \frac{\vec{k}}{k^2} \delta_m(\vec{k})$, which is applied by nbodykit itself, of course we than add the Hubble flow derived from the position of the gridpoints in the comoving mesh space $\vec{v}_{\text{H}} = H(z) \vec{x}$
- the voigt profile is used to consider both the gaussian thermal broadening that we will derive from the defined temperatures and the lorentzian distributed probability of absorption or emission of a photon by a neutral hydrogen atom as described by gamma natural, which we find by
$\gamma_{\text{nat}}=\frac{\lambda_\alpha^2 \cdot A_{21}}{2 \pi c^2}$,
with the Einstein coefficient $A_{21}$.

## Voigt absorption profile
the Voigt absorption is a convolution of the Gaussian thermal broadening and the Lorentzian due to the absorption probability of neutral hydrogen. We use the scipy.wofz error function to find its values. The computation includes finding the wavelength corresponding to the velocity of the considered tracer (including hubble flow and peculiar velocity) according to $\lambda_{\text{obs}} = \lambda_\alpha \cdot \sqrt{\frac{1+\beta}{1-\beta}}$. The peak of the absorption is derived by Viel_2004 p.5 absorption peak formula $\tau_{\text{peak}} = \tau_0 \cdot \left( \frac{\rho}{\langle \rho \rangle} \right)^{\tau_s}$, with $\tau_0 \approx 0.03$ and $\tau_s \approx 2.7 \cdot \gamma$.
