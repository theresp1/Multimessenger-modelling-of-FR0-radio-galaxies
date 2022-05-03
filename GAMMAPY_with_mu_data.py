# import numpy, astropy and matplotlib for basic functionalities
from ast import Return
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label



# import gammapy classes
from gammapy.modeling.models import (
    SpectralModel,
    Parameter,
    SPECTRAL_MODEL_REGISTRY,
    SkyModel,
)
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset
from gammapy.modeling import Fit



import pandas as pd
from astropy.table import QTable
from gammapy.modeling.models import PowerLawSpectralModel

class AgnpySSC(SpectralModel):
    """Wrapper of agnpy's synchrotron and SSC classes.
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters
    whose range is expected to cover several orders of magnitudes (normalisation,
    gammas, size and magnetic field of the blob).
    """

    tag = "SSC"
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=10)
    p1 = Parameter("p1", 2.1, min=-2.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=6)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=8)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=40)
    log10_B = Parameter("log10_B", -1, min=-4, max=2)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)

    @staticmethod
    def evaluate(
        energy,
        log10_k_e,
        p1,
        p2,
        log10_gamma_b,
        log10_gamma_min,
        log10_gamma_max,
        z,
        d_L,
        delta_D,
        log10_B,
        t_var,
    ):
        # conversions
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")

        nu = energy.to("Hz", equivalencies=u.spectral())
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
        )
        sed = sed_synch + sed_ssc
        return (sed / energy ** 2).to("1 / (cm2 eV s)")


# IMPORTANT: add the new custom model to the registry of spectral models recognised by gammapy
SPECTRAL_MODEL_REGISTRY.append(AgnpySSC)


## LOAD THE FLUX POINTS
df_1 = pd.read_csv('values3.csv')
x_v = df_1["energy"].copy()
y_v = df_1["flux"].copy()
error_bars = df_1["flux_error"].copy()
upper_limits = df_1["e2dnde_ul"].copy()


# make the x and y values into numpy arrays
x_v_b               = np.array(x_v)  #*u.Hz
y_v_b               = np.array(y_v)  # * (u.erg/( u.s * u.cm *u.cm ))
errorbars_b         = np.array(error_bars)   #* (u.erg/( u.s * u.cm *u.cm ))
upper_limits_b      = np.array(upper_limits) #* (u.erg/( u.s * u.cm *u.cm ))
# convert x_v_b from Hz to TeV
def convert_to_Tev(Xarray): 
    ReturnArray = np.zeros(np.size(Xarray))
    for i in range(np.size(Xarray)):
        ReturnArray[i] = (Xarray[i] * u.Hz).to(u.TeV,equivalencies=u.spectral()).value

    return ReturnArray


x_values = convert_to_Tev(x_v_b)

table = Table()
table["e_ref"] = x_values * u.TeV
table["e2dnde"] = y_v_b * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_err"] = errorbars_b * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_ul"] = upper_limits_b * u.erg/( u.s * u.cm *u.cm)
table["is_ul"] = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
#table["energy_max"] = np.zeros(np.size(x_values)) * u.TeV
#table["energy_min"] = np.zeros(np.size(x_values)) * u.TeV
table.meta["SED_TYPE"] = "e2dnde"

flux_points = FluxPoints.from_table(table,sed_type="e2dnde")
print("flux points", flux_points)
flux_points.plot(sed_type="e2dnde")
plt.show()
#print("table: ", table)



## ADD SYSTEMATIC ERRORS 

#array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 30% on VHE gamma-ray instruments
# - 10% on HE gamma-ray instruments
# - 10% on X-ray instruments
# - 5% on lower-energy instruments
x = x_values * u.TeV
y = y_v_b * u.erg/( u.s * u.cm *u.cm)
y_err_stat = errorbars_b * u.erg/( u.s * u.cm *u.cm)
y_err_syst = np.zeros(len(x))
# define energy ranges
e_vhe = 100 * u.GeV
e_he = 0.1 * u.GeV
e_x_ray_max = 300 * u.keV
e_x_ray_min = 0.3 * u.keV
vhe_gamma = x >= e_vhe
he_gamma = (x >= e_he) * (x < e_vhe)
x_ray = (x >= e_x_ray_min) * (x < e_x_ray_max)
uv_to_radio = x < e_x_ray_min
# declare systematics
y_err_syst[vhe_gamma] = 0.30
y_err_syst[he_gamma] = 0.10
y_err_syst[x_ray] = 0.10
y_err_syst[uv_to_radio] = 0.05
y_err_syst = y * y_err_syst
# sum in quadrature the errors
table["e2dnde_err"] = np.sqrt(y_err_stat ** 2 + y_err_syst ** 2)
# convert to "dnde" SED type to fit
flux_points = FluxPoints.from_table(table,sed_type="e2dnde")
print("FLUX POINTS: ", flux_points)
flux_points_n = flux_points.to_table("dnde")
print("FLUX POINTS NEW: ", flux_points_n)

## PERFORM THE FIT 

# declare a model
agnpy_ssc = AgnpySSC()
# initialise parameters
z = 0.0284
d_L = Distance(z=z).to("cm")
# - AGN parameters
agnpy_ssc.z.quantity = z
agnpy_ssc.z.frozen = True
agnpy_ssc.d_L.quantity = d_L
agnpy_ssc.d_L.frozen = True
# - blob parameters
agnpy_ssc.delta_D.quantity = 18
agnpy_ssc.log10_B.quantity = -1.3
agnpy_ssc.t_var.quantity = 1 * u.d
agnpy_ssc.t_var.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -7.9
agnpy_ssc.p1.quantity = 2.02
agnpy_ssc.p2.quantity = 3.43
agnpy_ssc.log10_gamma_b.quantity = 5
agnpy_ssc.log10_gamma_min.quantity = np.log10(500)
agnpy_ssc.log10_gamma_min.frozen = True
agnpy_ssc.log10_gamma_max.quantity = np.log10(1e6)
agnpy_ssc.log10_gamma_max.frozen = True


# define model
model = SkyModel(name="T0L1326-379", spectral_model=agnpy_ssc)
dataset_ssc = FluxPointsDataset(model, flux_points_n)
# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
dataset_ssc.mask_fit = dataset_ssc.data.energy_ref > E_min_fit

 # %%time
# define the fitter
fitter = Fit([dataset_ssc])
results = fitter.run(optimize_opts={"print_level": 1})
print(results)
print(agnpy_ssc.parameters.to_table())


## VISUALIZE THE FIT RESULT 
# plot best-fit model
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.show()

# we can use Gammapy functions to plot the covariance
agnpy_ssc.covariance.plot_correlation()
plt.show()