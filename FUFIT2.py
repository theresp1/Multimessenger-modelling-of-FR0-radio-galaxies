# import numpy, astropy and matplotlib for basic functionalities
from pickle import TRUE
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
    p1 = Parameter("p1", 2.1, min=1.0, max=3.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=5)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=6)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=0.1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=10000)
    log10_B = Parameter("log10_B", -2, min=-3, max=1.0)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)
    ######
    #gamma = Parameter("Gamma", 2 ,min=1, max = 10 )

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
        #gamma,
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
            #gamma,
            ssa = True
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
            #gamma,
            ssa =  False   # dont need this for the SSC
        )
        sed = sed_synch + sed_ssc
        return (sed / energy ** 2).to("1 / (cm2 eV s)")

df_1                = pd.read_csv('new_Tol_values3.csv')
x_v                 = df_1["frequency"].copy()
y_v                 = df_1["flux"].copy()
negative_error_bar  = df_1["flux_errn"].copy()
positive_error_bar  = df_1["flux_errp"].copy()
energy_min          = df_1["energy_min"].copy()
energy_max          = df_1["energy_max"].copy()
upper_limit         = df_1["e2dnde_ul"].copy()

#error_bars = df_1["flux_error"].copy()
#upper_limits = df_1["e2dnde_ul"].copy()

# make the x and y values into numpy arraysv (with the rest of the values)
x_v_b               = np.array(x_v)          #*u.Hz
y_v_b               = np.array(y_v)          #* (u.erg/( u.s * u.cm *u.cm ))
n_e_b               = np.array(negative_error_bar)
p_e_b               = np.array(positive_error_bar)
e_min               = np.array(energy_min)
e_max               = np.array(energy_max)
upper_limit_array   = np.array(upper_limit)

#errorbars_b         = np.array(error_bars)   #* (u.erg/( u.s * u.cm *u.cm ))
#upper_limits_b      = np.array(upper_limits) #* (u.erg/( u.s * u.cm *u.cm ))

# convert x_v_b from Hz to TeV
def convert_to_Tev(Xarray): 
    ReturnArray = np.zeros(np.size(Xarray))
    for i in range(np.size(Xarray)):
        ReturnArray[i] = (Xarray[i] * u.Hz).to(u.TeV,equivalencies=u.spectral()).value

    return ReturnArray


x_values     = convert_to_Tev(x_v_b)
e_min_values = convert_to_Tev(e_min)
e_max_values = convert_to_Tev(e_max)

#print(np.zeros(len(x_values)))

table = Table()
table["e_ref"]        = x_values * u.TeV
table["e2dnde"]       = y_v_b * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_err"]   = np.zeros(len(x_values)) *  u.erg/( u.s * u.cm *u.cm)
table["e2dnde_errn"]  = n_e_b * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_errp"]  = p_e_b * u.erg/( u.s * u.cm *u.cm)
table["e_min"]        = e_min_values * u.TeV
table["e_max"]        = e_max_values * u.TeV
table["e2dnde_ul"]    = upper_limit_array * u.erg/( u.s * u.cm *u.cm)
table["is_ul"]        = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False]



#table["e2dnde_err"] = errorbars_b  * u.erg/( u.s * u.cm *u.cm)
#table["e2dnde_ul"] = upper_limits_b * u.erg/( u.s * u.cm *u.cm)
#table["is_ul"] = [True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False]
#table["e2dnde_ul"] = np.array(y_v_b[0: 
#y_v_b * u.erg/( u.s * u.cm *u.cm)
#table["energy_max"] = np.zeros(np.size(x_values)) * u.TeV
#table["energy_min"] = np.zeros(np.size(x_values)) * u.TeV
table.meta["SED_TYPE"] = "e2dnde"

#print("table", table)

flux_points = FluxPoints(table)
#print("table", flux_points)
#flux_points.plot()
#plt.show()

# array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 30% on VHE gamma-ray instruments
# - 10% on HE gamma-ray instruments
# - 10% on X-ray instruments
# - 5% on lower-energy instruments
x = flux_points.table["e_ref"]
y = flux_points.table["e2dnde"]
y_err_stat = (flux_points.table["e2dnde_errn"] +  flux_points.table["e2dnde_errp"])/2
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
y_err_syst[vhe_gamma] = 0.10
y_err_syst[he_gamma] = 0.10
y_err_syst[x_ray] = 0.10
y_err_syst[uv_to_radio] = 0.05
y_err_syst = y * y_err_syst
# sum in quadrature the errors
## TODO: CHECK IF YOU NEED TO ADD THIS, OR IF THIS IS ALREADY INCLUDED IN THE DATA 
flux_points.table["e2dnde_err"] = np.sqrt(y_err_stat ** 2 + y_err_syst ** 2)
# convert to "dnde" SED type to fit
flux_points = flux_points.to_sed_type("dnde")

##vavlues set by eye
v_c = 5.38 *10**21
v_s = 7.93 *10**13

#Values set by Tavecchio
Radius = 5 *10**16 * u.cm

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
agnpy_ssc.delta_D.quantity = 1.4500e+00
agnpy_ssc.delta_D.frozen = True
agnpy_ssc.log10_B.quantity =  -5.9768e-01 
#agnpy_ssc.log10_B.quantity = np.log10( 1/(6.6466) * (1+z) * (v_s**2)/(2.8*10**6* v_c)  )
agnpy_ssc.log10_B.frozen = True
#agnpy_ssc.t_var. quantity = 1 * u.d
#agnpy_ssc.t_var.quantity = (Radius * (1+z))/(c.cgs *2) 
agnpy_ssc.t_var.quantity = 1.5412e+06 
#agnpy_ssc.t_var.quantity = 4.8970e+06
agnpy_ssc.t_var.frozen = False
#agnpy_ssc.gamma.quantity = 2
#agnpy_ssc.gamma.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -3.3682e+00  
agnpy_ssc.log10_k_e.frozen = False
agnpy_ssc.p1.quantity = 2.8554e+00 
agnpy_ssc.p2.quantity = 3.4728e+00 
agnpy_ssc.p1.frozen = True
agnpy_ssc.p2.frozen = True
agnpy_ssc.log10_gamma_b.quantity = 3.8783e+00
agnpy_ssc.log10_gamma_b.frozen = True
agnpy_ssc.log10_gamma_min.quantity = 2.6482e+00 
agnpy_ssc.log10_gamma_min.frozen = True
agnpy_ssc.log10_gamma_max.quantity = 4.4912e+00 - 0.01
agnpy_ssc.log10_gamma_max.frozen = True


# define model
model = SkyModel(name="Mrk421_SSC", spectral_model=agnpy_ssc)
dataset_ssc = FluxPointsDataset(model, flux_points)
#print("dataset", dataset_ssc)
# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
dataset_ssc.mask_fit = dataset_ssc.data.energy_ref > E_min_fit


# define the fitter
fitter = Fit([dataset_ssc])
results = fitter.run(optimize_opts={"print_level": 1})
print(results)
print(agnpy_ssc.parameters.to_table())
# plot best-fit model
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
#plt.savefig("FU2_FIT7/Fit8.png")
#plt.show()

#agnpy_ssc.covariance.plot_correlation()
#plt.savefig("FU2_FIT1/Base_correlation.png")
plt.show()
