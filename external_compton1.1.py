## PLotting each specific component 

# import numpy, astropy and matplotlib for basic functionalities
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from astropy.constants import k_B, m_e, c, G, M_sun

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus, SphericalShellBLR
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

# constants
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)



import pandas as pd
from astropy.table import QTable
from gammapy.modeling.models import PowerLawSpectralModel


# Global variables
z = 2.8400e-02
Gamma   = 3



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
y_err_stat = (flux_points.table["e2dnde_errn"] + flux_points.table["e2dnde_errp"])/2  
#print("initial stat error",y_err_stat)

unc_opt = 0.05
unc_x   = 0.1 
y_err_stat[0] = y[0] * unc_opt
y_err_stat[1] = y[1] * unc_opt
y_err_stat[2] = y[2] * unc_opt
y_err_stat[3] = y[3] * unc_opt
y_err_stat[4] = y[4] * unc_opt
y_err_stat[5] = y[5] * unc_opt
y_err_stat[6] = y[6] * unc_opt
y_err_stat[7] = y[7] * unc_opt
y_err_stat[8] = y[8] * unc_opt
#y_err_stat[9] = y[9] * unc
y_err_stat[10] = y[10] * unc_x
#y_err_stat[11] = y[11] * unc
#y_err_stat[12] = y[12] * unc
#print("y err stat: ", y_err_stat)
y_err_syst = np.zeros(len(x))
# define energy ranges
e_vhe = 100 * u.GeV
e_he  = 0.1 * u.GeV
e_x_ray_max = 300 * u.keV
e_x_ray_min = 0.3 * u.keV
e_optical_max = (3000 * 1e12 * u.Hz).to(u.eV,equivalencies=u.spectral()) # 3000 THz
e_optical_min = (300 * 1e9* u.Hz).to(u.eV,equivalencies=u.spectral()) ##300GHz
#print("e_optical_max",e_optical_max.to(u.keV))
vhe_gamma = x >= e_vhe
he_gamma = (x >= e_he) * (x < e_vhe)
x_ray = (x >= e_x_ray_min) * (x < e_x_ray_max)
#uv_to_radio = x < e_x_ray_min
optical = (x >= e_optical_min) * (x < e_optical_max)
micro_to_radio =  x < e_optical_min
# declare systematics
y_err_syst[he_gamma] = 0.15
#y_err_syst[x_ray] = 0.10
#y_err_syst[optical] = 0.7
#y_err_syst[micro_to_radio] = 0.05
#y_err_syst[uv] = 0.05
#print("y_err_syst", y_err_syst) 
## add something as statistical error for radio and optical 
y_err_syst = y * y_err_syst
# sum in quadrature the errors
## TODO: CHECK IF YOU NEED TO ADD THIS, OR IF THIS IS ALREADY INCLUDED IN THE DATA 
flux_points.table["e2dnde_err"] = np.sqrt(y_err_stat ** 2 + y_err_syst ** 2)
# convert to "dnde" SED type to fit
flux_points = flux_points.to_sed_type("dnde")



# define the emission region and the thermal emitters
k_e = 10 ** -3.2721e+00 * u.Unit("cm-3")
p1 = 1.0005e+00
p2 = 4.1000e+00
gamma_b = 10 ** 3.4150e+00
gamma_min = 10 ** 1.6990e+00
gamma_max = 10 ** 5.7709e+00
B = 10 ** -6.9897e-01* u.G
r = 10 **1.8699e+01* u.cm
delta_D = 4.0000e+00
R_b =  5 *10**16 * u.cm
# blob definition
parameters = {
    "p1": p1,
    "p2": p2,
    "gamma_b": gamma_b,
    "gamma_min": gamma_min,
    "gamma_max": gamma_max,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
blob = Blob(
    R_b,
    z,
    delta_D,
    Gamma,
    B,
    k_e,
    spectrum_dict,
    spectrum_norm_type="differential",
    gamma_size=500,
)
#print(blob)
#print(f"jet power in particles: {blob.P_jet_e:.2e}")
#print(f"jet power in B: {blob.P_jet_B:.2e}")

# Disk and DT definition
L_disk = 10 ** 4.3230e+01* u.Unit("erg s-1")
#print("L_disk", L_disk)
M_BH   = 10 ** 4.2202e+01 * u.Unit("g")
m_dot  = 2.2698e+23 * u.Unit("g s-1")
eta    = (L_disk / (m_dot * c ** 2)).to_value("")
R_g     = ((G * M_BH) / c ** 2).to("cm")
R_in    = 3   #* R_g
R_out   = 100 #* R_g #too much with 1000?  
mu_s   = 9.7227e-01
disk   = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
print("disk ::", disk)
T_dt   = 2.0000e+03 * u.K
xi_dt  = 5.0000e-01
R_dt   = 7.2250e+14 * u.cm
dt     = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
#print("dt::", dt)
#print(disk)
#print(dt)


#BLR 
xi_line      = 6.0000e-01
epsilon_line = 9.5367e+09
R_line       = 1.0000e+14  
#BLR          = SphericalShellBLR()

# define the radiative processes
synch       = Synchrotron(blob, ssa=True)
ssc         = SynchrotronSelfCompton(blob, synch)
ec_dt       = ExternalCompton(blob, dt, r)
ec_SSD      = ExternalCompton(blob, disk,r/100) 
print("ec_SSD ::", ec_SSD )
#ec_BLR      = ExternalCompton(blob,)
# SEDs
nu          = np.logspace(9, 27, 50) * u.Hz
synch_sed   = synch.sed_flux(nu)
ssc_sed     = ssc.sed_flux(nu)
ec_dt_sed   = ec_dt.sed_flux(nu)
disk_bb_sed = disk.sed_flux(nu, z)
#print("disk_bb_sed::",disk_bb_sed)
disk_ec_SSD = ec_SSD.sed_flux(nu)
dt_bb_sed   = dt.sed_flux(nu,z)
total_sed   = synch_sed + ssc_sed  + dt_bb_sed  + ec_dt_sed +disk_bb_sed + disk_ec_SSD



# plot everything
load_mpl_rc()
#plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()

ax.loglog(
    nu / (1 + z), total_sed, ls="-", lw=2.1, color="crimson", label="agnpy, total"
)
ax.loglog(
    nu / (1 + z),
    synch_sed,
    ls="--",
    lw=1.3,
    color="orange",
    label="Synchrotron",
)
ax.loglog(
    nu / (1 + z), ssc_sed, ls="--", lw=1.3, color="dodgerblue", label="agnpy, SSC"
)

ax.loglog(
    nu / (1 + z),
    ec_dt_sed,
    ls="--",
    lw=1.3,
    color="b",
    label="EC on DT",
)

ax.loglog(
    nu / (1 + z),
    disk_bb_sed,
    ls="-.",
    lw=1.3,
    color="dimgray",
    label="Disk blackbody",
)

ax.loglog(
    nu / (1 + z),
    disk_ec_SSD,
    ls="-.",
    lw=1.3,
    color="gray",
    label="EC on SSD",
)


ax.loglog(
    nu / (1 + z),
    dt_bb_sed,
    ls=":",
    lw=1.3,
    color="dimgray",
    label="DT blackbody",
)
# systematics error in gray
ax.errorbar(
    x.to("Hz", equivalencies=u.spectral()).value,
    y,
    yerr=y_err_syst,
    marker=",",
    ls="",
    color="gray",
    label="",
)
# statistics error in black
ax.errorbar(
    x.to("Hz", equivalencies=u.spectral()).value,
    y,
    yerr=y_err_stat,
    marker=".",
    ls="",
    color="k",
    label="Data points",
)

ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_xlim([1e3, 1e29])
ax.set_ylim([10 ** (-20), 10 ** (-5)])
ax.legend(
    loc="upper center", fontsize=9, ncol=2,
)
plt.savefig("Multimessenger-modelling-of-FR0-radio-galaxies/ex_c/Fit9.1.png")
plt.show()