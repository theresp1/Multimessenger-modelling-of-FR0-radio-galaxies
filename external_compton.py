
import os
os.system("export PYTHONIOENCODING=utf8")

# import numpy, astropy and matplotlib for basic functionalities
from math import gamma
import pkg_resources
from   pathlib import Path
import numpy as np
import astropy.units as u
from   astropy.constants import c
from   astropy.table import Table
from   astropy.coordinates import Distance
import matplotlib.pyplot as plt
from   astropy.constants import k_B, m_e, c, G, M_sun, sigma_sb
import astropy.constants as const

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
from agnpy.utils.plot import plot_sed  #,  load_mpl_r

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


# default arrays to be used for integration
gamma_to_integrate = np.logspace(1, 9, 200)
nu_to_integrate = np.logspace(5, 30, 200) * u.Hz  # used for SSC
mu_to_integrate = np.linspace(-1, 1, 100)
phi_to_integrate = np.linspace(0, 2 * np.pi, 50)

#print("M-sun:", M_sun.cgs)

class AgnpyEC(SpectralModel):
    """Wrapper of agnpy's synchrotron and SSC classes.
    The flux model accounts for the Disk and DT's thermal SEDs.
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters
    whose range is expected to cover several orders of magnitudes (normalisation,
    gammas, size and magnetic field of the blob).
    """

    tag = "EC"
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=10)
    p1 = Parameter("p1", 2.1, min=0.5, max=3.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=10.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=5)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=6)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=0.1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=15)
    log10_B = Parameter("log10_B", 0.0, min=-5, max=2.0)
    #t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)
    mu_s = Parameter("mu_s", 0.9, min=0.0, max=1.0)
    log10_r = Parameter("log10_r", 17.0, min=16.0, max=20.0)
    # disk parameters
    log10_L_disk = Parameter("log10_L_disk", 45.0, min=39.0, max=48.0)
    log10_M_BH = Parameter("log10_M_BH", 42, min=np.log10(0.8e7 * M_sun.cgs.value), max=np.log10(1.2e11 * M_sun.cgs.value))
    log10_r_ssd = Parameter("log10_r_ssd",17, min= 10.0,max=20.0)
    ## check if this is cgs or si ((it is si, but do we want it in grams? ))
    #log10_M_BH = Parameter("log10_M_BH", 42, min=np.log10(0.8e9 * M_sun), max=np.log10(1.2e9 * M_sun))

    m_dot = Parameter("m_dot", "1e26 g s-1", min=1e24, max=1e30)
    R_in = Parameter("R_in", "1e14 cm", min=1e12, max=1e16)
    R_out = Parameter("R_out", "1e17 cm", min=1e12, max=1e19)
    # DT parameters
    # xi_dt = Parameter("xi_dt", 0.6, min=0.0, max=1.0)
    # T_dt = Parameter("T_dt", "1e3 K", min=1e2, max=1e9)
    # R_dt = Parameter("R_dt", "2.5e18 cm", min=1.0e17, max=1.0e19)
    #BLR parameters 
    # xi_line = Parameter("xi_line",0.6, min =0.0, max=1.0)
    # epsilon_line = Parameter("epsilon_line",1e6,min= 1e-6, max = 1e16 )
    # R_line = Parameter("R_line","1e14 cm", min = 1e4,max = 1e20)


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
        #t_var,
        mu_s,
        log10_r,
        log10_r_ssd,
        log10_L_disk,
        log10_M_BH,
        m_dot,
        R_in,
        R_out,
    #    xi_dt,
    #    T_dt,
        # R_dt,
        # xi_line,
        # epsilon_line,
        # R_line,
    ):
        # conversions
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b =  5*10**16 * u.cm
        r = 10 ** log10_r * u.cm
        r_ssd = 10**log10_r_ssd * u.cm
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        M_BH = 10 ** log10_M_BH * u.Unit("g")
        #eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        nu = energy.to("Hz", equivalencies=u.spectral())
        # non-thermal components
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
            ssa = True,
            gamma=gamma_to_integrate,
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
            ssa =  False,   # dont need this for the SSC
            gamma=gamma_to_integrate,
        )
        # sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
        #     nu,
        #     z,
        #     d_L,
        #     delta_D,
        #     mu_s,
        #     R_b,
        #     L_disk,
        #     xi_dt,
        #     eps_dt,
        #     R_dt,
        #     r,
        #     BrokenPowerLaw,
        #     k_e,
        #     p1,
        #     p2,
        #     gamma_b,
        #     gamma_min,
        #     gamma_max,
        #     gamma=gamma_to_integrate,
        #)
        sed_ec_SSDisk = ExternalCompton.evaluate_sed_flux_ss_disk(
            nu,
            z,
            d_L,
            delta_D,
            mu_s,
            R_b,
            M_BH,
            L_disk,
            eta,
            R_in,
            R_out,
            r_ssd,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
            mu_size=100,
            phi=phi_to_integrate,
        )
        # sed_ec_blr = ExternalCompton.evaluate_sed_flux_blr(
            # nu,
            # z,
            # d_L,
            # delta_D,
            # mu_s,
            # R_b,
            # L_disk,
            # xi_line,
            # epsilon_line,
            # R_line,
            # r,
            # BrokenPowerLaw,
            # k_e,
            # p1,
            # p2,
            # gamma_b,
            # gamma_min,
            # gamma_max,
            # integrator=np.trapz,
            # gamma=gamma_to_integrate,
            # mu=mu_to_integrate,
            # phi=phi_to_integrate,

        #)
        # thermal components
        sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        # sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
        #     nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
        # )
        sed = sed_synch + sed_ssc + sed_bb_disk + sed_ec_SSDisk #+ sed_ec_dt  #+ sed_ec_blr + sed_bb_dt 
        return (sed / energy ** 2).to("1 / (cm2 eV s)")

# IMPORTANT: add the new custom model to the registry of spectral models recognised by gammapy
SPECTRAL_MODEL_REGISTRY.append(AgnpyEC)

df_1                = pd.read_csv('input/new_Tol_values3.csv')
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

##vavlues set by eye
v_c     = 5.38 *10**21
v_s     = 7.93 *10**13

#Values set by Tavecchio
#Radius = 5 *10**16 * u.cm

# declare a model 
agnpy_ec = AgnpyEC()
# global parameters of the blob,the DT and the BLR 
z       = 0.0284
d_L     = Distance(z=z).to("cm")
# blob
Gamma   =  1.05
delta_D =  1.30
Beta    = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
mu_s    =  (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
B       = 1.5 * u.G
# disk
L_disk  = 1.2e42 * u.Unit("erg s-1")  # disk luminosity
M_BH    = 0.5e8 * M_sun   #1e8-1e9 range leave it free 
eta     = 1 / 12
m_dot   = (L_disk / (eta * c ** 2)).to("g s-1") #about 40Msolar
R_g     = ((G * M_BH) / c ** 2).to("cm")
R_in    = 3 * R_g
R_out   = 400 * R_g #too much with 1000?  
# DT
# xi_dt   = 0.5  # fraction of disk luminosity reprocessed by the DT
# T_dt    = 0.5e3 * u.K # 500-2000K range   or 10**4 -10**9 range? 
# R_dt    = 2.5 * 10**18 * (L_disk/(10**45 * u.Unit("erg s-1")))**2  *u.cm  #6.47 * 1e18 * u.cm
#print("R_dt:", R_dt)
# BLR
# xi_line      = 0.5                 #fraction of the disk radiation reprocessed by the BLR
# epsilon_line = 1e6                 #dimensionless energy of the emitted line
# R_line       = 1e14                #radius of the BLR spherical shell



# size and location of the emission region
#t_var   = 7.9062e+05 * u.s
r       = 5e17 * u.cm
r_ssd   = 2e15 * u.cm
# instance of the model wrapping angpy functionalities

# - AGN parameters
# -- distances
agnpy_ec.z.quantity = z
agnpy_ec.z.frozen = True
agnpy_ec.d_L.quantity = d_L.cgs.value
agnpy_ec.d_L.frozen = True
# -- SS disk
agnpy_ec.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
agnpy_ec.log10_L_disk.frozen   = True
agnpy_ec.log10_M_BH.quantity = np.log10(M_BH.to_value("g"))
agnpy_ec.log10_M_BH.frozen   = True
agnpy_ec.m_dot.quantity = m_dot
agnpy_ec.m_dot.frozen    = True
agnpy_ec.R_in.quantity = R_in
agnpy_ec.R_in.frozen   = True
agnpy_ec.R_out.quantity = R_out
agnpy_ec.R_out.frozen   = True
# -- Dust Torus
# agnpy_ec.xi_dt.quantity = xi_dt
# agnpy_ec.xi_dt.frozen   = True
# agnpy_ec.T_dt.quantity = T_dt
# agnpy_ec.T_dt.frozen = True
# agnpy_ec.R_dt.quantity = R_dt
# agnpy_ec.R_dt.frozen = True
# - blob parameters
agnpy_ec.delta_D.quantity  = delta_D
agnpy_ec.delta_D.frozen    = True
agnpy_ec.log10_B.quantity = np.log10(B.to_value("G"))
agnpy_ec.log10_B.frozen   = True
agnpy_ec.mu_s.quantity = mu_s
agnpy_ec.mu_s.frozen   = True
#agnpy_ec.t_var.quantity = t_var
#agnpy_ec.t_var.frozen   = False
agnpy_ec.log10_r.quantity = np.log10(r.to_value("cm"))
agnpy_ec.log10_r.frozen   = True
agnpy_ec.log10_r_ssd.quantity = np.log10(r_ssd.to_value("cm"))
agnpy_ec.log10_r_ssd.frozen   = True
# - EED
agnpy_ec.log10_k_e.quantity = -2.1359e+00 
agnpy_ec.log10_k_e.frozen = False
agnpy_ec.p1.quantity =  2.0
agnpy_ec.p2.quantity =  4.1
agnpy_ec.p1.frozen = True
agnpy_ec.p2.frozen = True
agnpy_ec.log10_gamma_b.quantity = np.log10(2500)     #3.3962e+00  #np.log10(2490)
agnpy_ec.log10_gamma_b.frozen = True
agnpy_ec.log10_gamma_min.quantity = np.log10(100)   #2.0792e+00 
agnpy_ec.log10_gamma_min.frozen = True
agnpy_ec.log10_gamma_max.quantity =  np.log10(3.0*10**4)    #5.3802e+00    #np.log10(6.4e5)
agnpy_ec.log10_gamma_max.frozen = True

# nu          = np.logspace(9, 27, 50) * u.Hz
# sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L)

# print("sed_bb_disk Temperature:", sed_bb_disk)



# define model
model = SkyModel(name="Tol_1326-379", spectral_model=agnpy_ec)
dataset_ec = FluxPointsDataset(model, flux_points)
#print("dataset", dataset_ssc)
# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
dataset_ec.mask_fit = dataset_ec.data.energy_ref > E_min_fit


# define the fitter
fitter = Fit([dataset_ec])
results = fitter.run(optimize_opts={"print_level": 1})
print(results)
print(agnpy_ec.parameters.to_table())
# plot best-fit model
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ec.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
#plt.savefig("ex_c/Fit40.png")
#plt.show()

#agnpy_ssc.covariance.plot_correlation()
#plt.savefig("output_scan_B/Fit13_correlation.png")
#plt.show()



## PLotting each specific component 

# define the emission region and the thermal emitters
k_e = 10 ** agnpy_ec.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ec.p1.value
p2 = agnpy_ec.p2.value
gamma_b = 10 ** agnpy_ec.log10_gamma_b.value
gamma_min = 10 ** agnpy_ec.log10_gamma_min.value
gamma_max = 10 ** agnpy_ec.log10_gamma_max.value
B = 10 ** agnpy_ec.log10_B.value * u.G
r = 10 ** agnpy_ec.log10_r.value * u.cm
r_ssd = 10** agnpy_ec.log10_r_ssd.value * u.cm    ##distance between the disk and the blob
delta_D = agnpy_ec.delta_D.value
R_b =  5.0 *10**16 * u.cm
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
print(blob)
#print(f"jet power in particles: {blob.P_jet_e:.2e}")
#print(f"jet power in B: {blob.P_jet_B:.2e}")

# Disk and DT definition
L_disk = 10 ** agnpy_ec.log10_L_disk.value * u.Unit("erg s-1")
M_BH   = 10 ** agnpy_ec.log10_M_BH.value * u.Unit("g")
m_dot  = agnpy_ec.m_dot.value * u.Unit("g s-1")
eta    = (L_disk / (m_dot * c ** 2)).to_value("")
R_in   = agnpy_ec.R_in.value * u.cm
R_out  = agnpy_ec.R_out.value * u.cm
disk   = SSDisk(M_BH, L_disk, eta, R_in, R_out)
#dt     = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
#print(disk)
#print(dt)

# define the radiative processes
synch       = Synchrotron(blob, ssa=True)
ssc         = SynchrotronSelfCompton(blob, synch)
#ec_dt       = ExternalCompton(blob, dt, r)
ec_SSD      = ExternalCompton(blob,disk,r_ssd)
# SEDs
nu          = np.logspace(9, 29, 27) * u.Hz
synch_sed   = synch.sed_flux(nu)
ssc_sed     = ssc.sed_flux(nu)
#ec_dt_sed   = ec_dt.sed_flux(nu)
disk_bb_sed = disk.sed_flux(nu, z)
disk_ec_SSD = ec_SSD.sed_flux(nu)
#dt_bb_sed   = dt.sed_flux(nu,z)
total_sed   = synch_sed + ssc_sed  +disk_bb_sed + disk_ec_SSD #  + dt_bb_sed  + ec_dt_sed



# plot everything
#load_mpl_rc()
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

# ax.loglog(
#     nu / (1 + z),
#     ec_dt_sed,
#     ls="--",
#     lw=1.3,
#     color="b",
#     label="EC on DT",
# )

ax.loglog(
    nu / (1 + z),
    disk_bb_sed,
    ls="-.",
    lw=1.3,
    color="dimgray",
    label="Disk blackbody",
)
###
ax.loglog(
    nu / (1 + z),
    disk_ec_SSD,
    ls="-.",
    lw=1.3,
    color="seagreen",
    label="EC on SSD",
)
###

# ax.loglog(
#     nu / (1 + z),
#     dt_bb_sed,
#     ls=":",
#     lw=1.3,
#     color="dimgray",
#     label="DT blackbody",
# )
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
#plt.savefig("ex_c/Fit40.1.png")
#plt.show()

M_BH   = 10 ** agnpy_ec.log10_M_BH.value * u.Unit("g")
R_in   = agnpy_ec.R_in.value * u.cm
R_out  = agnpy_ec.R_out.value * u.cm

print('M_BH = ', M_BH)
print('R_in =', R_in)
print('R_out =', R_out)


M_BH2 =  5e8* M_sun
R_in2 =  3 * (G * M_BH2 / c ** 2)
R_out2 = 400 * (G * M_BH2 / c ** 2)

def evaluate_T(R, M_BH, m_dot, R_in):
        """black body temperature (K) at the radial coordinate R"""
        phi = 1 - np.sqrt((R_in / R).to_value(""))
        val = (3 * G * M_BH * m_dot * phi) / (8 * np.pi * np.power(R, 3) * sigma_sb)
        return np.power(val, 1 / 4).to("K")



R_values = np.logspace(10,40,100)
R_arr    = np.logspace(np.log10(R_in.value +100),np.log10(R_out.value),100) * u.cm
R_arr2   = np.logspace(np.log10(R_in2.value +100),np.log10(R_out2.value),100) * u.cm

#plt.loglog(R_values, evaluate_T(R_arr, M_BH, m_dot, R_in), label =r'M_BH = 0.5e8 \cdot M_sun ')
plt.loglog(R_values, evaluate_T(R_arr2, M_BH2, m_dot, R_in2 ), label =r'M_BH = 5e8 \cdot M_sun ')
#plt.loglog(R_values, evaluate_T(R_arr, 5e9* M_sun, m_dot, R_in), label =r'M_BH = 5e9 \cdot M_sun ')

plt.xlabel('R')
plt.ylabel('T')
plt.legend()
plt.show()
'''''
for i in range(3, 401): 
    n = i *((G * M_BH) / c ** 2)
    Temp[i]= SSDisk.T(disk1,n)

print("Temp: ",Temp)
#print("ec_disk_1 :",ec_disk_1)

nu = np.logspace(9, 30, 50) * u.Hz

ec_disk_sed_1 = ec_disk_1.sed_flux(nu)

plot_sed(nu/(1+z), ec_disk_sed_1, color="maroon", label=r"")
plt.show()
'''''