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
import numpy as np 
import matplotlib.pyplot as plt 
from   astropy import units as u
from   astropy import constants as const
from   ast import If
from   agnpy.targets import RingDustTorus
from   agnpy.utils.plot import plot_sed
from   agnpy.targets import SphericalShellBLR
from   astropy.constants import k_B, m_e, c, G, M_sun

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
from   astropy.constants import k_B, m_e, c, G, M_sun


# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus



class AgnpySSC(SpectralModel):
    """Wrapper of agnpy's synchrotron and SSC classes.
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters
    whose range is expected to cover several orders of magnitudes (normalisation,
    gammas, size and magnetic field of the blob).
    """

    tag = "SSC"
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=10)
    p1 = Parameter("p1", 2.1, min=1.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=7.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=5)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=7)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=0.1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=10000)
    log10_B = Parameter("log10_B", -2, min=-3, max=1.0)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)
    ######
    #gamma = Parameter("Gamma", 2 ,min=1, max = 10 )
    ##SSd
    #m_dot = Parameter("m_dot", "1e26 g s-1", min=1e18, max=1e30)
    #R_in = Parameter("R_in", "1e14 cm", min=1e12, max=1e16)
    #R_out = Parameter("R_out", "1e17 cm", min=1e12, max=1e19)
    log10_L_disk = Parameter("log10_L_disk", 45.0, min=39.0, max=48.0)
    #log10_M_BH = Parameter("log10_M_BH", 42, min=np.log10(0.8e7 * M_sun.cgs.value), max=np.log10(1.2e11 * M_sun.cgs.value))
    # ## DT
    xi_dt = Parameter("xi_dt", 0.5, min = 0.0, max = 1.0 )
    T_dt  = Parameter("T_dt", "1e3 K", min = 1e2, max = 1e6 )
    R_dt  = Parameter("R_dt", "2.5e18 cm", min = 1.0e15, max = 1.0e20)

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
        #R_in,
        #R_out,
        log10_L_disk,
        #log10_M_BH,
        #m_dot,
        #gamma,
        xi_dt,
        T_dt,
        R_dt
    ):
        # conversions
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        #M_BH = 10 ** log10_M_BH * u.Unit("g")

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
       #thermal components

        sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
            nu,z,xi_dt * L_disk,T_dt,R_dt, d_L

        )

        sed = sed_synch + sed_ssc + sed_bb_dt
        return (sed / energy ** 2).to("1 / (cm2 eV s)")
        





#Global variables 
N  = 10**2
E0 = 25 * u.GeV
z  = 0.03254

#Paliya data:   ##TODO: Put this into an imput file
# x1 = 2.038   Gev    y1 = 1.7220*10**(-13)  
# x2 = 8.5076  Gev    y2 = 1.2658*10**(-13)
# x3 = 35.433  Gev    y3 = 1.4145*10**(-13)
# xu = 146.51  Gev    yu = 3.3463*10**(-13)  #Upper limit
#x1_error = [0.9970,4.1465]         y1_error = [1.1233*10**(-13),2.2345*10**(-13)]
#x2_error = [4.1966, 17.271]        y2_error = [6.4178*10**(-14),1.8868*10**(-13)]
#x3_error = [17.4540,72.592]        y3_error = [2.7652*10**(-14),2.5383*10**(-13)]
#xu_error = [72.627,295.350]        yu_error = [1.7931*10**(-13),0]

x1_p = ( 2.038 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x2_p = ( 8.5076* u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x3_p = ( 35.433* u.GeV).to(u.Hz, equivalencies=u.spectral()).value
xu_p = (146.51 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value

x_p = np.array([x1_p,x2_p,x3_p,xu_p])

y1_p = 1.7220*10**(-13)
y2_p = 1.2658*10**(-13)
y3_p = 1.4145*10**(-13)
yu_p = 3.3463*10**(-13)

y_p = np.array([y1_p,y2_p,y3_p,yu_p])

x1_lower_error = (0.997  * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x2_lower_error = (4.1966 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x3_lower_error = (17.454 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
xu_lower_error = (72.627 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x1_upper_error = (4.1465 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x2_upper_error = (17.271 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x3_upper_error = (72.592 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value
xu_upper_error = (295.35 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value

#errorbars for the Paliya dataR_b
x_error_min_p = np.array([ x1_p - x1_lower_error,  x2_p - x2_lower_error , x3_p -x3_lower_error, xu_p -xu_lower_error])
x_error_max_p = np.array([ x1_upper_error - x1_p , x2_upper_error - x2_p  , x3_upper_error - x3_p, xu_upper_error - xu_p])
y_error_min_p = np.array([ y1_p - 1.1233*10**(-13), y2_p - 6.4178*10**(-14),y3_p - 2.7652*10**(-14), yu_p - 1.7931*10**(-13)] )
y_error_max_p = np.array([2.2345*10**(-13)- y1_p, 1.8868*10**(-13) - y2_p, 2.5383*10**(-13)-y3_p,0] )

xerror_p = [x_error_min_p,x_error_max_p]
yerror_p = [y_error_min_p, y_error_max_p]



def _is_interesting_line(line_str: str) -> bool:
    return line and line_str[0].isspace()

##  Make empty lists of everything we want from the file (plus a rest array to put everything we dont need)
data        = []
frequency   = []
Nufnu       = []
Nufnu_error = []
rest        = []
e_min       = []
e_max       = []
isup_lim    = []
up_array    = []

isup_lim_p = np.array([False,False,False,True])
up_lim_array_p = np.array([0,0,0,3.3463*10**(-13)])

## Read the data 
with open('input/LEDA55267.txt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        interesting = _is_interesting_line(line)
        if not interesting:
            if "#" not in line:
                #Have to check manually here where our desired values are(TODO: rewrite this so we don't need to check this manually)
                frequency.append(line.strip().split()[0])
                Nufnu.append(line.strip().split()[2])
                Nufnu_error.append(line.strip().split()[3])
                e_min.append(0)
                e_max.append(0)
                isup_lim.append(False)
                up_array.append(0)
            else: 
                rest.append(line)
        else:
            data.append(line.strip())

## Convert the arrays into numpy arrays and convert the values types into floats
frequency_array       = np.asarray(frequency,dtype = float) #+ x_p 
frequency_array       = np.append(frequency_array, x_p)

Nufnu_array           = np.asarray(Nufnu,dtype = float) #+ y_p
Nufnu_array           = np.append(Nufnu_array,y_p)

e_min_values          = np.asarray(e_min,dtype = float)
e_min_values          = np.append(e_min_values,x_error_min_p)

e_max_values          = np.asarray(e_max,dtype = float)
e_max_values          = np.append(e_max_values,x_error_max_p)

isup_lim_array        = np.asarray(isup_lim,dtype = bool)
isup_lim_array        = np.append(isup_lim_array, isup_lim_p)

uplim_array           = np.asarray(up_array,dtype=float)
uplim_array           = np.append(uplim_array,up_lim_array_p)


Nufnu_error_array     = np.asarray(Nufnu_error,dtype = float) #+yerror_p
Nufnu_error_array_min     = np.append(Nufnu_error_array,y_error_min_p)    
Nufnu_error_array_max     = np.append(Nufnu_error_array,y_error_max_p)

#print("f: ",frequency_array)
#print("nufnu: ", Nufnu_array)
#print("nufnu_error: ", Nufnu_error_array)
#errorbars_b         = np.array(error_bars)   #* (u.erg/( u.s * u.cm *u.cm ))
#upper_limits_b      = np.array(upper_limits) #* (u.erg/( u.s * u.cm *u.cm ))

# convert x_v_b from Hz to TeV
def convert_to_Tev(Xarray): 
    ReturnArray = np.zeros(np.size(Xarray))
    for i in range(np.size(Xarray)):
        ReturnArray[i] = (Xarray[i] * u.Hz).to(u.TeV,equivalencies=u.spectral()).value

    return ReturnArray

x_values     = convert_to_Tev(frequency_array)
x_err_min    = convert_to_Tev(e_min_values)
x_err_max    = convert_to_Tev(e_max_values)

#print(np.zeros(len(x_values)))

# TODO: Have to fix the input here, you need to fix e2dnde_ul, in addtition, you need to add e_min, e_max and is_up
table = Table()
table["e_ref"] = x_values * u.TeV
table["e2dnde"] = Nufnu_array* u.erg/( u.s * u.cm *u.cm)
#table["e2dnde_err"] =  Nufnu_error_array_min * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_err"]   = np.zeros(len(x_values)) *  u.erg/( u.s * u.cm *u.cm)
table["e2dnde_errn"]  = Nufnu_error_array_min * u.erg/( u.s * u.cm *u.cm)
table["e2dnde_errp"]  = Nufnu_error_array_max * u.erg/( u.s * u.cm *u.cm)
table["e_min"]        = x_err_min* u.TeV
table["e_max"]        = x_err_max * u.TeV
table["is_ul"]        = isup_lim_array         
table["e2dnde_ul"]    = uplim_array   * u.erg/( u.s * u.cm *u.cm)

#table["e2dnde_err"] =  np.zeros(len(x_values)) *  u.erg/( u.s * u.cm *u.cm)
#table["e2dnde_errn"] = Nufnu_error_array_min * u.erg/( u.s * u.cm *u.cm)
#table["e2dnde_errp"] = Nufnu_error_array_max * u.erg/( u.s * u.cm *u.cm)
#table["e2dnde_ul"]    = upper_limit_array * u.erg/( u.s * u.cm *u.cm)
table.meta["SED_TYPE"] = "e2dnde" 

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
flux_points.plot()
plt.show()
# array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 30% on VHE gamma-ray instruments
# - 10% on HE gamma-ray instruments
# - 10% on X-ray instruments
# - 5% on lower-energy instruments
x = flux_points.table["e_ref"]
y = flux_points.table["e2dnde"]
y_err_stat =   flux_points.table["e2dnde_errp"]
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
y_err_syst[he_gamma] = 0.10
#y_err_syst[x_ray] = 0.10
y_err_syst[optical] = 0.10
y_err_syst[micro_to_radio] = 0.05
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
v_c = 5.38 *10**21
v_s = 7.93 *10**13

#Values set by Tavecchio
Radius = 5 *10**16 * u.cm

# declare a model 
agnpy_ssc = AgnpySSC()
# initialise parameters



# global parameters of the blob,the DT and the BLR 
z       = 0.03254
d_L     = Distance(z=z).to("cm")
# blob
# Gamma   =  1.05
# delta_D =  1.3
# Beta    = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
# mu_s    =  (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
# B       = 0.09* u.G
# # disk
# L_disk  = 5.5e44 * u.Unit("erg s-1")  # disk luminosity
# M_BH    = 0.9e10 * M_sun   #1e8-1e9 range leave it free 
# eta     = 1 / 12
# m_dot   = m_dot   = (L_disk / (eta * c ** 2)).to("g s-1") #about 40Msolar 9e18    #9e18    #(L_disk / (eta * c ** 2)).to("g s-1") #about 40Msolar
# R_g     = ((G * M_BH) / c ** 2).to("cm")
# R_in    = 6 * R_g
# R_out   = 400 * R_g #too much with 1000?  
Gamma   =  1.05
delta_D =  1.3
B       = 0.09* u.G
L_disk  = 0.4e45 * u.Unit("erg s-1")  # disk luminosity
xi_dt = 0.5  # fraction of disk luminosity reprocessed by the DT
T_dt  = 3.7e3 * u.K
R_dt  = 6.47 * 1e18 * u.cm
agnpy_ssc.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
agnpy_ssc.log10_L_disk.frozen   = True
agnpy_ssc.xi_dt.quantity = xi_dt
agnpy_ssc.xi_dt.frozen = True
agnpy_ssc.T_dt.quantity = T_dt
agnpy_ssc.T_dt.frozen = True
agnpy_ssc.R_dt.quantity = R_dt
agnpy_ssc.R_dt.frozen = True


d_L = Distance(z=z).to("cm")
# - AGN parameters
agnpy_ssc.z.quantity = z
agnpy_ssc.z.frozen = True
agnpy_ssc.d_L.quantity = d_L
agnpy_ssc.d_L.frozen = True
# -- SS disk
agnpy_ssc.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
agnpy_ssc.log10_L_disk.frozen   = True
#agnpy_ssc.log10_M_BH.quantity = np.log10(M_BH.to_value("g"))
# agnpy_ssc.log10_M_BH.frozen   = True
# agnpy_ssc.m_dot.quantity = m_dot
# agnpy_ssc.m_dot.frozen    = True
# agnpy_ssc.R_in.quantity = R_in
# agnpy_ssc.R_in.frozen   = True
# agnpy_ssc.R_out.quantity = R_out
# agnpy_ssc.R_out.frozen   = True
# - blob parameters
agnpy_ssc.delta_D.quantity = delta_D
agnpy_ssc.delta_D.frozen = True
agnpy_ssc.log10_B.quantity = np.log10(B.to_value("G"))
#agnpy_ssc.log10_B.quantity = np.log10( 1/(6.6466) * (1+z) * (v_s**2)/(2.8*10**6* v_c)  )
agnpy_ssc.log10_B.frozen = True
#agnpy_ssc.t_var. quantity = 1 * u.d
#agnpy_ssc.t_var.quantity = (Radius * (1+z))/(c.cgs *2) 
agnpy_ssc.t_var.quantity = 4.1887e+06 
#agnpy_ssc.t_var.quantity = 4.8970e+06
agnpy_ssc.t_var.frozen = True
#agnpy_ssc.gamma.quantity = 2
#agnpy_ssc.gamma.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -4.3680e+00   
agnpy_ssc.log10_k_e.frozen = True
agnpy_ssc.p1.quantity = 1.9
agnpy_ssc.p2.quantity = 3.7
agnpy_ssc.p1.frozen = True
agnpy_ssc.p2.frozen = True
agnpy_ssc.log10_gamma_b.quantity = np.log10(6000)
agnpy_ssc.log10_gamma_b.frozen = True
agnpy_ssc.log10_gamma_min.quantity = np.log10(10)
agnpy_ssc.log10_gamma_min.frozen = True
agnpy_ssc.log10_gamma_max.quantity = 5.812e+00
agnpy_ssc.log10_gamma_max.frozen = True
# - DT 
# L_disk  = 5e44 * u.Unit("erg s-1")  # disk luminosity
# xi_dt = 0.5  # fraction of disk luminosity reprocessed by the DT
# T_dt  = 4.5e3 * u.K
# R_dt  = 6.47 * 1e18 * u.cm
# agnpy_ssc.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
# agnpy_ssc.log10_L_disk.frozen   = True
# agnpy_ssc.xi_dt.quantity = xi_dt
# agnpy_ssc.xi_dt.frozen = True
# agnpy_ssc.T_dt.quantity = T_dt
# agnpy_ssc.T_dt.frozen = True
# agnpy_ssc.R_dt.quantity = R_dt
# agnpy_ssc.R_dt.frozen = True


# define model
model = SkyModel(name="LEDA55267", spectral_model=agnpy_ssc)
dataset_ssc = FluxPointsDataset(model, flux_points)
print("dataset", dataset_ssc)
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
plt.ylim(1e-20,1e-8)
plt.savefig("LEDAoutput/LEDA55267_fit5.png")

plt.show()



# define the emission region and the thermal emitters
k_e = 10 ** agnpy_ssc.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ssc.p1.value
p2 = agnpy_ssc.p2.value
gamma_b = 10 ** agnpy_ssc.log10_gamma_b.value
gamma_min = 10 ** agnpy_ssc.log10_gamma_min.value
gamma_max = 10 ** agnpy_ssc.log10_gamma_max.value
B = 10 ** agnpy_ssc.log10_B.value * u.G
Gamma = 1.05
#r = 10 ** agnpy_ssc.log10_r.value * u.cm
delta_D = agnpy_ssc.delta_D.value
R_b = (
    c * agnpy_ssc.t_var.quantity * agnpy_ssc.delta_D.quantity / (1 + agnpy_ssc.z.quantity)
).to("cm")
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
L_disk = 10 ** (agnpy_ssc.log10_L_disk.value) * u.Unit("erg s-1")
#M_BH = 10 ** agnpy_ssc.log10_M_BH.value * u.Unit("g")
#m_dot = agnpy_ssc.m_dot.value * u.Unit("g s-1")
# eta = (L_disk / (m_dot * c ** 2)).to_value("")
xi_dt = agnpy_ssc.xi_dt.value
T_dt = agnpy_ssc.T_dt.value * u.K
R_dt = agnpy_ssc.R_dt.value * u.cm
#R_in = agnpy_ssc.R_in.value * u.cm
#R_out = agnpy_ssc.R_out.value * u.cm
#disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
#dt = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
#print(disk)
#print(dt)
# define the radiative processes
synch = Synchrotron(blob, ssa=True)
ssc = SynchrotronSelfCompton(blob, synch)
dt = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)

#ec_dt = ExternalCompton(blob, dt, r)
# SEDs
nu = np.logspace(9, 27, 100) * u.Hz
synch_sed = synch.sed_flux(nu)
ssc_sed = ssc.sed_flux(nu)
dt_bb_sed = dt.sed_flux(nu, z)
#ec_dt_sed = ec_dt.sed_flux(nu)
#disk_bb_sed = disk.sed_flux(nu, z)
#dt_bb_sed = dt.sed_flux(nu, z)
total_sed = synch_sed + ssc_sed + dt_bb_sed  #+ dt_bb_sed #+ ec_dt_sed + disk_bb_sed

# plot everything
load_mpl_rc()
#plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.loglog(
    nu / (1 + z), total_sed, ls="-", lw=2.1, color="crimson", label="total"
)
ax.loglog(
    nu / (1 + z),
    synch_sed,
    ls="--",
    lw=1.3,
    color="goldenrod",
    label="synchrotron",
)
ax.loglog(
    nu / (1 + z), ssc_sed, ls="--", lw=1.3, color="dodgerblue", label="SSC"
)
# ax.loglog(
#     nu / (1 + z),
#     ec_dt_sed,
#     ls="--",
#     lw=1.3,
#     color="lightseagreen",
#     label="agnpy, EC on DT",
# )
ax.loglog(
    nu / (1 + z),
    dt_bb_sed,
    ls="-.",
    lw=1.3,
    color="dimgray",
    label="blackbody",
)
# ax.loglog(
#     nu / (1 + z),
#     dt_bb_sed,
#     ls=":",
#     lw=1.3,
#     color="dimgray",
#     label="agnpy, DT blackbody",
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
    label="data",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_xlim([1e6, 1e29])
ax.set_ylim([10 ** (-25), 10 ** (-5)])
ax.legend(
    loc="upper center", fontsize=9, ncol=2,
)
plt.savefig("LEDAoutput/LEDA55267_fit5.1.png")
plt.show()



#L_edd = 1.26 *10**38 *(10**(agnpy_ssc.log10_M_BH.value))/((M_sun.cgs).value) * u.erg * 1/u.s
#print("L_edd", L_edd)
#M_edd = L_edd/(0.1 * (c.cgs)**2)
#mdot = agnpy_ssc.m_dot.value/ (M_edd) *u.g *1/u.s
#print("mdot :", mdot)