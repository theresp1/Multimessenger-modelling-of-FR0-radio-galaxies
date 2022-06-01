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


#Global variables 
N  = 10**2
E0 = 25 * u.GeV
z  = 0.03690



#Paliya data:   #TODO: put this into an imput file
# x1 = 2.0404  Gev    y1 = 1.9311*10**(-13)
# x2 = 8.5002  Gev    y2 = 1.5858*10**(-12)
# x3 = 35.464  Gev    y3 = 1.7022*10**(-12)
# xu = 146.39  Gev    yu = 3.4801*10**(-13)  #Upper limit 
#x1_error = [0.9962, 4.1452]        y1_error = [1.2601*10**(-13), 2.6*10**(-13)]
#x2_error = [4.2008, 17.246]        y2_error = [8.9629*10**(-14), 2.2758*10**(-13) ]
#x3_error = [17.477, 72.639]        y3_error = [4.3775*10**(-14), 2.9673*10**(-13) ] 
#xu_error = [72.626, 295.56]        yu_error = [1.8544*10**(-13), 0 ] 

x1_p = ( 2.0402* u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x2_p = ( 8.5100* u.GeV).to(u.Hz, equivalencies=u.spectral()).value
x3_p = ( 35.454* u.GeV).to(u.Hz, equivalencies=u.spectral()).value
xu_p = (146.51 * u.GeV).to(u.Hz, equivalencies=u.spectral()).value

x_p = np.array([x1_p,x2_p,x3_p,xu_p])

y1_p = 1.9311*10**(-13)
y2_p = 1.5858*10**(-13)
y3_p = 1.7022*10**(-13)
yu_p = 3.4801*10**(-13)

y_p = np.array([y1_p,y2_p,y3_p,yu_p])


#####x1_p = (2.0412 * u.GeV).to(u.eV, equivalencies=u.spectral()).value

x1_lower_error = (0.9962 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
x2_lower_error = (4.2008 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
x3_lower_error = (17.477 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
xu_lower_error = (72.626 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
x1_upper_error = (4.1452 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
x2_upper_error = (17.246 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
x3_upper_error = (72.639 * u.GeV).to(u.eV, equivalencies=u.spectral()).value
xu_upper_error = (295.56 * u.GeV).to(u.eV, equivalencies=u.spectral()).value

#errorbars for the Paliya data
x_error_min_p = np.array([x1_p - x1_lower_error,  x2_p - x2_lower_error , x3_p -x3_lower_error, xu_p -xu_lower_error])
x_error_max_p = np.array([x1_upper_error - x1_p , x2_upper_error - x2_p  , x3_upper_error - x3_p, xu_upper_error - xu_p])
y_error_min_p = np.array([y1_p - 1.2601*10**(-13), y2_p -8.9629*10**(-14),y3_p - 4.3775*10**(-14), yu_p - 1.8544*10**(-13)] )
y_error_max_p = np.array([2.6*10**(-13) - y1_p, 2.2758*10**(-13) - y2_p,2.9673*10**(-13)-y3_p,0] )

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


## Read the data 
with open('input/LEDA57137.txt') as f:
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
            else: 
                rest.append(line)
        else:
            data.append(line.strip())

## Convert the arrays into numpy arrays and convert the values types into floats
frequency_array       = np.asarray(frequency,dtype = float) #+ x_p 
frequency_array       = np.append(frequency_array, x_p)

Nufnu_array           = np.asarray(Nufnu,dtype = float) #+ y_p
Nufnu_array           = np.append(Nufnu_array,y_p)

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


#print(np.zeros(len(x_values)))

# TODO: Have to fix the input here, you need to fix e2dnde_ul, in addtition, you need to add e_min, e_max and is_up
table = Table()
table["e_ref"] = x_values * u.TeV
table["e2dnde"] = Nufnu_array* u.erg/( u.s * u.cm *u.cm)
table["e2dnde_err"] =  Nufnu_error_array_min * u.erg/( u.s * u.cm *u.cm)


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
y_err_stat =   flux_points.table["e2dnde_err"]
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
d_L = Distance(z=z).to("cm")
# - AGN parameters
agnpy_ssc.z.quantity = z
agnpy_ssc.z.frozen = True
agnpy_ssc.d_L.quantity = d_L
agnpy_ssc.d_L.frozen = True
# - blob parameters
agnpy_ssc.delta_D.quantity = 2.6
agnpy_ssc.delta_D.frozen = True
agnpy_ssc.log10_B.quantity =  np.log10(0.2)  
#agnpy_ssc.log10_B.quantity = np.log10( 1/(6.6466) * (1+z) * (v_s**2)/(2.8*10**6* v_c)  )
agnpy_ssc.log10_B.frozen = True
#agnpy_ssc.t_var. quantity = 1 * u.d
#agnpy_ssc.t_var.quantity = (Radius * (1+z))/(c.cgs *2) 
agnpy_ssc.t_var.quantity = 9.5710e+06 
#agnpy_ssc.t_var.quantity = 4.8970e+06
agnpy_ssc.t_var.frozen = False
#agnpy_ssc.gamma.quantity = 2
#agnpy_ssc.gamma.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -2.3682e+00  
agnpy_ssc.log10_k_e.frozen = False
agnpy_ssc.p1.quantity = 1.9
agnpy_ssc.p2.quantity = 3.7
agnpy_ssc.p1.frozen = True
agnpy_ssc.p2.frozen = True
agnpy_ssc.log10_gamma_b.quantity = np.log10(4000)
agnpy_ssc.log10_gamma_b.frozen = True
agnpy_ssc.log10_gamma_min.quantity = np.log10(1)
agnpy_ssc.log10_gamma_min.frozen = True
agnpy_ssc.log10_gamma_max.quantity = 4.4912e+00
agnpy_ssc.log10_gamma_max.frozen = True


# define model
model = SkyModel(name="LEDA57137", spectral_model=agnpy_ssc)
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
#plt.savefig("FU2_FIT7/Fit8.png")
plt.show()

#agnpy_ssc.covariance.plot_correlation()
#plt.savefig("FU2_FIT1/Base_correlation.png")


# Plotting Paliya data
#plt.scatter(x1_p,y1_p, color = "darkmagenta",s =5, label ="Paliya data")
#plt.scatter(x2_p,y2_p, color = "darkmagenta",s =5)
#plt.scatter(x3_p,y3_p, color = "darkmagenta",s =5)
#plt.scatter(xu_p,yu_p, color = "darkmagenta",s =5) 
#plt.errorbar(x_p,y_p, yerr = yerror_p, xerr = xerror_p, c ="darkmagenta" )



## Dust Torus (DT)
## Quantities defining the DT
T_dt   = 5e3 * u.K
xi_dt  = 0.1
L_disk = 2e46 * u.erg /(u.s)
dt     = RingDustTorus(L_disk, xi_dt, T_dt)

## Blue line region (BLR) 
# quantities defining the BLR
xi_line    = 0.1
R_line     = 1e17 * u.cm                                             ##set by eye (for now)
L_disk_blr = 9e45 * u.erg/(u.s)                                      ##set by eye (for now)
R_blr      = 1e17 * u.cm * (L_disk/(10e45 * u.erg/(u.s)) )**0.5


# x-axis 

nu             = np.logspace(8, 25)  *u.Hz

## PLOTTING



k_e = 10 ** agnpy_ssc.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ssc.p1.value
p2 = agnpy_ssc.p2.value
gamma_b = 10 ** agnpy_ssc.log10_gamma_b.value
print("gamma_b :", gamma_b)
gamma_min = 10 ** agnpy_ssc.log10_gamma_min.value
gamma_max = 10 ** agnpy_ssc.log10_gamma_max.value
B = 10 ** agnpy_ssc.log10_B.value * u.G
delta_D = agnpy_ssc.delta_D.value
R_b =  (c * agnpy_ssc.t_var.value * u.s * agnpy_ssc.delta_D.value / (1 + agnpy_ssc.z.value)).to("cm")
print("R_b", R_b)


sed_synch = Synchrotron.evaluate_sed_flux(
    nu,
    agnpy_ssc.z,
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
        #sed = sed_synch + sed_ssc
        ##return (sed / energy ** 2).to("1 / (cm2 eV s)")


plot_sed(nu/(1+z), sed_synch, color="maroon", label=r"synch")
plot_sed(nu/(1+z), sed_ssc, color="maroon", label=r"ssc")
plt.show()

