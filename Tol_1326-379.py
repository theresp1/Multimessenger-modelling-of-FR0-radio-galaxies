import numpy as np 
import matplotlib.pyplot as plt 
from   astropy import units as u
from   astropy import constants as const
import pandas as pd
from   astropy.constants import c
from   agnpy.spectra import ElectronDistribution
from   astropy.cosmology import FlatLambdaCDM
from   agnpy.emission_regions import Blob
from   agnpy.synchrotron import Synchrotron
from   agnpy.compton import SynchrotronSelfCompton
from   agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from   agnpy.utils.plot import load_mpl_rc
from   agnpy.spectra import BrokenPowerLaw
from   astropy.coordinates import Distance
from   astropy.table import Table
from   agnpy.emission_regions import Blob



###Information###
#We model the SED in two different scenarios, namely
#assuming (1) that the jet is closely aligned toward the observer (as
#for the case of blazars) and has a large Lorentz factor (blazar case)
#or (2) that the jet is misaligned and slow (radiogalaxy case).

# Global variables for Tol 1326-379
N                = 300 
z                = 0.0284    
velocity         = 8595*10**(3) * u.m/(u.s)  #8595 kms^-1   #velocity of the host galaxy 
viewing_angle_1  = 5.0*np.pi/180 #rad   np.cos(x) where x is in radiens
viewing_angle_2  = 30.0*np.pi/180
B_1              = 0.1*u.G
B_2              = 0.2*u.G 
gamma_min_1      = 400.0 
gamma_min_2      = 100.0 
gamma_break_1    = 5.0*10**3 
gamma_break_2    = 1.0*10**4 
gamma_max_1      = 3.0*10**4  
gamma_max_2      = 3*10**4 
Gamma_1          = 10.0
Gamma_2          = 2.0
R_1              = 3.5*10**(15)*u.cm
R_2              = 5.0*10**(16)*u.cm
P_e_1            = 1.2*10**(44)*u.erg*1/(u.s)
P_e_2            = 3.5*10**(43)*u.erg*1/(u.s)
P_j_1            = 2.7*10**(44)*u.erg*1/(u.s)
P_j_2            = 1.6*10**(44)*u.erg*1/(u.s)
p1               = 2.0 
p2               = 4.8 


#Luminocity distance 
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3, Tcmb0 = 2.725)
distanceL = cosmo.luminosity_distance(z)  #unit: Mpc
dis  = Distance(z=z).to("cm")


# Reading in the data with pandas
df_1 = pd.read_csv('values.csv')

# Data with relativistic boosting (as they were presented in Tavacchio)
x_v = df_1["energy"].copy()
y_v = df_1["flux"].copy()
error_bars = df_1["flux_error"].copy()

# Doppler factor
def relativistic_doppler_factor(Gamma,viewing_angle): 
    beta  = np.sqrt(1 - 1/(Gamma**2))
    doppler_factor = 1/(Gamma*(1-beta* np.cos(viewing_angle)))
    return doppler_factor


# doppler values fror _1 blazar case, _2 raidogalaxy case
doppler_1 = relativistic_doppler_factor(Gamma_1,viewing_angle_1)
doppler_2 = relativistic_doppler_factor(Gamma_2,viewing_angle_2)

print("doppler1:", doppler_1)
print("doppler2: ", doppler_2)

# make the x and y values into numpy arrays
x_v_boosted               = np.array(x_v)  #*u.Hz
y_v_boosted               = np.array(y_v)  # * (u.erg/( u.s * u.cm *u.cm ))
errorbars_boosted         = np.array(error_bars)   #* (u.erg/( u.s * u.cm *u.cm ))


# x-axis
frequancy = np.logspace(9, 25, N) * u.Hz


##Find the normalization 


U1  =  9/4  * P_e_1/(2*np.pi * Gamma_1**2 *R_1**2 * const.c.cgs)  #* 4/3
U2 =   9/4  * P_e_2/(2*np.pi * Gamma_2**2 *R_2**2 * const.c.cgs)  #* 4/3
EL = ElectronDistribution()
BPL = BrokenPowerLaw()
#SSC = SynchrotronSelfCompton()
#SYN  = Synchrotron()

k_e1_result = BPL.from_normalised_energy_density(U1,gamma_min = gamma_min_1, gamma_max= gamma_max_1,p1= 2, p2=4.8, gamma_b= gamma_break_1)
k_e2_result = BPL.from_normalised_energy_density(U2, gamma_min = gamma_min_2, gamma_max= gamma_max_2,p1= 2, p2=4.8, gamma_b= gamma_break_2)

k_e1 = k_e1_result.k_e
k_e2 = k_e2_result.k_e
print("ke1: ", k_e1)
print("ke2: ", k_e2)

# theese values where used for the test case used under: 
#k_e1 = 9*10**(-3)       * u.Unit("cm-3") 
#k_e2 = 9.18*10**(-5)    * u.Unit("cm-3") 


## Synchotron and SynchotronSelfCompton 
#   1: blazar case  
#   2: radiogalaxy case


# Method 1. define the emmisikon region with the Blob class
parameters_1 = { 
    "p1": p1, 
    "p2": p2,
    "gamma_b": gamma_break_1,
    "gamma_min" : gamma_min_1, 
    "gamma_max": gamma_max_1,

}

parameters_2 = { 
    "p1": p1, 
    "p2": p2,
    "gamma_b": gamma_break_2,
    "gamma_min" : gamma_min_2, 
    "gamma_max": gamma_max_2,

}


spectrum_dict_1 = {"type": "BrokenPowerLaw", "parameters": parameters_1}
spectrum_dict_2 = {"type": "BrokenPowerLaw", "parameters": parameters_2}


blob_1 =  Blob(R_1,z,doppler_1, Gamma_1, B_1, k_e1, spectrum_dict_1, spectrum_norm_type="differential" )
blob_2 =  Blob(R_2,z,doppler_2, Gamma_2, B_2, k_e2, spectrum_dict_2, spectrum_norm_type="differential" )
#print("blob1: ", blob_1)
#print("blob2: ", blob_2)

# have to use spectrum_norm_type = "differential", because we are using k_e to normalize the electron spectra (we could also have used n_e_tot or W_e)
#print(blob)


# The power of the jet
#print(f"jet power in particles blob 1: {blob_1.P_jet_e:.2e}")
#print(f"jet power in particles blob 2: {blob_2.P_jet_e:.2e}")
#print(f" jet power in B_1: {blob_1.P_jet_B:.2e}")
#print(f" jet power in B_2: {blob_2.P_jet_B:.2e}")


# define the radiative processes
synch_1 = Synchrotron(blob_1,ssa= True)
ssc_1   = SynchrotronSelfCompton(blob_1,ssa = True)

synch_2 = Synchrotron(blob_2,ssa= True)
ssc_2   = SynchrotronSelfCompton(blob_2,ssa = True)



synch_sed_1 = synch_1.sed_flux(frequancy)
synch_sed_2 = synch_2.sed_flux(frequancy)
ssc_sed_1   = ssc_1.sed_flux(frequancy)
ssc_sed_2   = ssc_2.sed_flux(frequancy)



# Method 2. use the other sed_flux function, which is the .evaluate_sed_flux 
def calc(x,z_value, Luminocity_distance, doppler_value, magnetic_field, Radius,k_ee,power_1,power_2, G_break, G_min, G_max):
    synch = Synchrotron.evaluate_sed_flux(
                x,
                z_value,
                Luminocity_distance,
                doppler_value,
                magnetic_field,
                Radius,
                BrokenPowerLaw,
                k_ee,
                power_1,
                power_2,
                G_break,
                G_min,
                G_max,
                ssa = True
    )

    ssc = SynchrotronSelfCompton.evaluate_sed_flux(
                x,
                z_value,
                Luminocity_distance,
                doppler_value,
                magnetic_field,
                Radius,
                BrokenPowerLaw,
                k_ee,
                power_1,
                power_2,
                G_break,
                G_min,
                G_max,
                ssa = True
          )

    return synch + ssc


# PLOTS

#The # correspond the the numbering of the figures. 
# 1: Blazar case 
# 2: radiogalaxy case

# matplotlib adjustments
load_mpl_rc()

# Plotting the data points  
plt.scatter(x_v_boosted,y_v_boosted,label = "data", color = "royalblue")
plt.errorbar(x_v_boosted, y_v_boosted, yerr = error_bars, xerr = 0, color = "royalblue", fmt= "none")

#1: Blazar Case 
#plot_sed(frequancy, synch_sed_1, label = "Synchrotron")
#plot_sed(frequancy, ssc_sed_1, label = "SSC")
#plot_sed(frequancy, synch_sed_1+ssc_sed_1, label = "Sum")
#plot_sed(frequancy, calc(frequancy,z, dis, doppler_1, B_1, R_1, k_e1,p1,p2,gamma_break_1, gamma_min_1, gamma_max_1  ), label="syn + ssc", color = "red")
#plt.ylim(10**-16,10**-7)
#plt.savefig("Tol1326379_blazar_case.png")


#print(blob_2.Gamma)

# 2: radiogalaxy case 
plot_sed(frequancy,synch_sed_2, label = "Synchrotron")
plot_sed(frequancy,ssc_sed_2, label = "SSC")
plot_sed(frequancy,synch_sed_2+ssc_sed_2, label = "Sum")
plot_sed(frequancy, calc(frequancy,z, dis, doppler_2, B_2, R_2, k_e2,p1,p2,gamma_break_2, gamma_min_2, gamma_max_2 ), label="syn + ssc", color = "red")
plt.ylim(10**-16,10**-10)
plt.savefig("Tol1326379_radiogalaxy_case.png")



# Plotting with other values for the normalization
#plot_sed(frequancy, calc(frequancy,z, dis, doppler_1, B_1, R_1, k_e1,p1,p2,gamma_break_1, gamma_min_1, gamma_max_1  ), label="blazar", color = "darkorange")
#plot_sed(frequancy, calc(frequancy,z, dis, doppler_2, B_2, R_2, k_e2,p1,p2,gamma_break_2, gamma_min_2, gamma_max_2 ), label="radio galaxy", color = "black")
#plt.ylim(3 * 10**-16,5 *10**-11)
#plt.savefig("Testplot.png")
#plt.show()