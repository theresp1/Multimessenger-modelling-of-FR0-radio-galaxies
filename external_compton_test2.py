# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from astropy.constants import k_B, m_e, c, G, M_sun

# import agnpy classes
from agnpy.emission_regions import Blob
from agnpy.compton import ExternalCompton
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.utils.plot import plot_sed, load_mpl_rc
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from astropy.constants import k_B, m_e, c, G, M_sun
import astropy.constants as const

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus, SphericalShellBLR
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label

load_mpl_rc()


spectrum_norm = 10**-3.2721e+00  * u.Unit("cm-3")
parameters = {
    "p1": 1.0005e+00,
    "p2": 4.1000e+00,
    "gamma_b": 10**(3.4150e+00),
    "gamma_min": 10**(1.6990e+00),
    "gamma_max": 10**(5.7709e+00),
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 5 *10**16 * u.cm
B = 10**(-6.9897e-01 ) * u.G
z = 2.8400e-02
delta_D = 4.0
Gamma = 3.0
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
blob.set_gamma_size(500)
#print(f"total number: {blob.N_e_tot:.2e}")
#print(f"total energy: {blob.W_e:.2e}")

# let us adopt the same disk parameters of Finke 2016
M_sun = const.M_sun.cgs
M_BH =  10 ** 4.2202e+01 * u.Unit("g")
M_BH2=  10 ** 3.7202e+01 * u.Unit("g")
L_disk = 10**(4.3230e+01) * u.Unit("erg s-1")
L_disk2= 10**(4.4230e+01) * u.Unit("erg s-1")
m_dot =2.2698e+23 *u.Unit ("g s-1")
eta =  (L_disk / (m_dot * c ** 2)).to_value("")
R_in = 3
R_in2 = 30
R_out = 100
R_out2 = 1000
#T_dt   = 2.0000e+03 * u.K
#xi_dt  = 5.0000e-01
#R_dt   = 7.2250e+14 * u.cm
r = 10 **1.8699e+01* u.cm
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
disk2= SSDisk(M_BH2, L_disk, eta, R_in, R_out, R_g_units=True)
disk3 = SSDisk(M_BH, L_disk2, eta, R_in, R_out, R_g_units=True)
disk4 = SSDisk(M_BH, L_disk, eta, R_in, R_out2, R_g_units=True)
disk5 = SSDisk(M_BH, L_disk, eta, R_in2, R_out, R_g_units=True)
#dt     = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)

ec_disk_1 = ExternalCompton(blob, disk, r=1e17 * u.cm)
ec_disk_2 = ExternalCompton(blob, disk, r=1e18 * u.cm)
ec_disk_3 = ExternalCompton(blob, disk, r=1e19 * u.cm)
ec_disk_T = ExternalCompton(blob, disk2, r=1e17 * u.cm)
ec_disk_L = ExternalCompton(blob, disk3, r=1e17 * u.cm)
ec_disk_R_out = ExternalCompton(blob, disk4, r=1e17 * u.cm)


#print("ec_disk_1 :",ec_disk_1)

nu = np.logspace(9, 30, 50) * u.Hz

ec_disk_sed_1 = ec_disk_1.sed_flux(nu)
ec_disk_sed_2 = ec_disk_2.sed_flux(nu)
ec_disk_sed_3 = ec_disk_3.sed_flux(nu)
ec_disk_sed_T = ec_disk_T.sed_flux(nu)
ec_disk_sed_L = ec_disk_L.sed_flux(nu)
ec_disk_sed_R_out = ec_disk_R_out.sed_flux(nu)
disk_bb_sed = disk.sed_flux(nu, z)
synch       = Synchrotron(blob, ssa=True)
synch_sed   = synch.sed_flux(nu)
ssc         = SynchrotronSelfCompton(blob, synch)
ssc_sed     = ssc.sed_flux(nu)
#ec_dt       = ExternalCompton(blob, dt, r)
#ec_dt_sed   = ec_dt.sed_flux(nu)
#dt_bb_sed   = dt.sed_flux(nu,z)
disk_bb_sed = disk.sed_flux(nu, z)
#print("ec_disk_sed_1 :",ec_disk_sed_1)
sum_sed = ec_disk_sed_1 + synch_sed + ssc_sed  + disk_bb_sed # +ec_dt_sed + dt_bb_sed
#print("sum_Sed: ", sum_sed)


plot_sed(nu/(1+z), ec_disk_sed_1, color="maroon", label=r"$r=10^{17}\,{\rm cm}$")
plot_sed(nu/(1+z), ec_disk_sed_2, color="crimson", label=r"$r=10^{18}\,{\rm cm}$")
plot_sed(nu/(1+z), ec_disk_sed_3, color="dodgerblue", label=r"$r=10^{19}\,{\rm cm}$")
plot_sed(nu/(1+z), ec_disk_sed_T, color="pink", label=r"higher T")
plot_sed(nu/(1+z), ec_disk_sed_L, color="lime", label=r"higher L")
plot_sed(nu/(1+z), ec_disk_sed_R_out, color="green", label=r"higher R_out")
#plot_sed(nu/(1+z), synch_sed, color = "brown", label = "synch")
#plot_sed(nu/(1+z), ssc_sed, color = "r", label = "ssc")
#plot_sed(nu/(1+z), ec_dt_sed, color = "olive", label = "ec on dt")
#plot_sed(nu/(1+z), dt_bb_sed, color = "aqua", label = "BB dt")
#plot_sed(nu/(1+z), disk_bb_sed, color = "gray", label = "BB dt")
#plot_sed(nu/(1+z), sum_sed, color = "black", label = "SUM")
# set the same axis limits as in the reference figure
#plt.xlim([1e18, 1e29])
plt.ylim([1e-26, 1e-9])
plt.show()









"""""
# Values
p_1           = agnpy_ec.p1.value
p_2           = agnpy_ec.p2.value
gamma_break   = 10 ** agnpy_ec.log10_gamma_b.value
gamma_minimun = 10 ** agnpy_ec.log10_gamma_min.value
gamma_maximum = 10 ** agnpy_ec.log10_gamma_max.value
B_value       = 10 ** agnpy_ec.log10_B.value * u.G
r_value       = 10 ** agnpy_ec.log10_r.value * u.cm
delta_D_value = agnpy_ec.delta_D.value
R_b_value     = 5 *10**16 * u.cm
z             = 2.8400e-02
M_sun         = const.M_sun.cgs
M_BH          = 10 ** agnpy_ec.log10_M_BH.value * u.Unit("g")
L_disk        = 10 ** agnpy_ec.log10_L_disk.value * u.Unit("erg s-1")
m_dot         = agnpy_ec.m_dot.value * u.Unit("g s-1")
eta           =  (L_disk / (m_dot * c ** 2)).to_value("")
R_in          = agnpy_ec.R_in.value * u.cm
R_out         = agnpy_ec.R_out.value * u.cm
T_dt          = agnpy_ec.T_dt.value * u.K
xi_dt         = agnpy_ec.xi_dt.value 
R_dt          = agnpy_ec.R_dt.value * u.cm
r             = 10** (agnpy_ec.log10_r.value)  * u.cm

#load_mpl_rc()

spectrum_norm = 10**-3.2721e+00 * u.Unit("cm-3")
parameters = {
    "p1": p_1,
    "p2": p_2,
    "gamma_b": gamma_break,
    "gamma_min": gamma_minimun,
    "gamma_max": gamma_maximum,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
blob    =  Blob(R_b_value, z, delta_D_value, Gamma, B_value, spectrum_norm, spectrum_dict)
blob.set_gamma_size(500)
#print(f"total number: {blob.N_e_tot:.2e}")
#print(f"total energy: {blob.W_e:.2e}")


disk      = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=False)
dt        = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
ec_disk_1 = ExternalCompton(blob, disk, r=1e17 * u.cm)


print("ec_disk_1 :",ec_disk_1)

nu = np.logspace(9, 30, 50) * u.Hz

ec_disk_sed_1 = ec_disk_1.sed_flux(nu)

disk_bb_sed = disk.sed_flux(nu, z)
synch       = Synchrotron(blob, ssa=True)
synch_sed   = synch.sed_flux(nu)
ssc         = SynchrotronSelfCompton(blob, synch)
ssc_sed     = ssc.sed_flux(nu)
ec_dt       = ExternalCompton(blob, dt, r)
ec_dt_sed   = ec_dt.sed_flux(nu)
dt_bb_sed   = dt.sed_flux(nu,z)
disk_bb_sed = disk.sed_flux(nu, z)
#print("ec_disk_sed_1 :",ec_disk_sed_1)
sum_sed = ec_disk_sed_1 + synch_sed + ssc_sed +ec_dt_sed +dt_bb_sed + disk_bb_sed
#print("sum_Sed: ", sum_sed)


plot_sed(nu/(1+z), ec_disk_sed_1, color="maroon", label=r"$r=10^{17}\,{\rm cm}$")
plot_sed(nu/(1+z), synch_sed, color = "brown", label = "synch")
plot_sed(nu/(1+z), ssc_sed, color = "r", label = "ssc")
plot_sed(nu/(1+z), ec_dt_sed, color = "olive", label = "ec on dt")
plot_sed(nu/(1+z), dt_bb_sed, color = "aqua", label = "BB dt")
plot_sed(nu/(1+z), disk_bb_sed, color = "gray", label = "BB dt")
plot_sed(nu/(1+z), sum_sed, color = "black", label = "SUM")
# set the same axis limits as in the reference figure
#plt.xlim([1e18, 1e29])
plt.ylim([1e-26, 1e-9])
plt.show()


"""""