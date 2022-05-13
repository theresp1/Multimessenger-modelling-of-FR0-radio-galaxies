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
import astropy.constants as const

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
M_BH   =  1.2 * 1e9 * const.M_sun
m_dot  = 2.2698e+23 * u.Unit("g s-1")
eta    = (L_disk / (m_dot * c ** 2)).to_value("")
R_g     = ((G * M_BH) / c ** 2).to("cm")
R_in    = 3   #* R_g
R_out   = 100 #* R_g #too much with 1000?  
mu_s   = 9.7227e-01
disk   = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
print("disk ::", disk)

nu          = np.logspace(9, 27, 200) * u.Hz
#BLR 
xi_line      = 6.0000e-01
epsilon_line = 9.5367e+09
R_line       = 1.0000e+14  
#BLR          = SphericalShellBLR()

# define the radiative processes
synch       = Synchrotron(blob, ssa=True)
ssc         = SynchrotronSelfCompton(blob, synch)
ec_disk_1   = ExternalCompton(blob, disk, r=1e17 * u.cm)
ec_disk_sed_1 = ec_disk_1.sed_flux(nu)

print("ec_disk_1:", ec_disk_1)
print("ec_disk_sed_1  ::", ec_disk_sed_1 )

# SEDs

synch_sed   = synch.sed_flux(nu)
ssc_sed     = ssc.sed_flux(nu)

disk_bb_sed = disk.sed_flux(nu, z)
#print("disk_bb_sed::",disk_bb_sed)
#disk_ec_SSD = ec_SSD.sed_flux(nu)

total_sed   = synch_sed + ssc_sed  +disk_bb_sed #+ disk_ec_SSD