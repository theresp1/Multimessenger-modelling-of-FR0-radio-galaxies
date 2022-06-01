import os
os.system("export PYTHONIOENCODING=utf8")

# import numpy, astropy and matplotlib for basic functionalities
from pickle import TRUE
import pkg_resources
from pathlib import Path
import numpy as np
import ast
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
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
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

mec2 = m_e.to("erg", equivalencies=u.mass_energy())

class AgnpyEC(SpectralModel):
    """Wrapper of agnpy's synchrotron and SSC classes.
    The flux model accounts for the Disk and DT's thermal SEDs.
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters
    whose range is expected to cover several orders of magnitudes (normalisation,
    gammas, size and magnetic field of the blob).
    """
    # default arrays to be used for integration
    gamma_to_integrate = np.logspace(1, 9, 200)
    nu_to_integrate = np.logspace(5, 30, 200) * u.Hz  # used for SSC
    mu_to_integrate = np.linspace(-1, 1, 100)
    phi_to_integrate = np.linspace(0, 2 * np.pi, 50)

    tag = "EC"
 
    def __init__(self,filename):
        # initialise parameters

        file = open(filename, "r")
        contents = file.read()
        self.input_params = ast.literal_eval(contents)

        
        self.z                        = Parameter("z", 0.1, min=0.01, max=0.1)
        self.z.quantity               = 0.03690
        self.z.frozen                 = True

        self.d_L                      = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
        self.d_L.quantity             = Distance(z=self.z.quantity).to("cm")
        self.d_L.frozen               = True


        # - blob parameters
        self.Gamma                    = 1.05
        self.delta_D                  = Parameter("delta_D", 10, min=0, max=10000)
        self.delta_D.quantity         = 2.6
        self.delta_D.frozen           = True
        self.Beta                     = np.sqrt(1 - 1 / np.power(self.Gamma, 2))                   # jet relativistic speed
        self.mu_s                     = Parameter("mu_s", 0.9, min=0.0, max=1.0)               
        self.mu_s.quantity            = (1 - 1 / (self.Gamma * self.delta_D.quantity)) / self.Beta   # viewing angle
        self.mu_s.frozen              = True
        self.log10_B                  = Parameter("log10_B", -2, min=-3, max=1.0)
        self.log10_B.quantity         = np.log10(0.2)  
        self.log10_B.frozen           = True
        self.t_var                    = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)
        self.t_var.quantity           = 9.5710e+06 
        self.t_var.frozen             = False

        # - EED
        self.log10_k_e                = Parameter("log10_k_e", -5, min=-20, max=10)
        self.log10_k_e.quantity       = -2.3682e+00  
        self.log10_k_e.frozen         = False
        self.p1                       = Parameter("p1", 2.1, min=1.0, max=5.0)
        self.p1.quantity              = 1.9
        self.p2                       = Parameter("p2", 3.1, min=-2.0, max=7.0)
        self.p2.quantity              = 3.5
        self.p2.frozen                = True  
        self.log10_gamma_min          = Parameter("log10_gamma_min", 1, min=0, max=4)
        self.log10_gamma_min.quantity = np.log10(10)
        self.log10_gamma_min.frozen   = True
        self.log10_gamma_b            = Parameter("log10_gamma_b", 3, min=1, max=5)
        self.log10_gamma_b.quantity   = np.log10(4000)
        self.log10_gamma_b.frozen     = True
        self.log10_gamma_max          = Parameter("log10_gamma_max", 5, min=4, max=7)
        self.log10_gamma_max.quantity = 4.4912e+00
        self.log10_gamma_max.frozen   = True
        
        # - SS disk
        self.log10_L_disk             = Parameter("log10_L_disk", 45.0, min=39.0, max=48.0)
        self.log10_L_disk.quantity    = np.log10(1.2e42)    #* u.Unit("erg s-1")  # disk luminosity
        self.log10_L_disk.frozen      = True
        self.log10_M_BH               = Parameter("log10_M_BH", 42, min=np.log10(0.8e7 * M_sun.cgs.value), max=np.log10(1.2e11 * M_sun.cgs.value))
        self.log10_M_BH.quantity      = np.log10(0.5e8 * M_sun.cgs.value)               #1e8-1e9 range leave it free 
        self.log10_M_BH.frozen        = True
        self.eta                      = 1 / 12
        self.m_dot                    = Parameter("m_dot", "1e26 g s-1", min=1e24, max=1e30)
        self.m_dot.quantity           = ((10**(self.log10_L_disk.quantity)  * u.Unit("erg s-1") )/ (self.eta * c ** 2)).to("g s-1") #about 40Msolar
        self.m_dot.frozen             = True
        self.R_g                      = ((G * 10**(self.log10_L_disk.quantity) * u.g) / c ** 2).to("cm")
        self.R_in                     = Parameter("R_in", "1e14 cm", min=1e12, max=1e16)
        self.R_in.quantity            = 3 * self.R_g
        self.R_in.frozen              = True
        self.R_out                    = Parameter("R_out", "1e17 cm", min=1e12, max=1e19)
        self.R_out.quantity           = 400 * self.R_g 
        self.R_out.frozen             = True
        self.log10_r_ssd              = Parameter("log10_r_ssd",17, min= 10.0,max=20.0)   #distance between the disk and the blob
        self.log10_r_ssd.quantity     = np.log10(2e15)   #* u.cm
        self.log10_r_ssd.frozen       = True


        # - DT
        self.xi_dt                    = Parameter("xi_dt", 0.6, min=0.0, max=1.0)
        self.xi_dt.quantity           = 0.5                                           # fraction of disk luminosity reprocessed by the DT
        self.xi_dt.frozen             = True
        self.T_dt                     = Parameter("T_dt", "1e3 K", min=1e2, max=1e9)
        self.T_dt.quantity            = 0.5e3 * u.K
        self.T_dt.frozen              = True
        self.log10_R_dt               = Parameter("log10_R_dt", "2.5e18", min=1.0e17, max=1.0e19)       #radius of the ting-like torus
        self.log10_R_dt.quantity      = np.log10( 2.5 * 10**18 * ((10 ** (self.log10_L_disk.quantity)*u.Unit("erg s-1"))/(10**45 * u.Unit("erg s-1")))**2  )  #*u.cm 
        self.log10_R_dt.frozen        = True
        self.log10_r_dt               = Parameter("log10_r_dt",17, min= 10.0,max=20.0)     # distance between the Broad Line Region and the blob (shouldnt it be the distance between the DT and the blob?)
        self.log10_r_dt.quantity      = np.log10(1e17)
        self.log10_r_dt.frozen        = False

        # - BLR 
        self.xi_line                  = Parameter("xi_line",0.6, min =0.0, max=1.0)  #fraction of the disk radiation reprocessed by the BLR
        self.xi_line.quantity         = 0.1
        self.xi_line.frozen           = False
        self.epsilon_line             = Parameter("epsilon_line",1e6,min= 1e-6, max = 1e16 )     #string  dimensionless energy of the emitted line, type of line emitted
        self.epsilon_line.quantity    = 1e3
        self.epsilon_line.frozen      = False
        self.log10_R_line             = Parameter("log10_R_line","1e14 cm", min = 1e4,max = 1e20)  #radius of the BLR spherical shell
        self.log10_R_line.quantity    = np.log10(1e14)           
        self.log10_R_line.frozen      = True
        self.log10_r_blr              = Parameter("log10_r_blr",17, min= 10.0,max=20.0)    #distance between the Broad Line Region and the blob
        self.log10_r_blr.quantity     = np.log10(1e18)
        self.log10_r_blr.frozen       = False



        @staticmethod
        def evaluate(
            self,
            energy,
        ):

        # conversions
            k_e          = 10 ** self.log10_k_e.quantity * u.Unit("cm-3")
            gamma_b      = 10 ** self.log10_gamma_b.quantity
            gamma_min    = 10 ** self.log10_gamma_min.quantity
            gamma_max    = 10 ** self.log10_gamma_max.quantity
            B            = 10 ** self.log10_B.quantity * u.G
            R_b          = (c * self.t_var.quantity * self.delta_D.quantity / (1 + self.z.quantity)).to("cm")
            R_dt         = 10 ** self.log10_R_dt.quantity
            r_dt         = 10 ** self.log10_r_dt.quantity * u.cm
            r_ssd        = 10 ** self.log10_r_ssd.quantity * u.cm
            r_blr        = 10 ** self.log10_r_blr.quantity * u.cm
            L_disk       = 10 ** self.log10_L_disk.quantity * u.Unit("erg s-1")
            M_BH         = 10 ** self.log10_M_BH.quantity * u.Unit("g")
            eps_dt       = 2.7 * (k_B * self.T_dt.quantity / mec2).to_value("")

            nu = energy.to("Hz", equivalencies=u.spectral())
            # non-thermal components
        








def main():
    AEC = AgnpyEC("input/test.txt")
    print(AEC.delta_D)
    print(AEC.input_params["a"])


if __name__ == "__main__":
    main()