import os
os.system("export PYTHONIOENCODING=utf8")

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
    xi_line = Parameter("xi_line",0.6, min =0.0, max=1.0)
    epsilon_line = Parameter("epsilon_line",1e6,min= 1e-6, max = 1e16 )
    R_line = Parameter("R_line","1e14 cm", min = 1e4,max = 1e20)

    def __init__(self):
        
        # initialise parameters
        
        self.z = Parameter("z", 0.1, min=0.01, max=0.1)
        self.z.quantity = 0.03690
        # - AGN parameters
        self.z.frozen = True
        self.d_L.quantity = Distance(z=self.z.quantity).to("cm")
        self.d_L.frozen = True
        # - blob parameters
        self.Gamma   =  1.05
        self.delta_D.quantity = 2.6
        Beta    = np.sqrt(1 - 1 / np.power(self.Gamma, 2))  # jet relativistic speed
        mu_s    =  (1 - 1 / (self.Gamma * self.delta_D.quantity)) / Beta  # viewing angle
        self.delta_D.frozen = True
        self.log10_B.quantity =  np.log10(0.2)  
        #agnpy_ssc.log10_B.quantity = np.log10( 1/(6.6466) * (1+z) * (v_s**2)/(2.8*10**6* v_c)  )
        self.log10_B.frozen = True
        #agnpy_ssc.t_var. quantity = 1 * u.d
        #agnpy_ssc.t_var.quantity = (Radius * (1+z))/(c.cgs *2) 
        self.t_var.quantity = 9.5710e+06 
        #agnpy_ssc.t_var.quantity = 4.8970e+06
        self.t_var.frozen = False
        #agnpy_ssc.gamma.quantity = 2
        #agnpy_ssc.gamma.frozen = True
        # - EED
        self.log10_k_e.quantity = -2.3682e+00  
        self.log10_k_e.frozen = False
        self.p1.quantity = 1.9
        self.p2.quantity = 3.5
        self.p1.frozen = True
        self.p2.frozen = True
        self.log10_gamma_b.quantity = np.log10(4000)
        self.log10_gamma_b.frozen = True
        self.log10_gamma_min.quantity = np.log10(10)
        self.log10_gamma_min.frozen = True
        self.log10_gamma_max.quantity = 4.4912e+00
        self.log10_gamma_max.frozen = True

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
        # BLR10_k_e.frozen = False
        self.p1.quantity = 1.9      #fraction of the disk radiation reprocessed by the BLR
        epsilon_line = 1e6                 #dimensionless energy of the emitted line
        R_line       = 1e14                #radius of the BLR spherical shell

        # -- SS disk
        self.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
        self.log10_L_disk.frozen   = True
        self.log10_M_BH.quantity = np.log10(M_BH.to_value("g"))
        self.log10_M_BH.frozen   = True
        self.m_dot.quantity = m_dot
        self.m_dot.frozen    = True
        self.R_in.quantity = R_in
        self.R_in.frozen   = True
        self.R_out.quantity = R_out
        self.R_out.frozen   = True
        # -- Dust Torus
        # agnpy_ec.xi_dt.quantity = xi_dt
        # agnpy_ec.xi_dt.frozen   = True
        # agnpy_ec.T_dt.quantity = T_dt
        # agnpy_ec.T_dt.frozen = True
        # agnpy_ec.R_dt.quantity = R_dt
        # agnpy_ec.R_dt.frozen = True
        # - blob parameters


        # size and location of the emission region
        #t_var   = 7.9062e+05 * u.s
        r       = 5e17 * u.cm
        r_ssd   = 2e15 * u.cm

        self.delta_D.frozen    = True
        self.log10_B.frozen   = True
        self.mu_s.quantity = mu_s
        self.mu_s.frozen   = True
        #agnpy_ec.t_var.quantity = t_var
        #agnpy_ec.t_var.frozen   = False
        self.log10_r.quantity = np.log10(r.to_value("cm"))
        self.log10_r.frozen   = True
        self.log10_r_ssd.quantity = np.log10(r_ssd.to_value("cm"))
        self.log10_r_ssd.frozen   = True



   




def main():
    AEC = AgnpyEC(filename)
    print(AEC.delta_D)


if __name__ == "__main__":
    main()