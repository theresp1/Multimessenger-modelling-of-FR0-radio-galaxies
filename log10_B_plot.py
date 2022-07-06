# import numpy, astropy and matplotlib for basic functionalities
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

# #scan B from fit 12 in the "output_new_test_data" folder (plot 12 was succsessfull)
# df_1        = pd.read_csv('results_best_fit/B_CHI2plot_best_fit.csv')
# log_10_B    = df_1["log_10_B"].copy()   
# chi2        = df_1["chi2"].copy()

# #scan B from fit 13 in the "output_new_test_data" folder (plot 13 was not succsessfull)
# df_2        = pd.read_csv('output_scan_B/chi2-B_plot.csv')
# log_10_B_2  = df_2["log_10_B"].copy()   
# chi2_2      = df_2["chi2"].copy()

# #Only succsessfull fits from df_1
# df_3        = pd.read_csv('results_best_fit/B_CHI2plot_best-fit_success.csv')
# log_10_B_3  = df_3["log_10_B"].copy()   
# chi2_3      = df_3["chi2"].copy()

# #scan over delta while having B unfrozen (all fits failed)(Changed B for each fit(should have has it set to the samw value))
# df_4        = pd.read_csv('Results_best_fit3/delta_chi2nr2.csv')
# delta_4     = df_4["delta"].copy()   
# chi2_4      = df_4["chi2"].copy()

# #scan over delta while having B frozen (we are changing B for each fit though(should ave had it set to the dame value))
# df_5        = pd.read_csv('Results_best_fit2/delta_chi2.csv')
# delta_5     = df_5["delta"].copy()   
# chi2_5      = df_5["chi2"].copy()


# #successfull fits over delta while having B frozen
# df_6        = pd.read_csv('Results_best_fit2/delta_chi2_s.csv')
# delta_6     = df_6["delta"].copy()   
# chi2_6      = df_6["chi2"].copy()


# ###Fit 1 of the FU DATA###


# # delta 
# df_7       = pd.read_csv('FU/FU_FIT1/delta_chi_s.csv')
# delta_7     = df_7["delta"].copy()   
# chi2_7      = df_7["chi2"].copy()

# # log_10(B) 
df_8        = pd.read_csv('B_chi_s.csv')
B_8         = df_8["log10_B"].copy()   
chi2_8      = df_8["chi2"].copy()


# # p1
# df_9        = pd.read_csv('FU_FIT3/p1_chi_s.csv')
# p1_9        = df_9["p1"].copy()   
# chi2_9      = df_9["chi2"].copy()


# # g_min
# df_10        = pd.read_csv('FU_FIT4/g_min_chi_s.csv')
# g_min_10        = df_10["g_min"].copy()   
# chi2_10      = df_10["chi2"].copy()


# # g_break
# df_11        = pd.read_csv('FU_FIT5/g_break_chi_s.csv')
# g_break_11   = df_11["g_break"].copy()   
# chi2_11      = df_11["chi2"].copy()

# # p2
# df_12        = pd.read_csv('FU_FIT6/P2_chi_s.csv')
# p2_12        = df_12["p2"].copy()   
# chi2_12      = df_12["chi2"].copy()

# # g_max
# df_13        = pd.read_csv('g_max_chi_s.csv')
# g_max_13     = df_13["g_max"].copy()   
# chi2_13      = df_13["chi2"].copy()


#delta =   np.array(delta_8)
B         = np.array(B_8)
# p1        = np.array(p1_9)
# g_min     = np.array(g_min_10)
# g_break   = np.array(g_break_11)
# # p2        = np.array(p2_12)
# g_max     = np.array(g_max_13)
y         = np.array(chi2_8)



plt.scatter(B,y)
plt.xlabel(r"$log_{10}(B)$")
plt.ylabel(r"$\chi^{2}$")
#plt.scatter(p1,y)
#plt.savefig("FU/FU_FIT1/delta_chi_S_plot.png")
#plt.savefig("FU_FIT2/B_chi_S_plot.png")
#plt.savefig("FU_FIT3/p1_chi_S_plot.png")
#plt.savefig("FU_FIT4/g_min_chi_S_plot.png")
#plt.savefig("FU_FIT5/g_break_chi_S_plot.png")
#plt.savefig("FU_FIT6/p2_chi_S_plot.png")
#plt.savefig("FU_FIT7/g_max_chi_S_plot.png")
plt.show()
