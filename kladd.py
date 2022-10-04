import astropy.units as u 
import astropy
import numpy as np 
from   astropy.constants import k_B, m_e, c, G, M_sun, sigma_sb, h
import matplotlib.pyplot as plt 

##0.5-7kev Chandra: 
x_1 = 8.72 * 10**-9 * (2.3 * 10**3 * u.eV).to(u.Hz,equivalencies=u.spectral()) * 10**-(23) * u.Jy
x_2 = 7.56 * 10**-9 * (2.3 * 10**3 * u.eV).to(u.Hz,equivalencies=u.spectral()) * 10**-(23) * u.Jy
#print(x)
#y = x*8.726 *10**(-9) * 10**-(23) * u.Jy 
#print(1*u.Hz).to(u.eV,equivalencies=u.spectral())
sum_band_1 = x_1 *2 + x_2 * 2
#print("x_1",x_1)
#print("x_2",x_2)
print("sum band 1: ", sum_band_1)

##1.2-2keV chandra

L_disk  = 1.2e42 * u.Unit("erg s-1")  # disk luminosity
eta     = 1 / 12
m_dot   = (L_disk / (eta * c ** 2)).to("g s-1")  #about 40Msolar
m_dot2  = 40 * M_sun.cgs *1/u.s
m_dot3  = 1e30 * u.g *1/u.s
m_dot4  = 1e28 * u.g *1/u.s
m_dot5  = 1e24 * u.g *1/u.s

print('m_dot2 :', m_dot2)
print('m_dot :', m_dot)
print('test :', 0.5e8 * M_sun)
print(M_sun)
M_BH    = 9.942049353490285e+40 * u.g
R_in    = 22149375570751.87 * u.cm
R_out   = 2953250076100249.5 * u.cm

print('M_BH = ', M_BH)
print('R_in =', R_in)
print('R_out =', R_out)


M_BH2 =  5e8* M_sun.cgs
R_in2 =  6 * ((G * M_BH2 / c ** 2)).cgs
#print('R_in2 :', R_in2)
R_out2 = 400 * ((G * M_BH2 / c ** 2)).cgs


M_BH3  =  5e9* M_sun.cgs
R_in3  =  6 * ((G * M_BH3/ c ** 2)).cgs
R_out3 =  400 * ((G * M_BH3 / c ** 2)).cgs


M_BH4 = 5e6 * M_sun.cgs
R_in4 = 6 * ((G * M_BH4/ c ** 2)).cgs
R_out4 = 400 * ((G * M_BH4 / c ** 2)).cgs

print('M_BH4 :', M_BH4)

def evaluate_T(R, M_BH, m_dot, R_in):
        """black body temperature (K) at the radial coordinate R"""
        phi = 1 - np.sqrt((R_in / R).to_value(""))
        val = (3 * G * M_BH * m_dot * phi) / (8 * np.pi * np.power(R, 3) * sigma_sb)
        return np.power(val, 1 / 4).to("K")


R_values = np.logspace(10,40,100)
R_arr    = np.logspace(np.log10(R_in.value +100),np.log10(R_out.value),100) * u.cm
R_arr2   = np.logspace(np.log10(R_in2.value +100),np.log10(R_out2.value),100) * u.cm
R_arr3   = np.logspace(np.log10(R_in3.value +100),np.log10(R_out3.value),100) * u.cm
R_arr4   = np.logspace(np.log10(R_in4.value +100),np.log10(R_out4.value),100) * u.cm

plt.rcParams.update({'font.size': 18.5})

#plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot2, R_in), label =r'$M_{BH} = 5 \cdot 10^{7}  M_{\odot}$ ')
#plt.loglog(R_arr2, evaluate_T(R_arr2, M_BH2, m_dot2, R_in2 ), label =r'$M_{BH} = 5 \cdot 10^{8} M_{\odot}$ ')
#plt.loglog(R_arr3, evaluate_T(R_arr3, M_BH3, m_dot2, R_in3), label =r'$M_{BH} = 5 \cdot 10^{9} M_{\odot}$ ')
#plt.loglog(R_arr4, evaluate_T(R_arr4, M_BH4, m_dot2, R_in4), label =r'M_BH = 5e6 \cdot M_sun ')

#plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot2, R_in ), label =r'$\dot{m} = 40 M_{\odot}$/s')
plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot3, R_in), label =r'$\dot{M} = 1 \cdot 10^{30}$ g/s')
plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot4, R_in), label =r'$\dot{M} = 1 \cdot 10^{28}$ g/s ')
plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot5, R_in), label =r'$\dot{M} = 1 \cdot 10^{24}$ g/s')
#plt.loglog(R_arr, evaluate_T(R_arr, M_BH, m_dot, R_in), label =r'$\dot{m} =  1.6 \cdot 10^{22}$ g/s')



#print(evaluate_T(R_arr, M_BH, m_dot, R_in))
#print(evaluate_T(R_arr2, M_BH2, m_dot, R_in2 ))
#print('R_in2: ', R_in2/1e12)
#print('R-arr2 :', R_arr2)
#plt.title('R_in = 3Rg, R_out = 400Rg, mdot = 1.6e22 g/s')
#plt.title('R_in = 3Rg, R_out = 400Rg, mdot = 8e34 g/s')
#plt.title('R_in = 3Rg, R_out = 400Rg, M_BH = 5e7 * M_sun')

plt.xlabel(r'$R$ [cm]')
plt.ylabel(r'$T$ [K]')
plt.legend()
plt.show()


#nu_peak_58 = 3.66e14 * 1/u.s
#nu_peak_31 = 3.08e14 * u.Hz
#nu_peak_57 = 3.2943e14 * u.Hz
#nu_peak_55 = 3.2943e+14 * u.Hz
#T_58 = (h * nu_peak_58)/(k_B * 3.93)
#T_31 = (h * nu_peak_31)/(k_B * 3.93)
#T_57 = (h * nu_peak_57)/(k_B * 3.93)
#T_55 = (h * nu_peak_55)/(k_B * 3.93)
#print("Leda58527 T :" , T_58)
#print("Leda31768 T :" , T_31)
#print("Leda57137 T :" , T_57)
#print("Leda55267 T :" , T_55)

