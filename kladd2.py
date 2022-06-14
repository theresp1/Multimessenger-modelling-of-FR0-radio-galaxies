import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import Distance
import ast
from astropy.constants import c, G, M_sun, k_B, sigma_sb, m_e, h, e
from astropy.constants import astropyconst20 as const


class RIAF:


    def __init__(self,L,xi,f_min,f_break,f_max,R,alpha_1,alpha_2): 
        self.name    = "RIAF" 
        self.L       = L
        self.xi      = xi
        self.f_min   = f_min
        self.f_max   = f_max
        self.f_break = f_break
        self.R       = R
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2 


    @staticmethod
    def brokenpowerlaw(nu,A,z,f_min,f_break,f_max, alpha_1,alpha_2):
        K   = np.zeros(np.size(nu))  
        for i in range(np.size(nu)):
            nu[i] *= 1+z
            if ((nu[i] < f_break) and (nu[i] > f_min)): 
                K[i] =  A * (nu[i]/f_break)**-(alpha_1)

            elif ((nu[i] > f_break) and (nu[i] < f_max)): 
                K[i] = A* (nu[i]/f_break)**(-alpha_2)

            else: 
                K[i] = 0

        return K 


    @staticmethod
    def evaluate_bb_sed(nu,A,z,f_min,f_break,f_max,R,alpha_1, alpha_2):
        nu *= 1+z 
        d_L = Distance(z=z).to("cm").value
        prefactor = np.pi * np.power((R / d_L), 2)  #* u.sr
        I_nu =  RIAF.brokenpowerlaw(nu,A,z,f_min,f_break,f_max,alpha_1,alpha_2)
        return (prefactor * nu * I_nu)


x    = np.logspace(10,22,100)
z    = 0.3365 
amp  = 1e-1
Radi = 1e10
a_1  = -0.1
a_2  =  ( (np.log10(1e42 - 9e41)/(np.log10(1e20 - 1e12)))- 1)/2
f_min= 1e10
f_break = 1e12
f_max  = 1e19
y    = RIAF.evaluate_bb_sed(x,amp, z ,f_min,f_break,f_max,Radi, a_1,a_2)
#print(y)
#y    = RIAF.evaluate_bb_sed2(x,amp, z,f_break,Radi, a_1,a_2)

#plt.loglog(x,y)
#plt.show()




class RIAF2:


    def __init__(self): 
    # initialise parameters
        #file = open(filename, "r")
        #contents = file.read()
        #self.input_params = ast.literal_eval(contents)


        self.name    = "RIAF" 
        self.z       = 0.3365
        self.alpha   = 0.3
        self.beta    = 0.5
        self.m       = 5e9
        self.mdot    = 3e-4
        self.c1      = 0.5
        self.c3      = 0.3
        self.T_e     = 1e9
        self.r_in    = 3
        self.r_out   = 1e3


    
    def  cyclosynchrotron(nu,z,alpha,beta,c1,c3,mdot,m,T_e,amp,R_min,R_max):
        theta_e = 0.1686 ## TODO: change this
        nu *= 1+z
        N   = np.size(nu)
        r   = np.linspace(R_min,R_max,N)
        B   = np.zeros(N) 
        L   = np.zeros(N)
        nuL = np.zeros(N)
        x_m = np.zeros(N) 
        s2  = np.zeros(N)
        for i in range(N): 
            s1 = 1.42*1e9 * (alpha)**-1/2 * (1-beta)**1/2  * c1**(-1/2) * c3**(1/2)
            s3 = 1.05e-24
            
            B[i]   = (s1 * m**(-1/2) * mdot**(1/2) * r[i]**-5/4) 
            nu_v   = ((const.e * B[i] * u.G)/(2* np.pi* m_e * c)).to("cm-1")
            print("nu_v :", nu_v)
            x_m[i] = ((2 * nu[i])/ (3* nu_v.value * (theta_e)**2)) 
            s2[i]  = 1.19*1e-13 * x_m[i]
            
            L[i]   = s3 * (s1*s2[i])**8/5 * (m)**6/5 * (mdot)**4/5 * (T_e )**21/5 * (nu[i])**2/5 
            print("L :",L[i] )
            nuL[i] = nu[i] * L[i]

        return nuL


    def F(): 
        theta_e = 0.1686 #TODO: this changes with T
        if theta_e < 1: 
            return 4* (2*theta_e/(np.pi)**3)**1/2 * (1+1.781*theta_e**(1.34)) + 1.73*(theta_e)**(3/2)*(1+1.1*theta_e**2 -1.25*theta_e**5/2)

        elif theta_e > 1:
            return (9* theta_e/2*np.pi)* np.ln((1.123*theta_e +0.48)+1.5) +2.3 * theta_e * (np.ln(1.123*theta_e + 1.28))

    def Bremsstrahlung(nu,alpha,c1,r_min,r_max,mdot,m,T_e):
        L   = np.zeros(np.size(nu))
        nuL = np.zeros(np.size(nu))
        for i in range(np.size(nu)):
            L[i]   = 2.29 * 1e24 * alpha**-2 * c1**-2 * np.log(r_max/r_min) * RIAF2.F()  * T_e**-1 * np.exp(-(h * nu[i] * 1/u.s)/(k_B * T_e*u.K)) * m * mdot**2
            nuL[i] = nu[i] * L[i]

        return nuL

 # array to integrate R
        R = np.linspace(R_in, R_out, 100)
        _R, _nu = axes_reshaper(R, nu)
        _T = SSDisk.evaluate_T(_R, M_BH, m_dot, R_in)
        #print("T: ", _T )
        _I_nu = BlackBody().evaluate(_nu, _T, scale=1)
        integrand = _R / np.power(d_L, 2) * _I_nu * u.sr
        F_nu = 2 * np.pi * np.trapz(integrand, R, axis=0)
        return mu_s * (nu * F_nu).to("erg cm-2 s-1")

z       = 0.3365
alpha   = 0.3
beta    = 0.5
m       = 5e9
mdot    = 3e-4
c1      = 0.5
c3      = 0.3
T_e     = 1e9
r_in    = 3
r_out   = 1e3
ampp    = 1e-198





x    = np.logspace(10,12,100) #* u.Hz
# x2   = np.logspace(12,20)
y    = RIAF2.cyclosynchrotron(x,z,alpha,beta,c1,c3,mdot,m,T_e,r_in,r_out,ampp)
# y2   = RIAF2.Bremsstrahlung(x2,alpha,c1,r_in,r_out,mdot,m,T_e)

plt.loglog(x,y)
# plt.loglog(x2,y2)
plt.show()