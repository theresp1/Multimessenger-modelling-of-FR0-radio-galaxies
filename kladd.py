import astropy.units as u 
import astropy
import numpy as np 


##0.5-7kev Chandra: 
x_1 = 8.72 * 10**-9 * (2.3 * 10**3 * u.eV).to(u.Hz,equivalencies=u.spectral()) * 10**-(23) * u.Jy
x_2 = 7.56 * 10**-9 * (2.3 * 10**3 * u.eV).to(u.Hz,equivalencies=u.spectral()) * 10**-(23) * u.Jy
#print(x)
#y = x*8.726 *10**(-9) * 10**-(23) * u.Jy 
#print(1*u.Hz).to(u.eV,equivalencies=u.spectral())
sum_band_1 = x_1 *2 + x_2 * 2
#print("x_1",x_1)
#print("x_2",x_2)
#print("sum band 1: ", sum_band_1)

##1.2-2keV chandra



f = np.array([1,2,3])
print(f)
p = [0,9,99]
f = np.append(f,p)
print(f)
