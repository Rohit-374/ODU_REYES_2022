from operator import le
import matplotlib.pyplot as plt 

#Defining empty lists to store data
kinetic_exp = []
sigma_exp = []
dsigma_exp=[]
  
f = open('Fitting.txt','r')

#Loop for storing data in list
for row in f:
    row = row.split(',')
    kinetic_exp.append(float(row[1]))
    sigma_exp.append(float(row[2]))
    dsigma_exp.append(float(row[3]))


import numpy as np
import scipy.optimize as spo 

def f(x,paramt1): # Defining sigma_total function interms of x(kinetic energy) and 4 parmeters a_s,
    a_s,r_s,a_t,r_t=paramt1
    return ( (np.pi) / ( ((1/a_s) - ((1/4)*r_s*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01 + ( (3*np.pi)/ ( ((1/a_t) - ((1/4)*r_t*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01

################################################################################################################

#Binding vs E_min

BE1 = []

for j in range (len(kinetic_exp)):
    def fit_function(paramt): #Defining chi-square function with the help of experimental result and theoritical formula
        a_s,r_s,a_t,r_t=paramt
        y=0
        for i in range (j, len(kinetic_exp)): #Make necessary modification here
            y += ((sigma_exp[i]-f(kinetic_exp[i],paramt))/dsigma_exp[i])**2
        return y
    guess=[-23,2,5,1] #guessing the value for optimization
    result=spo.minimize(fit_function, guess, tol=0.00001)
    #print (result)

    m=938.92 #Average mass of a Nucleon

    M = 2*np.sqrt(m**2+(2*(197.3)**2/(result.x[3])**2)*((result.x[3]/result.x[2])-1+np.sqrt((1-2*(result.x[3]/result.x[2])))))

    BE=2*m - M #Calculating Binding Energy

    if BE > 0:
      BE1.append(BE)
    else:
      BE1.append(0)
    print(j) #Just to Keep Track

#plt.ylim([-0.2,4])
plt.xscale("log")
plt.plot(kinetic_exp, BE1 , 'o')
plt.xlabel('$E_{max}$ (Keeping $E_{min}$ Fixed)', fontsize = 12)
plt.ylabel('Binding Energy (MeV)', fontsize = 12)

plt.show()
