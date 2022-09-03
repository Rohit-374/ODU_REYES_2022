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

#Plotting the data
plt.xscale("log") #Setting x-axis to log scale
plt.scatter(kinetic_exp, sigma_exp, c ="green")
#plt.plot(kinetic_exp, sigma_exp)
plt.xlabel('Kinetic Energy (KeV)', fontsize = 12)
plt.ylabel('Cross-Section (barn)', fontsize = 12)

plt.show() #Print Plot

################################################################################################################

import numpy as np
import scipy.optimize as spo 

def f(x,paramt1): # Defining sigma_total function interms of x(kinetic energy) and 4 parmeters a_s,r_s,a_t,r_t where s= singlet; t=triplet
    a_s,r_s,a_t,r_t=paramt1
    return ( (np.pi) / ( ((1/a_s) - ((1/4)*r_s*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01 + ( (3*np.pi)/ ( ((1/a_t) - ((1/4)*r_t*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01


def fit_function(paramt): #Defining chi-square function with the help of experimental result and theoritical formula
    a_s,r_s,a_t,r_t=paramt
    y=0
    for i in range (len(kinetic_exp)):
        y += ((sigma_exp[i]-f(kinetic_exp[i],paramt))/dsigma_exp[i])**2
    return y

guess=[-23,2,5,1] #guessing the value for optimization
result=spo.minimize(fit_function, guess, tol=0.00001)
print (result, '\n')

################################################################################################################

#fitting the experimental result with theroritical formula

a1 = result.x[0]
r1 = result.x[1]
a2 = result.x[2]
r2 = result.x[3]

x = np.array([i for i in range(0,10000)])
y=( (np.pi) / ( ((1/a1) - ((1/4)*r1*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01 + ( (3*np.pi)/ ( ((1/a2) - ((1/4)*r2*940*25.6889*10**(-9)*x))**2 + ((1/2)*940*25.6889*10**(-9)*x) ))*0.01

#Plotting Fitted Curve
plt.xscale("log")
plt.plot(x,y,'-r')
plt.plot(kinetic_exp, sigma_exp, 'og')
plt.xlabel('Kinetic Energy(KeV)')
plt.ylabel('Cross-Section (barn)')
plt.show()

m=938.92 #Average mass of a Nucleon

#Calculating Energy of Triplet Bound-State
M=2*np.sqrt(m**2+(2*(197.3)**2/(result.x[3])**2)*((result.x[3]/result.x[2])-1+np.sqrt((1-2*(result.x[3]/result.x[2])))))

#Calculating Binding-Energy of Triplet Bound-State
BE=2*m - M

print('The total mass (energy) of triplet bound state is', M)
print('The binding-energy of triplet bound state is', BE)
