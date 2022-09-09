#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as spo

f=open("Ran_Data.txt","r")
lines=f.readlines()
firstCol=[]
secondCol = []

for x in lines:
    firstCol.append(x.split('   ')[0].split(' '))
    secondCol.append(x.split('  ')[2].split(' '))
f.close()


X_1 = [float(firstCol[i][0]) for i in range(len(firstCol))]
Y_1 = [float(secondCol[i][0]) for i in range(len(secondCol))]
#X = np.array(X_1)
#Y = np.array(Y_1)

def f(x,paramt1): 
    a, b, c, d, e,  = paramt1
    return a + b*x + c*(x**2) + d*(x**3) + e*(x**4) 
lambda_0=[]
max_x=[]
for i in range (len(X_1)-1,5,-1):
  X=[]
  Y=[]
  for j in range(0,i):
    X.append(X_1[j])
    Y.append(Y_1[j])
  def fit_function(paramt): #Defining chi-square function with the help of experimental result and theoritical formula
    a, b, c, d, e = paramt
    y=0
    for i in range(len(X)):
        y += (Y[i]-f(X[i],paramt))**2
    return y

  guess=[1.5,1,1,1, 1 ] #guessing the value for optimization
  result=spo.minimize(fit_function, guess, tol=0.1)
  lambda_0.append(result.x[0])
  max_x.append(max(X))
print (lambda_0)
print (max_x)

import matplotlib.pyplot as plt
plt.plot(max_x, lambda_0, 'o:')
plt.show()

