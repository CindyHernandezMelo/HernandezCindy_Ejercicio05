import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 

%matplotlib inline
ejemplo = 'scscccsssssc'


pH = np.ones(len(ejemplo))
Hs = np.linspace(0,1,100)

pobsH = np.ones(np.shape(Hs))

for H in range(np.shape(Hs)[0]):
    for i in ejemplo:
        if i == 's':
            pobsH[H] = pobsH[H]*Hs[H]
        else:
            pobsH[H] = pobsH[H]*(1-Hs[H])


pobs = np.sum(pobsH)*Hs[1]
pHobs = pobsH/pobs 
L = np.log(pHobs)
#def bayesiana(ejemplo):

L = L[1:-1]
#derivada L
dL = (L[2:]-L[0:-2])/(2*Hs[1])

arg_max = np.argmin(abs(dL))
H0 = Hs[arg_max]
dL2 = (dL[2:]-dL[0:-2])/(2*Hs[1])
sigma  = ((-(dL[2:]-dL[0:-2])/(2*Hs[1]))**(-0.5))[arg_max]

plt.plot(Hs,pHobs)
from scipy.stats import norm
np.pi
sigma**2

gausiana = norm.pdf(Hs,H0,sigma)
plt.plot(Hs,gausiana,'--')
plt.title('H = %f $\pm$ %f'%(H0,sigma))
plt.xlabel('H')
plt.ylabel('P(H|{obs})')

plt.savefig('bayes.pdf')
