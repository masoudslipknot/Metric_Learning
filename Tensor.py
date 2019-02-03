import numpy as np
import tensorly as tl
import math
from numpy import linalg as LA
import tensorflow as tf

def g(z):
    result = 1/(3*math.log(1+math.exp(-3*z),10))
    return result



R = 3
M = 3
P = 3
D1 = np.matrix([[10, math.log(0.01, 10)],[10, math.log(0.02, 10)],[10, math.log(0.015, 10)],[10.5, math.log(0.02, 10)],[15, math.log(0.4, 10)],[16, math.log(0.5, 10)],[18, math.log(0.6, 10)],[15.5, math.log(0.45, 10)],[20, math.log(7.75, 10)]])
D2 = np.matrix([[20, math.log(0.15, 10)],[10, math.log(8.02, 10)],[10, math.log(5.015, 10)],[20.5, math.log(2.02, 10)],[35, math.log(3.4, 10)],[36, math.log(2.5, 10)],[18.5, math.log(3.6, 10)],[25.7, math.log(3.45, 10)],[28, math.log(9.35, 10)]])
D3 = np.matrix([[30, math.log(3.01, 10)],[10, math.log(4.02, 10)],[10, math.log(8.015, 10)],[30.6, math.log(7.09, 10)],[47, math.log(7.4, 10)],[26, math.log(1.5, 10)],[29.5, math.log(5.6, 10)],[35.9, math.log(7.68, 10)],[87, math.log(12.75, 10)]])


delta1=[]
delta2=[]
delta3=[]


for v in range(0,8):
    for c in range(1,9):
        delta1.append(D1[v,:]-D1[c,:])

for v in range(0,8):
    for c in range(1,9):
        delta2.append(D1[v,:]-D1[c,:])

for v in range(0,8):
    for c in range(1,9):
        delta3.append(D1[v,:]-D1[c,:])






U1 = tl.tensor(np.arange(36).reshape((9, 2, 2)))
U2 = tl.tensor(np.arange(36).reshape((9, 2, 2)))
U3 = tl.tensor(np.arange(36).reshape((9, 2, 2)))

ALLU = [U1, U2, U3]

w = 0                                          # w is covarianse of tensor
sum2 = 0
sum1 = 0
b = 0
ER =0
nm = 9
Ymk = [-1, 1]
Nprim = (nm*(nm-1))/2

ALLDELTA = [delta1, delta2, delta3]




#ER = tf.eye  #identity tesnor that i should find
#print(ER)

ans = tl.mode_dot(ER, ALLU[0], 1)
for i in range(0,P):
    for j in range(1,M):
        ans = tl.mode_dot(ans, ALLU[j], j)
    b = w^P - ans
    sum2 = sum2 + LA.norm(b,'fro')^2

sum3 = 0
Gamma = [2, 1, 3]
for n in range(0, M):
    sum3 = sum3 + Gamma[n]*LA.norm(ALLU[n],1)
#print(LA.norm(b, 'fro'))
for m in range(1,M):
    for k in range(1,Nprim):
        curdel = ALLDELTA[m]
        sum1 = sum1 + g(Ymk[0](1-np.matmul(np.matmul(np.matmul(curdel[k].transpose(),ALLU[m])),ALLU[m].transpose()),curdel[k]))
