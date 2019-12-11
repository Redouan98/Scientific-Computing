# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math as math
from scipy.linalg import lu_factor, lu_solve
from scipy import sparse
from scipy.sparse import identity
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sla
from scipy.linalg import norm


   
       

        

def fill(h):
    n = int(1./h)
    A=csr_matrix((n-1,n-1)).toarray()
    A[0,0]=2
    A[0,1]=-1
    A[n-2,n-2]=2
    A[n-2,n-3]=-1
    for i in range(1,n-2):
        A[i,i-1]=-1.
        A[i,i]=2.
        A[i,i+1]=-1.
    return sparse.csr_matrix(A*(1/(h**2)))


    
def fillin2Dk(h):
    n = int(1./h)
    return sparse.kron(identity(n-1),fill(h))+sparse.kron(fill(h),identity(n-1))

def fillin3Dk(h):
    n = int(1./h)
    return sparse.kron(fill(h),sparse.kron(identity(n-1),identity(n-1)))+sparse.kron(identity(n-1),sparse.kron(fill(h),identity(n-1)))+sparse.kron(identity(n-1),sparse.kron(identity((n-1)),fill(h)))
    
    

def use(h):
    n = int(1./h)
    A=csr_matrix(((n+1)*(n+1),(n+1)*(n+1)))
    for i in range(n+1):
        A[i,i]=1
    for i in range((n+1)*(n+1)-n-1,(n+1)*(n+1)):
        A[i,i]=1
    for i in range(1,n):
        A[i*(n+1),i*(n+1)]=1
        A[i*(n+1)+n,i*(n+1)+n]=1
    B=fillin2Dk(h)
    for y in range(1,n):
        for x in range(1,n):
            A[y*(n+1)+x,y*(n+1)+x]=B.tocsr()[0,0]
            A[y*(n+1)+x,y*(n+1)+x+1]=B.tocsr()[0,0]/-4.
            A[y*(n+1)+x,y*(n+1)+x-1]=B.tocsr()[0,0]/-4.
            A[y*(n+1)+x,y*(n+1)+x+n+1]=B.tocsr()[0,0]/-4.
            A[y*(n+1)+x,y*(n+1)+x-n-1]=B.tocsr()[0,0]/-4.
    return A


t=use(1/(2**4)).toarray()
print(t)

def use2(h):
    n = int(1./h)
    A=csr_matrix(((n+1)*(n+1)*(n+1),(n+1)*(n+1)*(n+1)))
    for i in range((n+1)*(n+1)):
        A[i,i]=1
    for i in range((n+1)*(n+1)*(n+1)-(n+1)*(n+1),(n+1)*(n+1)*(n+1)):
        A[i,i]=1
    for i in range(1,n):
        for x in range(n+1):
            A[i*((n+1)*(n+1))+x,i*((n+1)*(n+1))+x]=1
            A[i*((n+1)*(n+1))+(n+1)*(n+1)-(n+1)+x,i*((n+1)*(n+1))+(n+1)*(n+1)-(n+1)+x]=1
    for i in range(1,n):
        for t in range(1,n):
            A[i*(n+1)*(n+1)+t*(n+1),i*(n+1)*(n+1)+t*(n+1)]=1
            A[i*(n+1)*(n+1)+t*(n+1)+n,i*(n+1)*(n+1)+t*(n+1)+n]=1
    B=fillin3Dk(h)
    for z in range(1,n):
        for y in range(1,n):
            for x in range(1,n):
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x]=B.tocsr()[0,0]
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x-1]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x+1]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x-n-1]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x+n+1]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x-1]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x+(n+1)**2]=B.tocsr()[0,0]/-6.
                A[z*(n+1)**2+y*(n+1)+x,z*(n+1)**2+y*(n+1)+x-(n+1)**2]=B.tocsr()[0,0]/-6.   
    return A



def rightside2(h):
    n=int(1./h)
    A=csr_matrix(((n+1)*(n+1),1))
    for i in range(n+1):
        A[i,0]=0
    for i in range((n+1)*(n+1)-n-1,(n+1)*(n+1)):
        A[i,0]=math.sin((i-(n+1)*(n+1)+n+1)*h)
    for i in range(1,n):
        A[i*(n+1),0]=0
        A[i*(n+1)+n,0]=math.sin(i*h)
    for y in range(1,n):
        for x in range(1,n):
            A[y*(n+1)+x,0]=((x*h)**2+(y*h)**2)*math.sin(x*h*y*h)
    return A
g=rightside2(1/4.).toarray()
    
    

def rightside3(h):
    n=int(1./h)
    A=csr_matrix(((n+1)*(n+1)*(n+1),1))
    for i in range((n+1)**2):
        A[i,0]=0
    for z in range(1,n):
        for i in range(1,n):
            A[z*(n+1)**2+(n+1)*i,0]=0
            A[z*(n+1)**2+(n+1)*i+n,0]=math.sin(z*h*i*h)
    for z in range(1,n):
        for i in range(n):
            A[z*(n+1)**2+i,0]=0
            A[z*(n+1)**2+i+n*(n+1),0]=math.sin(i*h+z*h)
    d=(n+1)**3-(n+1)**2
    for y in range(n+1):
        for x in range(n+1):
            A[d,0]=math.sin(x*h*y*h)
            d=d+1
    for z in range(1,n):
        for y in range(1,n):
            for x in range(1,n):
                A[z*(n+1)**2+(n+1)*y+x,0]=math.sin(x*h*y*h*z*h)*((x*h)**2+(y*h)**2+(z*h)**2)
    return A
            
    
s=rightside3(1/8.).toarray()





def Question2A():
    for p in range(2,11):
        h=1./(2**p)
        lu=sla.splu(use(h))
        b=rightside2(h)
        x=lu.solve(b.toarray())
        print(norm(x-exact2(h),np.inf))





def Question3A():
    for p in range(2,9):
        h=1./(2**p)
        lu=sla.splu(use2(h))
        b=rightside3(h)
        x=lu.solve(b.toarray())
        print(norm(x-exact3(h),np.inf))

        

                    
def exact2(h):
    n=int(1./h)
    ex=np.zeros(((n+1)*(n+1),1))
    d=0
    for y in range(n+1):
        for x in range(n+1):
            ex[d,0]=math.sin(x*h*y*h)
            d=d+1
    return ex

def exact3(h):
    n=int(1./h)
    ex=np.zeros(((n+1)*(n+1)*(n+1),1))
    d=0
    for z in range(n+1):
        for y in range(n+1):
            for  x in range(n+1):
                ex[d,0]=math.sin(x*h*y*h*z*h)
                d=d+1
    return ex                 
            

Question2A()    

            

        
        
        
        
        
        
        
        
        
        



