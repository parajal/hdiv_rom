import scipy.linalg
from scipy import random
import scipy.sparse as sp
import numpy as np
from numpy import linalg
from ngsolve import *


class rapod:
    def __init__(self,s,vecs,nodebound,rootbound,solvecs,var,timestep,tend):
        self.s= s
        self.vecs = vecs
        self.nodebound = nodebound
        self.rootbound = rootbound
        self.snap = []
        self.solvecs = solvecs
        self.mycnt = 0
        self.var = var
        self.timestep = timestep
        self.tend = tend
        self.spacing = 100
        self.slice = np.ceil(self.tend/(self.spacing*self.timestep))


    def rapod_step(self,fvecs):
        #Appending dominant modes to the snapshot matrix
        for j in range(len(self.vecs)):
            fvecs.Append(self.vecs[j])

        if self.mycnt >=1 and self.mycnt < self.slice:
            #print("node")
            self.snap.append((self.spacing))
            newbound = (self.nodebound*np.sqrt(np.sum(self.snap))/np.sqrt(self.slice -1))**2
            #print("newbound",newbound)
            asmall =  InnerProduct (fvecs, self.s *  fvecs)
            ev,evec = scipy.linalg.eigh(a = asmall)
            #Sorting eigenvalues in decreasing order
            idx = np.argsort(ev)[::-1]
            ev = ev[idx]
            evec = evec[:,idx]
            #Computing number of dominant modes
            i = 1
            ratio = ev[len(ev)-1]**2
            while ratio <= newbound and i <= len(ev):
                ratio +=  ev[len(ev)-i-1]**2
                i += 1
            M = len(ev)-i+1
            #print(M)
            self.vecs = MultiVector(self.var,M)
            if M > 0:
                self.vecs[:] = fvecs * Matrix(evec[:,0:M])
                d = [np.sqrt(abs(ev[k])) for k in range(M)]
                self.vecs.Scale(Vector(d))


        else:
            #print("root")
            self.snap.append((self.spacing))
            newbound =(np.sqrt(np.sum(self.snap))*self.rootbound)**2

            asmall =  InnerProduct (fvecs, self.s *  fvecs)
            ev,evec = scipy.linalg.eigh(a = asmall)
            #Sorting eigenvalues in decreasing order
            idx = np.argsort(ev)[::-1]
            ev = ev[idx]
            evec = evec[:,idx]
            #Computing the number of dominant modes
            i = 1
            ratio = ev[len(ev)-1]**2
            while ratio <= newbound and i <= len(ev):
                ratio +=  ev[len(ev)-i-1]**2
                i += 1
            M = len(ev)-i+1
            #Computing dominant modes
            self.vecs = MultiVector(self.var,M)
            if M > 0:
                self.vecs[:] = fvecs * Matrix(evec[:,0:M])
                d = [1/np.sqrt(abs(ev[k])) for k in range(M)]
                self.vecs.Scale(Vector(d))

    #Slices the snapshot matrix
    def Append(self,vec):
        self.solvecs.Append(vec)
        if len(self.solvecs)%self.spacing == 0:
            fvecs = self.solvecs[self.mycnt*(self.spacing):(self.mycnt+1)*self.spacing]
            self.mycnt += 1
            #print("self.mycnt at append stage",self.mycnt)
            self.rapod_step(fvecs)

    def output(self):
        return self.vecs

class pod:
    def __init__(self,s,bound,solvecs):
        self.s= s
        self.bound = bound
        self.solvecs = solvecs
        #self.total_snap = total_snap

    def Append(self,vec):
        self.solvecs.Append(vec)

    def pod_only(self):
        asmall =  InnerProduct ( self.solvecs , self.s *  self.solvecs )
        ev,evec = scipy.linalg.eigh(a = asmall)

        #Sorting eigenvalues in decreasing order
        idx = np.argsort(ev)[::-1]
        ev = ev[idx]
        evec = evec[:,idx]

        #Computing number of dominant modes
        i = 1
        ratio = ev[len(ev)-1]
        while ratio <= self.bound**2*len(self.solvecs):
            ratio +=  ev[len(ev)-i-1]
            i += 1
        M = len(ev)-i+1
        return M

        #print("number of POD dominant modes",len(ev)-i +1)
    def output(self):
        return self.pod_only()
