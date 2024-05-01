#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Levelset transport class based on Upwind DG formulation.
Includes:
 [x] a preliminary simple Eikonal solver for reparametrization
     away from the cut elements (taken from Anton Georg Schmidt)
     (not very robust)
 [ ] redistancing at the cut interface 
"""


from ngsolve import *
import sys
sys.path.append('../')
sys.path.append('./')
from .global_settings import *
from xfem import InterpolateToP1, CutInfo, IF, HASNEG, HASPOS
from .eikonalsolver import eikonalsolver
__all__ = ["LevelsetTransport"]

class LevelsetTransport:
    
    def __init__(self, mesh, lset_init, wind = None,
                 timestep = None,
                 order=2,
                 ):
        self.mesh = mesh
        self.wind = wind
        self.order = order
        self.dt = Parameter(timestep if timestep != None else 0)
        self.Vh = L2(mesh,order=self.order)
        self.Fh = FacetFESpace(mesh,order=self.order, dirichlet=".*")
        self.Wh = self.Vh * self.Fh
        self.Vh_cont = H1(mesh,order=self.order)
        self.Vh_p1 = H1(mesh,order=1)

        self.gf_lset_hdg = GridFunction(self.Wh)
        self.gf_lset, self.gf_lset_facet = self.gf_lset_hdg.components
        self.gf_lset_cont = GridFunction(self.Vh_cont)
        self.gf_lsetp1 = GridFunction(self.Vh_p1)

        self.gf_lset.Set(lset_init)
        self.gf_lset_cont.Set(self.gf_lset)
        self.gf_lsetp1.Set(self.gf_lset)
        
        self.tmp = self.gf_lset_hdg.vec.CreateVector()
        
        n = specialcf.normal(mesh.dim)
        wn = wind*n
        (u,uF),(v,vF) = self.Wh.TnT()
        self.m = BilinearForm(self.Wh)
        self.m += u*v*dx 
        self.m.Assemble()
        self.mstar = self.m.mat.CreateMatrix()

        convection = wind * grad(u) * v * dx \
            + wn*(uF-u)*IfPos(wn,vF,v) * dx(element_boundary=True) #\
            #+ wn*(uF-u)*IfPos(wn,0,-v) * ds(skeleton=True)
        self.c = BilinearForm(self.Wh)
        self.c += convection

        
    @property
    def lsetp1(self):
        return self.gf_lsetp1
    
    @property
    def lset_cont(self):
        return self.gf_lset_cont

    @property
    def lset(self):
        return self.gf_lset
    
    def SetLevelset(self, u0):
        self.gf_lset.Set (u0)
        
    def SetTimeStep(self, dt):
        if dt != None and dt != self.dt.Get():
            self.dt.Set(dt)
        
    def DoTimeStep(self, dt=None):
        self.SetTimeStep(dt)
            
        self.c.Assemble()
        self.mstar.AsVector().data = self.m.mat.AsVector() + self.dt.Get() * self.c.mat.AsVector()
        try:
            self.invmstar = self.mstar.Inverse(self.Wh.FreeDofs(), inverse="pardiso")
        except:
            print("pardiso not avail or failed")
            self.invmstar = self.mstar.Inverse(self.Wh.FreeDofs(), inverse="umfpack")

        self.gf_lset_facet.Set(self.gf_lset_cont,BND)
        self.tmp.data = - self.c.mat * self.gf_lset_hdg.vec
        self.gf_lset_hdg.vec.data += self.dt.Get() * self.invmstar * self.tmp

        if self.roughness > 0.1:
            print("self.roughness too large :: Trying to repair:",end="")
            ci = CutInfo(mesh, self.lsetp1)
            es = eikonalsolver(mesh = mesh, order = self.order, BND="", dt=0.1, diffusioncoefficient = 0.1*self.roughness, solve_on_elements=~ci.GetElementsOfType(IF))
            es.Set(self.lset)
            for j in range(20):
                es.DoIteration()
                if j > 0:
                    self.lset.Set(es.gfu)
                    Redraw()
                    # print("eik : "+str(j)+" ::: " + str(self.roughness))
                    if self.roughness < 0.05:
                        break
            print("repair done after "+str(j)+" steps. New roughness: " + str(self.roughness))
        
        self.gf_lset_cont.Set(self.gf_lset)
        InterpolateToP1(self.gf_lset_cont,self.gf_lsetp1)
        # self.gf_lsetp1.Set(self.gf_lset)

    @property
    def roughness(self):
        om = Integrate(1,self.mesh)
        r1 = Integrate((Norm(grad(self.gf_lset))-1)**2, self.mesh, order=2*self.order) / om
        return r1
        
import netgen
from netgen.geom2d import SplineGeometry
from ngsolve import *
import math
if __name__ == "__main__":
    ngsglobals.msg_level = 0
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (1, 1), bc = "outer")
    mesh = Mesh( geo.GenerateMesh(maxh=0.04))

    r = sqrt((x-0.5)**2+(y-0.5)**2)
    # wind = IfPos(r-0.5,CoefficientFunction((0,0)),CoefficientFunction((0.5-y,x-0.5)))
    # wind = CoefficientFunction((1,0))
    wind = CoefficientFunction((0.5-y,x-0.5))
    lset0 = sqrt((x-0.6)**2+(y-0.5)**2)-0.2
    lsettransp = LevelsetTransport(mesh, lset0, wind = wind, order=3)
    Draw(lsettransp.lset-lset0,mesh,"lsetdiff")
    Draw(lsettransp.lsetp1,mesh,"lsetp1")
    Draw(lsettransp.lset_cont,mesh,"lset_cont")
    Draw(lsettransp.lset,mesh,"lset")
    # input("")

    dt = 0.01
    T = 2*math.pi
    for i in range(0,int(ceil(T/dt))):
        lsettransp.DoTimeStep(dt)
        Redraw()
        print("t = ", i*dt, "roughness: ", lsettransp.roughness)
        # if (i%10==0):
        #     input("")
