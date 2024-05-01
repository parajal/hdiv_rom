#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normal extension of a function defined on a zero level of a level set function.
The module is in a very early stage. There are several TODOs:
# * doc-strings 
# * convergence tests / studies for higher order case
# * implement and test the higher order case for the scenario that 
#   everything should be considered on an undeformed domain. 
"""

import sys
import os
pwd = os.path.abspath(os.getcwd())
sys.path.insert(1,pwd+'/../../..')
sys.path.insert(1,pwd+'/../..')
sys.path.insert(1,pwd+'/..')
sys.path.insert(1,pwd+'/.')

import math
from netgen.geom2d import SplineGeometry
import netgen

from xfem.lsetcurv import *
from xfem import *
from ngsolve import *
from goengs.modules.global_settings import *


class NormalExtension:
    """
Class to compute a proper normal extension. Assuming a function `gfu` is given
on a zero level of a level set function (here P1 +/- mesh deformation), we 
compute the discrete solution on the mesh that is obtained by extending the 
function constant in normal direction. To this end a TraceFEM mass matrix 
problem (on the surface) with a normal diffusion stabilization (in the 
surrounding bulk) is first solved on the cut element. Afterwards the normal
diffusion is used to extend this function layer by layer through the mesh. 
    """

    def __init__(self, gfu, lsetmeshadap = None):
        self.gfu = gfu
        self.Vh = gfu.space
        self.order = self.Vh.globalorder
        print("order = ", self.order)
        self.mesh = gfu.space.mesh
        self.gfext = GridFunction(self.Vh)
        if lsetmeshadap:
            self.ownlsetadap = False
            self.lsetmeshadap = lsetmeshadap
        else:   
            self.ownlsetadap = True
            self.lsetmeshadap = LevelSetMeshAdaptation(self.mesh, order=self.order, threshold=1000, discontinuous_qn=True)
        self.gfu_on_mapped = GridFunction(self.Vh)
        self.ci = CutInfo(self.mesh)

    @property
    def extension(self):
        return self.gfext

    @property
    def levelsetp1(self):
        return self.lsetmeshadap.lset_p1

    @property
    def deformation(self):
        return self.lsetmeshadap.deform

    @TimeFunction
    def Extend(self, lset, on_mapped_mesh=False, layerwidth = None, showsteps = False):

        Vh = self.Vh
        
        if self.ownlsetadap:
            self.lsetmeshadap.CalcDeformation(lset)
            # with self.lsetmeshadap:
            #     print("maxdist:",self.lsetmeshadap.CalcMaxDistance(lset))
        deformation = self.lsetmeshadap.deform

        lsetp1 = self.lsetmeshadap.lset_p1
        lset_if = {"levelset": lsetp1, "domain_type": IF, "subdivlvl": 0}

        if on_mapped_mesh:
            self.gfu_on_mapped.vec.data = self.gfu.vec
        else:
            self.mesh.UnsetDeformation()
            self.gfu_on_mapped.Set(shifted_eval(self.gfu, forth = deformation))
            # print("error0:",sqrt(IntegrateX(lset_if, self.mesh, (self.gfu_on_mapped - shifted_eval(self.gfu, forth = deformation))**2)))
            self.mesh.SetDeformation(deformation)


        n = Normalize(grad(lsetp1))
        h = specialcf.mesh_size

        self.ci.Update(lsetp1)

        ba_IF = BitArray(self.mesh.ne)
        ba_IF[:] = self.ci.GetElementsOfType(IF)

        freedofs1 = BitArray(Vh.ndof)
        freedofs1[:] = Vh.FreeDofs()
        freedofs1 &= GetDofsOfElements(Vh, ba_IF)

        u, v = Vh.TnT()

        a = BilinearForm(Vh, symmetric=True, check_unused=False)
        a += SymbolicCutBFI(levelset_domain=lset_if, form=InnerProduct(u,v))
        a += SymbolicBFI(form=h*(InnerProduct(Grad(u) * n, Grad(v) * n)),
                         definedonelements=ba_IF)

        f = LinearForm(Vh)
        f += SymbolicCutLFI(levelset_domain=lset_if, form=InnerProduct(self.gfu_on_mapped, v))

        with Timer("Assembly1"): 
            a.Assemble()
            f.Assemble()

        svec = self.gfu_on_mapped.vec
        svec[:] = 0.0
        with Timer("LinSolve2"):
            svec.data = a.mat.Inverse(freedofs1) * f.vec
        
        unsolveddofs = sum(Vh.FreeDofs()) - sum(freedofs1)
        lset_neg = {"levelset": lsetp1, "domain_type": NEG, "subdivlvl": 0}
        lset_pos = {"levelset": lsetp1, "domain_type": POS, "subdivlvl": 0}

        ind = BitArrayCF(self.ci.GetElementsOfType(IF))
        dist =IntegrationPointExtrema(lset_pos,self.mesh,ind*lsetp1)[1] - IntegrationPointExtrema(lset_neg,self.mesh,ind*lsetp1)[0]
        lsetshift = lsetp1.vec.CreateVector()
        lsetshift[:] = 0.5*dist if layerwidth == None else layerwidth

        lsetshifted_neg = GridFunction(lsetp1.space)
        lsetshifted_neg.vec.data = lsetp1.vec
        lsetshifted_pos = GridFunction(lsetp1.space)
        lsetshifted_pos.vec.data = lsetp1.vec

        ba_band = BitArray(self.mesh.ne)
        ba_band_old = BitArray(self.mesh.ne)
        ba_band_old[:] = ba_IF

        freedofs2 = BitArray(Vh.ndof)

        cnt = 0
        if showsteps:
            self.gfext.vec.data = svec
            Redraw()
            input("Interface band is solved for. Starting extension layer-wise:")
        fac = 1
        timer2 = Timer("Assembly2")
        timer3 = Timer("LinSolve2")
        while unsolveddofs > 0:
            lsetshifted_neg.vec.data += fac * lsetshift
            lsetshifted_pos.vec.data -= fac * lsetshift
            haspos = BitArray(Vh.ndof)
            hasneg = BitArray(Vh.ndof)
            self.ci.Update(lsetshifted_neg)
            haspos[:] = self.ci.GetElementsOfType(HASPOS)
            self.ci.Update(lsetshifted_pos)
            hasneg[:] = self.ci.GetElementsOfType(HASNEG)

            ba_band[:] = haspos & hasneg
            ba_band &= ~ ba_band_old
            ba_band_old = haspos & hasneg

            freedofs2[:] = Vh.FreeDofs()
            freedofs2 &= GetDofsOfElements(Vh, ba_band)
            freedofs2 &= ~freedofs1

            freedofs1 |= freedofs2


            a = BilinearForm(Vh, symmetric=True, check_unused=False)
            do = dx(definedonelements=ba_band)
            a += InnerProduct(Grad(u) * n, Grad(v) * n) * do
            #a += IfPos(lsetp1,lsetp1,-lsetp1)/dist * 0.1 * fac *InnerProduct(Grad(u), Grad(v)) *do
    
            f = LinearForm(Vh)
            with timer2:
                a.Assemble()
                f.Assemble()

            with timer3:
                f.vec.data = -a.mat * svec
                svec.data += a.mat.Inverse(freedofs2) * f.vec

            cnt += 1
            fac *= 2

            if showsteps:
                self.gfext.vec.data = svec
                Redraw()
                print("finished step " + str(cnt) + " which reduced the dofs that need to be solved for from ", end="")
                print(unsolveddofs,"to ",end="")
            unsolveddofs -= sum(freedofs2)
            if showsteps:
                print(unsolveddofs, end="")
                input("")
        if ngsglobals.msg_level > 2:
            print("normal extension done in "+str(cnt)+" steps.")

        if not on_mapped_mesh:
            self.mesh.UnsetDeformation()
            self.gfext.Set(shifted_eval(self.gfu_on_mapped, back = deformation))
            # print("error1:",sqrt(IntegrateX(lset_if, self.mesh, (self.gfu_on_mapped - shifted_eval(self.gfext, back = deformation))**2)))
        else:
            self.gfext.vec.data = svec
            # print("error1:",sqrt(IntegrateX(lset_if, self.mesh, (self.gfext - self.gfu)**2)))
        return self.gfext




if __name__ == "__main__":
    #from ngsolve.webgui import Draw
    ngsglobals.msg_level = 0
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1))
    lasterr = None
    for ref in range(5):
        mesh = Mesh(geo.GenerateMesh(maxh=1/2**(2+ref)))
        with TaskManager(pajetrace=10*1000*1000*1000):

            lset0 = sqrt((x-0.5)**2+(y-0.5)**2)-0.3

            gfu = GridFunction(H1(mesh,order=1,dim=2))
            sol = (-y+0.5,x-0.5)

            lsetmeshadap = LevelSetMeshAdaptation(mesh, order=gfu.space.globalorder, threshold=1000, discontinuous_qn=True)
            mesh.SetDeformation(lsetmeshadap.deform)
            lsetmeshadap.CalcDeformation(lset0)

            gfu.Set(sol)
            ne = NormalExtension(gfu,lsetmeshadap=lsetmeshadap)
            Draw(ne.levelsetp1, mesh, "lsetp1")
            Draw(gfu, mesh, "gfu")
            Draw(ne.deformation, mesh, "deformation")
            Draw(ne.gfext, mesh, "gfext")
            ne.Extend(lset0,on_mapped_mesh=True, showsteps=False)
            lsetp1 = ne.lsetmeshadap.lset_p1
            lset_if = {"levelset": lset0, "domain_type": IF, "subdivlvl": 0}
            err = sqrt(IntegrateX(lset_if, mesh, Norm(ne.extension - sol)**2))
            print("deviation from initial interface function (in L2):",err,end="")
            if lasterr:
                eoc = log(lasterr/err)/log(2)
                print(", eoc:", eoc)
            else:   
                print("")
            lasterr = err
            mesh.UnsetDeformation()
            input("")
