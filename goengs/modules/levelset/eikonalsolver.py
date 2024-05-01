#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem import GetDofsOfElements

class eikonalsolver:
    """
    Solver for the eikonal equation with righthandside equal to 1. The solution is the distantance to the nearest wall.
    Parameters
    ----------
    mesh: ngsolve.comp.Mesh
        computational mesh
    order: int
        order of the element space
    BND: str
        regular expression for the boundaries
    dt: float, optional
        timestep
    diffusioncoefficient: float, optional 
        diffusioncoefficient, needs to be larger for coarser meshes
    
    Notes
    -----
    This module provides a solver for the eikonal equation on a domain :math:'Omega' with boundary :math:'\partial Omega'
    .. math:: |\nabla d| = 1, \ in\  Omega,\ 0 \ on \ \partial Omega 
    We reformulate this by squaring the equation to use an upwind DG formulation for the lefthandside. We then add a time derivative of the field to look for stationary solutions. For stability we introduce an artificial diffusion term. The diffusioncoefficient depends on the meshsize, for finer grids the diffusioncoefficient and thus the model error can be reduced. Using a higher order usually improves the accuracy.
    """
    
    def __init__(self, mesh, order, BND="", dt = 0.1, diffusioncoefficient = 0.01, solve_on_elements=None):
        
        self.mesh = mesh
        self.order = order
        self.V = L2(self.mesh, order = self.order, dgjumps = True)
        self.u, self.v = self.V.TnT()
        self.gfu = GridFunction(self.V)
        self.dt = dt
        self.difcoeff = diffusioncoefficient
        self.BND = BND
        # Draw(self.gfu, self.mesh, "distance")
        
        # declaration of instance attributes
        
        self.diffusion = BilinearForm(self.V)
        self.gradterm = BilinearForm(self.V)
        self.invmstar = 0
        
        
        #Set righthandside
        self.f = BilinearForm(self.V)
        self.f += self.u * self.v *dx + self.dt* IfPos(self.gfu,1,-1)*self.v * dx
        self.f.Assemble()
        
        #Assemble mass matrix
        self.m = BilinearForm(self.V)
        self.m += self.u*self.v*dx 
        self.m.Assemble()

        if solve_on_elements:
            
            freedofs = self.V.FreeDofs()
            freedofs &= GetDofsOfElements(self.V,solve_on_elements)

    def Set(self,init):
        self.gfu.Set(init)
        
    def _SetandAssembleDiffusion(self):
        
        """Set and assemble bilinearform for the diffusion""" 
        u = self.u
        v = self.v
        
        jump_u = u-u.Other()
        jump_v = v-v.Other()
        n = specialcf.normal(2)
        mean_dudn = 0.5*n * (grad(u)+grad(u.Other()))
        mean_dvdn = 0.5*n * (grad(v)+grad(v.Other()))
        
        alpha = 4
        h = specialcf.mesh_size
        self.diffusion = BilinearForm(self.V)
        # scale=IfPos(self.gfu,self.gfu,-self.gfu)*10
        scale=1
        self.diffusion += scale*grad(u)*grad(v) * dx \
        +scale*alpha*self.order**2/h*jump_u*jump_v * dx(skeleton=True) \
        +scale*(-mean_dudn*jump_v-mean_dvdn*jump_u) * dx(skeleton=True) \
        +scale*alpha*self.order**2/h*u*v * ds(skeleton=True, definedon=self.mesh.Boundaries(self.BND)) \
        +scale*(-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True, definedon=self.mesh.Boundaries(self.BND))
        self.diffusion.Assemble()
    
    def _SetandAssemblegradterm(self):
        
        """Set and assemble bilinearform for the gradterm"""
        u = self.u
        v = self.v
        jump_v = v-v.Other()
        n = specialcf.normal(2)
        dS = dx(skeleton=True)
        
        b = IfPos(self.gfu,1,-1)*grad(self.gfu)
        self.gradterm = BilinearForm(self.V)
        uup = IfPos(b*n, u, u.Other())
        self.gradterm += b * grad(u) * v* dx + b *n * (uup-u) *jump_v * dS
        self.gradterm.Assemble()
        
    def _impliciteEuler(self):
        
        """Compute the inverse for the implicite Euler scheme"""
        self.mstar = self.m.mat.CreateMatrix()
        self.mstar.AsVector().data = self.m.mat.AsVector() +self.dt *self.gradterm.mat.AsVector() + self.dt*self.difcoeff*self.diffusion.mat.AsVector()
        self.invmstar = self.mstar.Inverse(self.V.FreeDofs(), "umfpack")
    
    def DoIteration(self):
        """Do an iteration"""
        with TaskManager():
            rhs = self.gfu.vec.CreateVector()
            self.f.Apply(self.gfu.vec, rhs)
            self._SetandAssembleDiffusion()
            self._SetandAssemblegradterm()
            self._impliciteEuler()
            diffvec = self.gfu.vec.CreateVector()
            rhs.data -= self.mstar * self.gfu.vec
            diffvec.data = self.invmstar * rhs
            self.diff = Norm(diffvec)
            self.gfu.vec.data += diffvec
            # print(self.diff)
            Redraw()
