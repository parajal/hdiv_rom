from ngsolve import *
from .global_settings import *
__all__ = ["NavierStokes", "NavierStokesExplicit"]
class NavierStokes:

    def __init__(self, mesh, nu,
                 inflow="", outflow="", wall="", slip="",
                 ubnd = None,
                 timestep = None,
                 order=2,
                 omega_rot = (0.0,0.0,0.0),
                 volumeforce=None,
                 divpenalty=1e6,
                 formulation="HdivHDG"
                 ):
        self.mesh = mesh
        self.nu = nu
        self.with_pressure = (divpenalty == None) or (divpenalty == 0.0)
        if divpenalty == None:
            self.divpenalty = 0
        else:
            divpenalty = nu * divpenalty
            self.divpenalty = divpenalty
        self.timestep = timestep
        if ubnd == None:
            self.ubnd = CoefficientFunction(tuple([0 for i in range(mesh.dim)]))
        else:
            self.ubnd = ubnd
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        self.slip = slip
        self.omega_rot = CoefficientFunction(omega_rot)

        if not (self.omega_rot.dim == 3):
            raise Exception("omega_rot should a 3D vector (also in 2D).")

        if mesh.dim == 2:
            if (abs(Integrate(Norm(self.omega_rot)**2,mesh) - Integrate((self.omega_rot[2])**2,mesh)) > 1e-12):
                raise Exception("omega_rot should only be non-zero in the third component")

        assert(formulation in ["HdivHDG","HybridMCS"])

        VHDiv = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall+"|"+slip, hodivfree=False)
        Q = L2(mesh, order = order-1)
        if formulation == "HdivHDG":
            Vhat = TangentialFacetFESpace(mesh, order=order, dirichlet=inflow+"|"+wall+"|"+outflow,
                                          highest_order_dc=False, hide_highest_order_dc=True)
            Vhat = Compress(Vhat)
            fespacelist = [VHDiv, Vhat]
        elif formulation == "HybridMCS":
            Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow)
            Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)
            if mesh.dim == 2:
                S = L2(mesh, order=order-1)
            else:
                S = VectorL2(mesh, order=order-1)
            Sigma.SetCouplingType(IntRange(0,Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
            Sigma = Compress(Sigma)
            S.SetCouplingType(IntRange(0,S.ndof), COUPLING_TYPE.HIDDEN_DOF)
            S = Compress(S)
            fespacelist = [VHDiv, Vhat, Sigma, S]

        if self.with_pressure:
            fespacelist.append(Q)
        self.V = FESpace (fespacelist)

        tnt_pairs = list(zip(self.V.TrialFunction(),self.V.TestFunction()))
        u,v = tnt_pairs[0]
        uhat,vhat = tnt_pairs[1]
        if self.with_pressure:
            p, q = tnt_pairs[-1]

        if formulation == "HybridMCS":
            sigma, tau = tnt_pairs[2]
            W, R = tnt_pairs[3]

        if mesh.dim == 2:
            def Skew2Vec(m):
                return m[1,0]-m[0,1]
        else:
            def Skew2Vec(m):
                return CoefficientFunction( (m[0,1]-m[1,0], m[2,0]-m[0,2], m[1,2]-m[2,1]) )

        dS = dx(element_boundary=True)
        n = specialcf.normal(mesh.dim)

        def tang(u): return u-(u*n)*n
        def eps(u): return 0.5*(Grad(u)+Grad(u).trans)


        ### viscosity discretization form:
        if formulation == "HdivHDG":
            h = specialcf.mesh_size
            sigma = 10*(order+1)*order/h
            viscosity = 2 * nu * InnerProduct(eps(u),eps(v)) * dx \
                       + (( InnerProduct(-2*nu*eps(u)*n,tang(v-vhat)) \
                       +InnerProduct(-2*nu*eps(v)*n,tang(u-uhat)) \
                       +(sigma/h)*nu*InnerProduct(tang(u-uhat),tang(v-vhat)))) * dS
        else:
            viscosity = -0.5/nu * InnerProduct(sigma,tau) * dx \
                       + (div(sigma)*v+div(tau)*u) * dx \
                       + (InnerProduct(W,Skew2Vec(tau)) + InnerProduct(R,Skew2Vec(sigma))) * dx \
                       - (((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS \
                       + (-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS \

        ### Coriolis discretization form:
        self.use_coriolis = Integrate(Norm(self.omega_rot)**2,mesh) > 0
        viscosity_coriolis = viscosity
        if self.use_coriolis:
            if mesh.dim == 3:
                viscosity_coriolis += 2 * Cross(self.omega_rot,u) * v * dx
            else:
                viscosity_coriolis += 2 * self.omega_rot[2] * (-u[1]*v[0]+u[0]*v[1]) * dx

        ### velocity-pressure coupling form:

        if self.with_pressure:
            divconstraint = (div(u)*q+p*div(v)-1e-8*p*q) * dx
        else:
            divconstraint = self.divpenalty*div(u)*div(v) * dx

        ### mass form:

        mass = u * v * dx

        ### Mstar and Stokes matrix setup:

        self.a = BilinearForm (self.V, eliminate_hidden = True)
        self.a += viscosity_coriolis + divconstraint

        self.gfu = GridFunction(self.V)

        self.f = LinearForm(self.V)

        self.mstar = BilinearForm(self.V, eliminate_hidden = True)
        self.mstar += mass + timestep * viscosity_coriolis + timestep * divconstraint

        ### Convection definition:

        if True:
            u,v = VHDiv.TnT()
            self.conv = BilinearForm(VHDiv, nonassemble=True)
            self.conv += SymbolicBFI(InnerProduct(grad(v)*u, u).Compile(True, wait=True), bonus_intorder=order)
            self.conv += SymbolicBFI((-IfPos(u * n, u*n*u*v, u*n*u.Other(bnd=self.ubnd)*v)).Compile(True, wait=True), element_boundary = True)
            emb = Embedding(self.V.ndof, self.V.Range(0))
            self.conv_operator = emb @ self.conv.mat @ emb.T
        else:
            VL2 = VectorL2(mesh, order=order, piola=True)
            ul2,vl2 = VL2.TnT()
            self.conv_l2 = BilinearForm(VL2, nonassemble=True)

            self.conv_l2 += SymbolicBFI(-InnerProduct(Grad(ul2).trans*ul2, vl2).Compile(realcompile=realcompile, wait=True),bonus_intorder=order-1)
            self.conv_l2 += SymbolicBFI((-IfPos(ul2 * n, 0, ul2*n*(ul2.Other(bnd=self.ubnd)-ul2)*vl2)).Compile(realcompile=realcompile, wait=True),element_boundary = True, bonus_intorder=order)

            self.convertl2 = VHDiv.ConvertL2Operator(VL2) @ Embedding(self.V.ndof, self.V.Range(0)).T
            self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

        self.gfu.components[0].Set (self.ubnd,BND)
        # self.gfu.components[0].Set (self.ubnd, BND)

        if False: # this is the standard approach which may however not allow for exact integration a.t.m.:
            self.gfu.components[1].Set (tang(self.ubnd), BND)
        else: # this is the manual version of "Set" to obtain bonus_intorder=1
            aF = BilinearForm(Vhat)
            lF = LinearForm(Vhat)
            uhatF, vhatF = Vhat.TnT()
            aF += tang(uhatF.Trace()) * tang(vhatF.Trace()) * ds
            lF += tang(self.ubnd) * tang(vhatF.Trace()) * ds(bonus_intorder=1)
            aF.Assemble()
            lF.Assemble()
            self.gfu.components[1].vec.data = aF.mat.Inverse(~Vhat.FreeDofs(),inverse="sparsecholesky") * lF.vec

        self.invmstar = None
        self.mean_val_pressure = Parameter(0)
        self.divrhs = 0

    @property
    def velocity(self):
        return self.gfu.components[0]

    @property
    def tang(self):
        return self.gfu.components[1]


    @property
    def pressure_without_correction(self):
        if self.with_pressure:
            return self.gfu.components[-1]
        else:
            return self.divpenalty*(div(self.gfu.components[0]) - self.divrhs)

    @property
    def pressure(self):
        if self.outflow == "":
            return self.pressure_without_correction - self.mean_val_pressure
        else:
            return self.pressure_without_correction

    def SetInitial(self, u0):
        self.gfu.components[0].Set (self.u0)

    ### bad copy of SolveInitial (needs clean up)
    def SolveForSupremizer(self,divrhs):

        #self.a.Assemble()
        if not self.with_pressure:
            raise Exception("SolveForSupremizer only works with pressure.")

        f = LinearForm(self.f.space)
        f += divrhs * self.f.space.TestFunction()[-1]  * dx
        f.Assemble()
        temp = self.a.mat.CreateColVector()
        if self.use_coriolis:
            try:
                inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="pardiso")
            except:
                print("pardiso not avail or failed")
                inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="umfpack")
        else:
            inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")

        temp.data = -self.a.mat * self.gfu.vec + f.vec
        temp.data = Projector(self.V.FreeDofs(),True) * temp
        #print("Stokes res:", Norm(temp))
        for i in range(2):
            self.gfu.vec.data = inv * f.vec
            temp.data = -self.a.mat * self.gfu.vec + f.vec
            temp.data = Projector(self.V.FreeDofs(),True) * temp
            #print("Stokes res:", Norm(temp))

        #self.mean_val_pressure.Set(Integrate(self.pressure_without_correction,self.mesh)/Integrate(1,self.mesh))

    def SolveInitial(self):
        self.a.Assemble()
        self.f.Assemble()

        temp = self.a.mat.CreateColVector()
        if self.use_coriolis:
            try:
                inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="pardiso")
            except:
                print("pardiso not avail or failed")
                inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="umfpack")
        else:
            inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")

        temp.data = -self.a.mat * self.gfu.vec + self.f.vec
        temp.data = Projector(self.V.FreeDofs(),True) * temp
        print("Stokes res:", Norm(temp))
        for i in range(2):
            self.gfu.vec.data += inv * temp
            temp.data = -self.a.mat * self.gfu.vec + self.f.vec
            temp.data = Projector(self.V.FreeDofs(),True) * temp
            print("Stokes res:", Norm(temp))

        self.mean_val_pressure.Set(Integrate(self.pressure_without_correction,self.mesh)/Integrate(1,self.mesh))

    def AddForce(self, force):
        self.f += CoefficientFunction(force)*self.V.TestFunction()[0]*dx(bonus_intorder=0)

    def AddTangentialForce(self, force, boundary):
        n = specialcf.normal(self.mesh.dim)
        def tang(u): return u-(u*n)*n
        self.f += CoefficientFunction(force)*tang(self.V.TestFunction()[1].Trace())*ds(bonus_intorder=0,definedon=self.mesh.Boundaries(boundary))

    def AddDivRHS(self, rhs):
        self.divrhs = rhs
        rhs = CoefficientFunction(rhs)
        if self.with_pressure:
            q  = self.V.TestFunction()[-1]
            self.f += rhs*q*dx(bonus_intorder=0)
        else:
            v = self.V.TestFunction()[0]
            self.f += self.divpenalty*rhs*div(v)*dx(bonus_intorder=0)

    def DoTimeStep(self):

        if not self.invmstar:
            self.mstar.Assemble()
            if self.use_coriolis:
                try:
                    self.invmstar = self.mstar.mat.Inverse(self.V.FreeDofs(), inverse="pardiso")
                except:
                    print("pardiso not avail or failed")
                    self.invmstar = self.mstar.mat.Inverse(self.V.FreeDofs(), inverse="umfpack")
            else:
                self.invmstar = self.mstar.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")

        self.temp = self.a.mat.CreateColVector()
        self.f.Assemble()
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec

        self.gfu.vec.data += self.timestep * self.invmstar * self.temp

        self.mean_val_pressure.Set(Integrate(self.pressure_without_correction,self.mesh)/Integrate(1,self.mesh))



class NavierStokesExplicit:

    def __init__(self, mesh, nu,
                 inflow="", outflow="", wall="", slip="",
                 ubnd = None,
                 timestep = None,
                 order=2,
                 volumeforce=None
                 ):
        self.mesh = mesh
        self.nu = nu
        self.timestep = timestep
        if ubnd == None:
            self.ubnd = CoefficientFunction(tuple([0 for i in range(mesh.dim)]))
        else:
            self.ubnd = ubnd
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        self.slip = slip

        # W = HDiv(mesh, order=order, hodivfree = False, discontinuous = True)
        W = VectorL2(mesh, order=order, piola=True)
        Q = L2(mesh, order = order-1)
        QF = FacetFESpace(mesh, order = order, dirichlet=outflow)


        self.V = W*Q*QF

        tnt_pairs = list(zip(self.V.TrialFunction(),self.V.TestFunction()))
        u,v = tnt_pairs[0]
        p,q = tnt_pairs[1]
        pF,qF = tnt_pairs[2]

        dS = dx(element_boundary=True)

        n = specialcf.normal(mesh.dim)


        h = specialcf.mesh_size

        ### velocity-pressure coupling form:
        divconstraint = -(div(u)*q+p*div(v)+1e-8*p*q) * dx
        divconstraint += (u*n * qF + v*n * pF -1e-8*pF*qF) * dS

        ### mass form:
        mass = u * v * dx

        self.mass = BilinearForm(self.V, nonassemble=True)
        self.mass += mass

        self.divconstraint = BilinearForm(self.V, nonassemble=True)
        self.divconstraint += divconstraint

        ### Mhelmh and Stokes matrix setup:

        self.gfu = GridFunction(self.V)

        self.mhelmh = BilinearForm(self.V)
        self.mhelmh += mass + divconstraint

        ### Convection definition:

        w = u
        wn = w*n
        self.convdiff = BilinearForm(self.V, nonassemble=True)

        self.convdiff += SymbolicBFI(-InnerProduct(Grad(u).trans*u, v).Compile(realcompile=realcompile, wait=True),bonus_intorder=order-1)
        self.convdiff += SymbolicBFI((-IfPos(u * n, 0, u*n*(u.Other(bnd=self.ubnd)-u)*v)).Compile(realcompile=realcompile, wait=True), element_boundary = True, bonus_intorder=order)

        dX = dx(skeleton=True)
        dXbnd = ds(skeleton=True)

        def eps(u): return 0.5*(Grad(u)+Grad(u).trans)
        self.convdiff += (-nu * InnerProduct(eps(u), eps(v))*dx).Compile(realcompile=realcompile, wait=True)
        self.convdiff += (nu * 0.5*(eps(u)+eps(u.Other()))*n*(v-v.Other())*dX).Compile(realcompile=realcompile, wait=True)
        self.convdiff += (nu * 0.5*(eps(v)+eps(v.Other()))*n*(u-u.Other())*dX).Compile(realcompile=realcompile, wait=True)
        self.convdiff += (-nu * 5*order**2/h*(u-u.Other())*(v-v.Other())*dX).Compile(realcompile=realcompile, wait=True)

        self.convdiff += (nu * eps(u)*n*v*dXbnd).Compile(realcompile=realcompile, wait=True)
        self.convdiff += (nu * eps(v)*n*(u - self.ubnd)*dXbnd).Compile(realcompile=realcompile, wait=True)
        self.convdiff += (-nu * 5*order**2/h*InnerProduct(u-self.ubnd,v)*dXbnd).Compile(realcompile=realcompile, wait=True)

        self.fbnd = LinearForm(self.V)
        self.fbnd += self.ubnd * n * qF.Trace()* ds(definedon=mesh.Boundaries(inflow+"|"+wall))
        self.fbnd.Assemble()

        self.invmhelmh = None
        self.temp = self.fbnd.vec.CreateVector()


    @property
    def velocity(self):
        return self.gfu.components[0]

    @property
    def pressure(self):
        print("warning: pressure is not correct a.t.m.")
        return self.gfu.components[1]

    def SetInitial(self, u0):
        self.gfu.components[0].Set (u0)

    def SolveInitial(self, u0=None):
        if not self.invmhelmh:
            self.mhelmh.Assemble()
        self.invmhelmh = self.mhelmh.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")

        if u0 != None:
            self.f = LinearForm(self.V)
            self.f += u0*self.V.TestFunction()[0]*dx
            self.f.Assemble()
            self.temp.data = self.f.vec + self.fbnd.vec
        else:
            self.temp.data = self.fbnd.vec

        self.gfu.vec.data = self.invmhelmh * self.temp


    def AddForce(self, force):
        raise Exception("not implemented")

    def AddTangentialForce(self, force, boundary):
        raise Exception("not implemented")

    def AddDivRHS(self, rhs):
        raise Exception("not implemented")

    def DoTimeStep(self):

        if not self.invmhelmh:
            self.mhelmh.Assemble()
            self.invmhelmh = self.mhelmh.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")

        self.convdiff.Apply(self.gfu.vec,self.temp)
        self.gfu.vec.data += self.timestep * self.invmhelmh * self.temp
