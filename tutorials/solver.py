from ngsolve import *
import numpy as np
from scipy import random
import scipy.sparse.linalg

class vprom:
    def __init__(self,nu,fes,fes_v,var,tol,relax,tend,timestep,tstart,spacing = 100):
        self.fes = fes
        self.nu = nu
        self.fes_v = fes_v
        self.var = var
        self.tol = tol
        self.relax = relax
        self.tend = tend
        self.timestep = timestep
        self.tstart = tstart
        self.snapshot_count = []
        self.mycnt = 0
        self.spacing = spacing
        self.solvecs = MultiVector(self.var,0)
        self.snapshot =  MultiVector(self.var,0)
        u,v   = self.fes.TnT()
        self.mass = BilinearForm(self.fes)
        self.mass += u*v * dx
        self.mass.Assemble()
        self.stiff = BilinearForm(self.fes)
        self.stiff += self.nu*InnerProduct(grad(u),grad(v))*dx
        self.stiff.Assemble()
        #self.initial_energy = InnerProduct(self.mass.mat*self.var,self.var)
        self.nodebound = self.tol * np.sqrt(1 - self.relax**2) #* self.initial_energy
        self.rootbound = self.tol * self.relax #* self.initial_energy
        self.slice = np.ceil((self.tend - self.tstart)/(self.spacing*self.timestep))
        self.vecs = MultiVector(self.var,0)
        u1,v1   = self.fes_v.TnT()
        self.mass_v = BilinearForm(self.fes_v)
        self.mass_v += u1*v1 * dx
        self.mass_v.Assemble()
        self.stiff_v = BilinearForm(self.fes_v)
        self.stiff_v += self.nu*InnerProduct(grad(u1),grad(v1))*dx
        self.stiff_v.Assemble()

    def rapod_step(self,fvecs):
        #Here the dominant modes are appended to the new snapshots.
        for j in range(len(self.vecs)):
            fvecs.Append(self.vecs[j])

        if self.mycnt >= 1 and self.mycnt < self.slice :
            #print("child node")
            if len(self.snapshot_count) == 0:
                self.snapshot_count.append(self.spacing)
            else:
                self.snapshot_count.append( self.spacing + self.spacing - len(self.vecs) )
            newbound = (self.nodebound*np.sqrt(np.sum(self.snapshot_count))/np.sqrt(self.slice-1))**2
            asmall =  InnerProduct ( fvecs , self.mass.mat * fvecs)
            ev,evec = scipy.linalg.eigh(a = asmall)

            #Sorting eigenvalues in decreasing order
            idx = np.argsort((ev))[::-1]
            ev = ev[idx]
            evec = evec[:,idx]

            #Computing number of POD modes
            i =1
            ratio = ev[len(ev)-1]
            while ratio <= newbound and i <= len(ev):
                ratio +=  ev[len(ev)-i-1]
                i += 1
            M = len(ev)-i+2
            #Computing POD modes
            self.vecs = MultiVector(self.var,M)
            if M > 0:
                self.vecs[:] = fvecs * Matrix(evec[:,0:M])
                #Here, we divide by the sqrt of eigenvalues but we also scale by sqrt of eigenvalues. So they cancel each other.
        else:
            #print("root")
            self.snapshot_count.append( self.spacing + self.spacing - len(self.vecs) )

            newbound =(np.sqrt(np.sum(self.snapshot_count))*self.rootbound)**2
            asmall =  InnerProduct ( fvecs , self.mass.mat *  fvecs )
            ev,evec = scipy.linalg.eigh(a = asmall)

            #Sorting eigenvalues in decreasing order
            idx = np.argsort((ev))[::-1]
            ev = ev[idx]
            evec = evec[:,idx]

            #Computing the number of dominant modes
            i = 1
            ratio = ev[len(ev)-1]
            #print("ratio",ratio)
            while ratio <= newbound:
                ratio +=  ev[len(ev)-i-1]
                i += 1
            M = len(ev)-i+2

            #Computing dominant modes
            self.vecs = MultiVector(self.var,M)
            if M > 0:
                self.vecs[:] = fvecs * Matrix(evec[:,0:M])
                d = [1/np.sqrt(abs(ev[k])) for k in range(M)]
                self.vecs.Scale(Vector(d))

    def Append_rapod(self,vec,force_compression = False):
        self.solvecs.Append(vec)
        if len(self.solvecs)%100 == 0 or force_compression:
            #print("counter1 = {0}, counter2 = {1}".format(self.mycnt*spacing, (self.mycnt+1)*spacing))
            fvecs = self.solvecs[self.mycnt*(self.spacing):(self.mycnt+1)*self.spacing]
            self.mycnt += 1
            self.rapod_step(fvecs)

    def output_rapod(self):
            return self.vecs


    def Append_pod(self,vec):
            self.snapshot.Append(vec)

    def pod_only(self):
            asmall =  InnerProduct ( self.snapshot ,self.mass.mat *  self.snapshot )
            ev,evec = scipy.linalg.eigh(a = asmall)
            #Sorting eigenvalues in decreasing order
            idx = np.argsort(ev)[::-1]
            ev = ev[idx]
            evec = evec[:,idx]
            #Computing number of dominant modes
            i = 1
            ratio = ev[len(ev)-1]
            while ratio <= self.tol**2*len(self.snapshot):
                ratio +=  ev[len(ev)-i-1]
                i += 1
            M = len(ev)-i+2
            return M


            #print("number of POD dominant modes",len(ev)-i +1)
    def output_pod(self):
            return self.pod_only()


    def prepare_ROM(self,gfubnd):
        """
        prepares online/ROM computation
        """
        #print("velocity dominant modes:",len(self.vecs))
        self.vecs.Append(gfubnd)
        self.stiff_v_red = InnerProduct(self.vecs, self.stiff.mat * self.vecs)
        self.mass_v_red = InnerProduct(self.vecs, self.mass.mat * self.vecs)
        self.mass_v_red,self.stiff_v_red = self.mass_v_red.NumPy(), self.stiff_v_red.NumPy()
        l = len(self.vecs)
        gfw = GridFunction(self.fes)
        u,v   = self.fes.TnT()
        conv = BilinearForm(self.fes, nonassemble=True)
        conv += (Grad(u) * gfw) * v * dx
        convection=True
        tmp = self.vecs[0].CreateVector()
        if convection:
          self.conv_v_red = np.ndarray(shape=(l,l,l),dtype=float)
          for k in range(l):
              gfw.vec.data = self.vecs[k]
              for j in range(l):
                  tmp = conv.mat * self.vecs[j]
                  for i in range(l):
                      self.conv_v_red[i,j,k] = InnerProduct(tmp, self.vecs[i])


    def simulate_ROM(self,ntimesteps = 10000, timestep = None, initial_vals = None,
                 highdim_output = True):
        """
        simulate the ROM with:
        * semi-implicit Euler (linearized velocity)
        * initial values (in reduced basis)
        * number of time steps
        * time step size
        Result is stored in 'self.sol_v'

        parameters:
         * highdim_output: store the resulting high dim. solution vectors (for testing / evaluation)
         * ...
        """
        if timestep == None:
            timestep = self.timestep
        t = 0
        self.gfu = []
        self.sol_v = MultiVector(self.var,0)
        l = len(self.vecs)
        if initial_vals:
            self.gfu_v[:] = initial_vals
        else:
            self.gfu_v = np.zeros(l)
            self.gfu_v[0:l-2] = 0
            self.gfu_v[l-1] = 1
        self.gfu.append(self.gfu_v.copy())
        self.sol_v.Append(self.vecs * Vector(self.gfu_v))
        #navstokes.velocity.vec.data = self.sol_v[-1]
        convection = True
        for i in range(ntimesteps):
                S_N = self.mass_v_red + self.stiff_v_red * timestep
                if convection:
                    S_N += timestep * self.conv_v_red @ self.gfu_v
                S_N_star = S_N[0:l-1,0:l-1]
                F_N = self.mass_v_red @ self.gfu_v - S_N[:,l-1]
                F_N_star = F_N[0:l-1]

                self.gfu_v[0:l-1] =  (np.linalg.solve(S_N_star, F_N_star))
                self.gfu.append(self.gfu_v.copy())
                if highdim_output:
                    self.sol_v.Append(self.vecs * Vector(self.gfu_v))

    def error_v_L2(self):
        """
        compare solutions (mean L2 error) of high dim and ROM models assuming
        the same time distance between list entries.
        """
        num = len(self.snapshot)
        gfu_fom = GridFunction(self.fes)
        gfu_rom = GridFunction(self.fes)
        err = 0
        for i in range(num):
            gfu_fom.vec.data = self.snapshot[i]
            gfu_rom.vec.data = self.sol_v[i]
            err += Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,self.fes.mesh)
            #print("err",err)
        err = np.sqrt(err) / num
        print("L2 velocity error",err)

    def error_v_l2(self):
        """
        compare solutions (l2 error) of high dim and ROM models assuming
        the same time distance between list entries.
        """
        num = len(self.snapshot)
        snapshot = np.zeros( (self.fes.ndof ,num))
        solution = np.zeros( (self.fes.ndof ,num))
        norm = 0
        for i in range(num):

            snapshot[:,i] = self.snapshot[i][:]
            solution[:,i] = self.sol_v[i][:]
        diff = snapshot - solution
        for i in range(num):
            norm += np.linalg.norm(diff[:,i]**2)
        err = np.sqrt(norm)/num
        print("l2 velocity error", err)


    def compare_err_v(self,extra_snapshot,mesh):
        num = len(extra_snapshot)
        gfu_fom = GridFunction(self.fes)
        gfu_rom = GridFunction(self.fes)
        mean_err = 0
        for i in range(num):
            gfu_fom.vec.data = extra_snapshot[i]
            gfu_rom.vec.data = self.sol_v[i]
            mean_err += Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,mesh)
        mean_err = np.sqrt(mean_err)/num
        print("relative velocity mean error",mean_err)

    def offline_p(self,modes_sup,vecs):
        """vecs : POD velocity modes
        self.vecs : POD pressure modes"""
        #print("pressure dominant modes:",len(self.vecs))

        l= len(vecs)
        l_sup = len(modes_sup)

        b = BilinearForm(trialspace=self.fes,testspace=self.fes_v)
        b += self.fes.TrialFunction() * div(self.fes_v.TestFunction()) * dx
        b.Assemble()
        self.b_red = InnerProduct(modes_sup,b.mat * self.vecs)
        self.stiff_p_red = InnerProduct(modes_sup,self.stiff_v.mat * vecs)
        self.mass_p_red = InnerProduct(modes_sup,self.mass_v.mat * vecs)
        self.stiff_p_red, self.mass_p_red = self.stiff_p_red.NumPy(), self.mass_p_red.NumPy()

        #Convection for pressure
        gfw = GridFunction(self.fes_v)
        conv = BilinearForm(self.fes_v, nonassemble=True)
        u,v = self.fes_v.TnT()
        conv += (Grad(u) * gfw) * v * dx
        tmp = self.vecs[0].CreateVector()
        self.conv_p_red = np.ndarray(shape=(l_sup,l,l),dtype=float)
        for k in range(l):
            gfw.vec.data = vecs[k]
            for j in range(l):
                tmp = conv.mat * vecs[j]
                for i in range(l_sup):
                    self.conv_p_red[i,j,k] = InnerProduct(tmp, modes_sup[i])

    def reconstruct_ROM_p(self, gfu, gfuold):
        """
        Reconstruct the pressure based on residual (based on gfu and gfuold from
        the ROM) on velocity subspace that is orthogonal to the divergence-free
        subspace.
        """

        F_N = -(self.stiff_p_red + self.conv_p_red @ gfu)
        rhs =  F_N @ gfu - (self.mass_p_red @ gfu - self.mass_p_red @ gfuold)/self.timestep
        return np.linalg.solve(self.b_red, rhs)

    def online_p(self,gfu, ntimesteps=10000, timestep = None, highdim_output=True):
        """
        Reconstruct the pressure for the whole time series of velocity ROM
        solutions.
        """
        if timestep == None:
            timestep = self.timestep
        self.gfp_col = []
        self.gfp = np.zeros(len(self.vecs))
        if highdim_output:
            self.sol_p = MultiVector(self.var,0)
        for i in range(1,ntimesteps):
            self.gfp[:] = self.reconstruct_ROM_p(gfu[i], gfu[i-1])
            self.gfp_col.append(self.gfp.copy())
            if highdim_output:
                self.sol_p.Append(self.vecs * Vector(self.gfp))


    def error_p_L2(self):
        num = len(self.snapshot)
        gfu_fom = GridFunction(self.fes)
        gfu_rom = GridFunction(self.fes)
        err = 0
        err_i = 0
        for i in range(num):
            gfu_fom.vec.data = self.snapshot[i]
            gfu_rom.vec.data = self.sol_p[i]
            L2err_i = Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,self.fes.mesh)
            err += L2err_i
            #err += Integrate(InnerProduct(gfu_fom_p -self.gfu_rom_p,gfu_fom_p-self.gfu_rom_p) ,self.fes.mesh)
        err = np.sqrt(err)/num
        print("L2 pressure error",err)
        #print("L2 pressure error",err1)

    def error_p_l2(self):
            num = len(self.snapshot)
            snapshot = np.zeros( (self.fes.ndof ,num))
            solution = np.zeros( (self.fes.ndof ,num))
            norm = 0
            for i in range(num):
                snapshot[:,i] = self.snapshot[i][:]
                solution[:,i] = self.sol_p[i][:]
            diff = snapshot - solution
            for i in range(num):
                norm += np.linalg.norm(diff[:,i]**2)
            err = np.sqrt(norm)/num
            print("l2 pressure error", err)

class dragliftrom:
    def __init__(self, rom_v, rom_p):
        self.rom_v = rom_v
        self.rom_p = rom_p


    def offline_draglift(self,nu):
        """
        offfline_draglift
        compute the functional for the drag and lift computation
        """
        D = self.rom_v.fes.mesh.dim
        n = specialcf.normal(D)
        mesh = self.rom_v.fes.mesh
        self.drag_x_v_vals = []
        self.drag_y_v_vals = []
        self.drag_x_p_vals = []
        self.drag_y_p_vals = []
        gfu_v = GridFunction(self.rom_v.fes)
        for vec in self.rom_v.vecs:
            gfu_v.vec.data = vec
            gradu = Grad(gfu_v)
            c_drag, c_lift = [Integrate ( BoundaryFromVolumeCF( nu * (gradu * n)[i] ) * ds(mesh.Boundaries("cyl")) , mesh ) for i in range(D)]
            c_drag = 2*c_drag  /(1.0*0.05)
            c_lift = -2*c_lift /(1.0*0.05)
            self.drag_x_v_vals.append ( c_drag )
            self.drag_y_v_vals.append ( c_lift)
        gfp = GridFunction(self.rom_p.fes)
        for vec_p in self.rom_p.vecs:
            gfp.vec.data = vec_p
            c_drag, c_lift = [Integrate ( BoundaryFromVolumeCF( - gfp * n[i] ) * ds(mesh.Boundaries("cyl")) , mesh ) for i in range(D)]
            c_drag = 2*c_drag  /(1.0*0.05)
            c_lift = -2*c_lift /(1.0*0.05)
            self.drag_x_p_vals.append ( c_drag )
            self.drag_y_p_vals.append ( c_lift )

    def compute_draglift(self, gfu, gfp):
        '''gfu_L = gfu.tolist()
        gfp_L = gfp.tolist()
        if type(gfp_L) == float:
            gfp_L = [gfp_L]
        if type(gfu_L) == float:
            gfu_L = [gfu_L]'''
        #print("gfu_L = ", gfu_L)
        #print("gfp_L = ", gfp_L)
        #print("self.drag_x_v_vals = ", self.drag_x_v_vals)
        #print("self.drag_x_p_vals = ", self.drag_x_p_vals)
        drag = sum([l*m for l,m in zip(self.drag_x_v_vals,gfu)]) + sum([l*m for l,m in zip(self.drag_x_p_vals,gfp)])
        lift = sum([l*m for l,m in zip(self.drag_y_v_vals,gfu)]) + sum([l*m for l,m in zip(self.drag_y_p_vals,gfp)])
        return drag, lift

    def compute_draglift_time_series(self):
        self.rom_drag = []
        self.rom_lift = []


        for gfu, gfp in zip(self.rom_v.gfu,self.rom_p.gfp_col):

            drag, lift = self.compute_draglift(gfu,gfp)
            self.rom_drag.append(drag)
            self.rom_lift.append(lift)


        return self.rom_drag, self.rom_lift

    def compute_error_draglift(self,ref_drag_x_vals,ref_drag_y_vals):
        mse_d =0
        mse_l = 0
        for i in range(len(ref_drag_x_vals)):
            mse_d += (ref_drag_x_vals[i]-self.rom_drag[i])**2
            mse_l += (ref_drag_y_vals[i]-self.rom_lift[i])**2
        error_d = np.sqrt(mse_d/len(ref_drag_x_vals))
        error_l = np.sqrt(mse_l/len(ref_drag_x_vals))
        return error_d, error_l
