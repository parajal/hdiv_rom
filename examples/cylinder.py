from ngsolve import *
import sys
#sys.path.append('../goe-ngs-templates')
sys.path.append('../goengs')
sys.path.append('../')
sys.path.append('./')
from modules.global_settings import *
from modules.NavierStokes import *

from ngsolve.internal import visoptions
from geometries.cylinder import GenerateMesh


ngsglobals.msg_level = 8
SetHeapSize(100*1000*1000)
mesh = GenerateMesh(maxh=0.07, curve_order=3)

uinflow = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
ubnd_dict = {"inflow" : uinflow }
ubnd = CoefficientFunction([ ubnd_dict[bc] if bc in ubnd_dict.keys() else CoefficientFunction((0,0)) for bc in mesh.GetBoundaries()])


explicit = False
if explicit:
    timestep = 0.00025
    navstokes = NavierStokesExplicit (mesh, nu=0.001, order=3, timestep = timestep,
                              inflow="inflow", outflow="outlet", wall="wall|cyl",
                              ubnd=ubnd)
    navstokes.SolveInitial(uinflow)
else:
    timestep = 0.001
    navstokes = NavierStokes (mesh, nu=0.001, order=3, timestep = timestep,
                              inflow="inflow", outflow="outlet", wall="wall|cyl",
                              ubnd=ubnd, formulation="HdivHDG")
    navstokes.SolveInitial()

Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'
tend = 2
t = 0
with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        navstokes.DoTimeStep()
        # input("")
        t = t+timestep
        Redraw(blocking=False)


        solvecs = MultiVector(navstokes.velocity.vec,0)
        solvecs.Append(navstokes.velocity.vec)
        with TaskManager(pajetrace=100*1000*1000):
            while t < tend:
                print (t)
                navstokes.DoTimeStep()
                solvecs.Append(navstokes.velocity.vec)
                # input("")
                t = t+timestep
                Redraw(blocking=False)
        solvecs.Orthogonalize()
        print(mesh.nv)

        #for i in range(len(solvecs)):
            #navstokes.velocity.vec.data = solvecs[i]
            #Redraw()
            #input("")


        #eigenvalue solver
        from ngsolve import *
        from ngsolve.webgui import Draw
        from netgen.geom2d import unit_square
        import math
        import scipy.linalg
        from scipy import random

        fes = H1(mesh, order=4, dirichlet=".*")
        u = fes.TrialFunction()
        v = fes.TestFunction()

        a = BilinearForm(fes)
        a += 0.0005 * InnerProduct(solvecs,solvecs)* u * v * dx

        pre = Preconditioner(a, "multigrid")

        m = BilinearForm(fes)
        m += u * v * dx

        a.Assemble()
        m.Assemble()

        u = GridFunction(fes)

        r = u.vec.CreateVector()
        w = u.vec.CreateVector()
        Mu = u.vec.CreateVector()
        Au = u.vec.CreateVector()
        Mw = u.vec.CreateVector()
        Aw = u.vec.CreateVector()

        r.FV().NumPy()[:] = random.rand(fes.ndof)
        u.vec.data = Projector(fes.FreeDofs(), True) * r

        num = 5
        u = GridFunction(fes, multidim=num)

        r = u.vec.CreateVector()
        Av = u.vec.CreateVector()
        Mv = u.vec.CreateVector()

        vecs = []
        for i in range(2*num):
            vecs.append (u.vec.CreateVector())

        for v in u.vecs:
            r.FV().NumPy()[:] = random.rand(fes.ndof)
            v.data = Projector(fes.FreeDofs(), True) * r


            asmall = Matrix(2*num, 2*num)
        msmall = Matrix(2*num, 2*num)
        lams = num * [1]

        for i in range(20):

            for j in range(num):
                vecs[j].data = u.vecs[j]
                r.data = a.mat * vecs[j] - lams[j] * m.mat * vecs[j]
                vecs[num+j].data = pre.mat * r

            for j in range(2*num):
                Av.data = a.mat * vecs[j]
                Mv.data = m.mat * vecs[j]
                for k in range(2*num):
                    asmall[j,k] = InnerProduct(Av, vecs[k])
                    msmall[j,k] = InnerProduct(Mv, vecs[k])

            ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
            lams[:] = ev[0:num]
            #print (i, ":", [lam/math.pi**2 for lam in lams])

            for j in range(num):
                u.vecs[j][:] = 0.0
                for k in range(2*num):
                    u.vecs[j].data += float(evec[k,j]) * vecs[k]

        #for i in range(num):
        #Draw(u.vecs[i], mesh, "first node")




        Draw (u.vecs[2])
