import scipy.linalg
from numpy import linalg as LNG
from scipy import random
import scipy.sparse as sp
import numpy as np
from numpy import linalg
import sys
sys.path.append('../goengs')
sys.path.append('../')
sys.path.append('./')
sys.path.append('../tutorials')
from modules.global_settings import *
from modules.NavierStokes import *
import ngsolve.internal
from ngsolve.internal import visoptions
from geometries.cylinder import GenerateMesh
from rapod import *
from netgen import gui
from ngsolve import *
ngsglobals.msg_level = 1
SetHeapSize(100*1000*1000)
mesh = GenerateMesh(maxh=0.035, curve_order=3)
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
    navstokes = NavierStokes(mesh, nu=0.001, order=3, timestep = timestep,
                                  inflow="inflow", outflow="outlet", wall="wall|cyl",
                                  ubnd=ubnd, formulation="HdivHDG", divpenalty = None)
    navstokes.SolveInitial()


Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
Draw (div(navstokes.velocity), mesh, "div_velocity")
visoptions.scalfunction='velocity:0'

gfuinitvec = navstokes.velocity.vec.CreateVector()
gfuinitvec.data = navstokes.velocity.vec
gfuinitvec_p = navstokes.pressure.vec.CreateVector()
gfuinitvec_p.data = navstokes.pressure.vec
#Velocity
V = navstokes.velocity.space
u,v   = V.TnT()
mass_v = BilinearForm(V)
mass_v += u*v * dx
mass_v.Assemble()

stiff_v = BilinearForm(V)
stiff_v += 0.001*InnerProduct(grad(u),grad(v))*dx
stiff_v.Assemble()


#Pressure
Q = navstokes.pressure.space
p,q = Q.TnT()
mass_p = BilinearForm(Q)
mass_p += p*q * dx
mass_p.Assemble()

#Initial energy
initial_velocity_energy = InnerProduct(mass_v.mat * navstokes.velocity.vec, navstokes.velocity.vec)
initial_pressure_energy = InnerProduct(mass_p.mat * navstokes.pressure.vec, navstokes.pressure.vec)

#Global Error bound and relaxation parameter
bound_v = 0.01# * initial_velocity_energy #bound for overall POD
bound_p = 0.01#* initial_pressure_energy #bound for overall POD
relax = 0.5 #weight between performance of POD vs. dimension of reduced space
print("velocity energy",initial_velocity_energy)
print("pressure enrgy",initial_pressure_energy)
# Computing local error bounds for Velocity
nodebound_v = bound_v * np.sqrt(1 - relax**2) ##accuracy on each node except root
rootbound_v = bound_v * relax ##accuracy on root

#Computing local error bounds for Pressure
nodebound_p = bound_p * np.sqrt(1 - relax**2) ##accuracy on each node except root
rootbound_p = bound_p * relax ##accuracy on root
cnt = 0
t = 0
tend = 0.6
tstart = 0.2
#vecs : MultiVector where we store dominant modes
vecs_v = MultiVector(navstokes.velocity.vec,0)
solvecs_v = MultiVector(navstokes.velocity.vec,0)
myrapod_v = rapod(mass_v.mat,vecs_v,nodebound_v,rootbound_v,solvecs_v,navstokes.velocity.vec,timestep,tend)

vecs_p = MultiVector(navstokes.pressure.vec,0)
solvecs_p = MultiVector(navstokes.pressure.vec,0)
myrapod_p = rapod(mass_p.mat,vecs_p,nodebound_p,rootbound_p,solvecs_p,navstokes.pressure.vec,timestep,tend)

solvecs_pod = MultiVector(navstokes.velocity.vec,0)
mypod = pod(mass_v.mat,bound_v,solvecs_pod)

solvecs_pod_p = MultiVector(navstokes.pressure.vec,0)
mypod_p = pod(mass_p.mat,bound_p,solvecs_pod_p)
first = True
with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        navstokes.DoTimeStep()

        if t > tstart:
            if first:

                gfuinitvec.data = navstokes.velocity.vec
                gfuinitvec_p.data = navstokes.pressure.vec
                first = False
            else:

                solvecs_v.Append(navstokes.velocity.vec)
                solvecs_p.Append(navstokes.pressure.vec)

                myrapod_p.Append(navstokes.pressure.vec)
                myrapod_v.Append(navstokes.velocity.vec - gfuinitvec)
                mypod.Append(navstokes.velocity.vec)
                mypod_p.Append(navstokes.pressure.vec)

                cnt = cnt + 1

        t = t+timestep


        Redraw(blocking=False)

print("number of snapshots",cnt)

#POD
#M = 1
#vecs_v = MultiVector(navstokes.velocity.vec,M)
#mypod_v = pod(mass_v.mat,solvecs,vecs_v,M)
#mypod_v.pod_only()
#modes_v = mypod_v.output()
#modes_v.Append(gfuinitvec)
#Number of snapshots

#RAPOD
#Velocity modes
print("The number of dominant velocity modes using POD :",mypod.output())
print("The number of dominant pressure modes using POD :",mypod_p.output())

modes_v = myrapod_v.output()
print("rapod velocity modes:", len(modes_v))

modes_v.Append(gfuinitvec)
l = len(modes_v)

#Pressure modes
modes_p = myrapod_p.output()
l_p = len(modes_p)
print("pressure modes:", l_p)

#Velocity supremizer modes
modes_sup = MultiVector(gfuinitvec,0)
for i in range(l_p):
    navstokes.pressure.vec.data = modes_p[i]
    navstokes.SolveForSupremizer(navstokes.pressure)
    modes_sup.Append(navstokes.gfu.components[0].vec)
    #Redraw()
    #input("")
l_sup = len(modes_sup)

#Offline Phase for Velocity

#Transformation of mass/stiffness
stiff_v_red = InnerProduct(modes_v, stiff_v.mat * modes_v)
mass_v_red = InnerProduct(modes_v, mass_v.mat * modes_v)

#Convection reduced form
gfw = GridFunction(V)
conv = BilinearForm(V, nonassemble=True)
conv += (Grad(u) * gfw) * v * dx

convection=True

tmp = modes_v[0].CreateVector()
if convection:
  conv_v_red = np.ndarray(shape=(l,l,l),dtype=float)
  for k in range(l):
      gfw.vec.data = modes_v[k]
      for j in range(l):
          tmp = conv.mat * modes_v[j]
          for i in range(l):
              conv_v_red[i,j,k] = InnerProduct(tmp, modes_v[i])


#Online phase for velocity
stiff_v_red = stiff_v_red.NumPy()
mass_v_red = mass_v_red.NumPy()

sol_v = MultiVector(navstokes.velocity.vec,0)
gfu = []
gfu_v = np.zeros(l)
gfu_v[0:l-2] = 0
gfu_v[l-1] = 1

for i in range(10000):
    S_N = mass_v_red + stiff_v_red * timestep
    if convection:
        S_N += timestep * conv_v_red @ gfu_v
    S_N_star = S_N[0:l-1,0:l-1]

    F_N = mass_v_red @ gfu_v - S_N[:,l-1]
    F_N_star = F_N[0:l-1]

    gfu_v[0:l-1] =  (np.linalg.solve(S_N_star, F_N_star))
    gfu.append(gfu_v.copy())
    sol_v.Append(modes_v * Vector(gfu_v))

    #Redraw()
    #input("")

#L^2 error for Velocity
num = len(solvecs_v)

gfu_fom = GridFunction(V)
gfu_rom = GridFunction(V)

err = 0
for i in range(num):
     #err_i =  InnerProduct(solvecs_v[i]-sol_v[i],solvecs_v[i]- sol_v[i])
     gfu_fom.vec.data = solvecs_v[i]
     gfu_rom.vec.data = sol_v[i]
     err += Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,mesh)
     #print("i = {0}, L2_err_i = {1}, err_i = {2}".format(i,L2err_i ,err_i ) )

err = np.sqrt(err)/ num
print("L2 velocity error",err)
#Pressure

print("Offline Phase for pressure")

b = BilinearForm(trialspace=Q,testspace=V)
b += Q.TrialFunction() * div(V.TestFunction()) * dx
b.Assemble()

b_red = InnerProduct(modes_sup,b.mat * modes_p)

print("Mixed term")
stiff_p_red = InnerProduct(modes_sup,stiff_v.mat * modes_v)
mass_p_red = InnerProduct(modes_sup,mass_v.mat * modes_v)

gfw = GridFunction(V)
conv = BilinearForm(V, nonassemble=True)
conv += (Grad(u) * gfw) * v * dx

convection=True
conv_p_red = np.ndarray(shape=(l_sup,l,l),dtype=float)
tmp = modes_v[0].CreateVector()
for k in range(l):
    gfw.vec.data = modes_v[k]
    for j in range(l):
        tmp = conv.mat * modes_v[j]
        for i in range(l_sup):
            conv_p_red[i,j,k] = InnerProduct(tmp, modes_sup[i])

gfu_p = np.zeros(l_p)
stiff_p_red = stiff_p_red.NumPy()
mass_p_red = mass_p_red.NumPy()
sol_p = MultiVector(navstokes.pressure.vec,0)

for i in range(1,10000):

    F_N = -(stiff_p_red @ gfu[i] + conv_p_red @ gfu[i])  @ gfu[i]
    rhs = F_N @ gfu[i] + (mass_p_red @ gfu[i] - mass_p_red @ gfu[i-1])/timestep

    gfu_p[:] = np.linalg.solve(b_red, rhs)
    sol_p.Append(modes_p * Vector(gfu_p))


#L^2 error for Pressure
num = len(solvecs_p)
gfu_fom = GridFunction(Q)
gfu_rom = GridFunction(Q)

mean_err_p = 0
for i in range(l_p,num):
     err_i =  InnerProduct(solvecs_p[i]-sol_p[i],solvecs_p[i]- sol_p[i])
     gfu_fom.vec.data = solvecs_p[i]
     gfu_rom.vec.data = sol_p[i]
     L2err_i = Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,mesh)
     #print("i = {0}, L2_err_i = {1}, err_i = {2}".format(i,L2err_i ,err_i ) )
     err += L2err_i

mean_err_p = np.sqrt(err)/ num
print("pressure mean error",mean_err_p)
