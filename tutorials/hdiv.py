import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
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
from solver import *
from netgen import gui
from ngsolve import *
ngsglobals.msg_level = 1
SetHeapSize(100*1000*1000)
mesh = GenerateMesh(maxh=0.035, curve_order=3)
uinflow = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
ubnd_dict = {"inflow" : uinflow }
ubnd = CoefficientFunction([ ubnd_dict[bc] if bc in ubnd_dict.keys() else CoefficientFunction((0,0)) for bc in mesh.GetBoundaries()])

#peter vogel room
'''
import time
from netgen.geom2d import *

start_time = time.time()

# -------------------- BUILD DOMAIN WITH MESH --------------------
geo = SplineGeometry()
pnts = [ (0,0), (0.84,0), (0.84,0.5), (0,0.5), (0,0.455), (0,0.413), (0,0.340), (0,0.280) ]
pnums = [geo.AppendPoint(*p) for p in pnts]
geo.Append( ["line", pnums[0], pnums[1]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[1], pnums[2]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[2], pnums[3]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[3], pnums[4]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[4], pnums[5]], leftdomain=1, rightdomain=0, bc="inflow")
geo.Append( ["line", pnums[5], pnums[6]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[6], pnums[7]], leftdomain=1, rightdomain=0, bc="outlet")
geo.Append( ["line", pnums[7], pnums[0]], leftdomain=1, rightdomain=0, bc="wall")
from ngsolve import *
ngsglobals.msg_level = 1
SetHeapSize(10000000)
comm = mpi_world

realcompile = comm.size == 1

if comm.rank == 0:
    ngmesh = geo.GenerateMesh(maxh=0.035)
    if comm.size > 1:
        ngmesh.Distribute(comm)
else:
    import netgen
    ngmesh = netgen.meshing.Mesh.Receive(comm)
refine = 0
for i in range(refine):
    ngmesh.Refine()
mesh = Mesh(ngmesh)
Draw(mesh)

top = 0.455
bottom = 0.413
Rey = 107.8
vel_in = 0.4
length = top - bottom
nu = vel_in*length/Rey
uinflow = vel_in * CoefficientFunction( (1.5*4*(top-y)*(y-bottom)/(top-bottom)**2, 0) )
ubnd_dict = {"inflow" : uinflow }
ubnd = CoefficientFunction([ ubnd_dict[bc] if bc in ubnd_dict.keys() else CoefficientFunction((0,0)) for bc in mesh.GetBoundaries()])
#end of PeterVogel
'''
#FOM H(div-HDG) Solver
explicit = False
nu = 0.01
timestep = 0.001
if explicit:
    timestep = 0.00025
    navstokes = NavierStokesExplicit (mesh, nu=0.001, order=3, timestep = timestep,
                                  inflow="inflow", outflow="outlet", wall="wall|cyl",
                                  ubnd=ubnd)
    navstokes.SolveInitial(uinflow)
else:
    #timestep = 0.001
    navstokes = NavierStokes(mesh, nu=nu, order=3, timestep = timestep,
                                  inflow="inflow", outflow="outlet", wall="wall|cyl",
                                  ubnd=ubnd, formulation="HdivHDG", divpenalty = None)
    navstokes.SolveInitial()

Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
Draw (div(navstokes.velocity), mesh, "div_velocity")
visoptions.scalfunction='velocity:0'

V = navstokes.velocity.space
Q = navstokes.pressure.space

t = 0
tend = 1
tstart = 0
tol = 0.000001
relax = 0.5
gfuinitvec = navstokes.velocity.vec.CreateVector()
gfuinitvec.data = navstokes.velocity.vec


rom_v = vprom(nu,V,Q,navstokes.velocity.vec,tol,relax,tend,timestep,tstart,
              spacing = 100)
rom_p = vprom(nu,Q,V,navstokes.pressure.vec, tol,relax,tend,timestep,tstart,
              spacing = 100)
drag_x_vals = []
drag_y_vals = []
n = specialcf.normal(mesh.dim)
cnt =0
first = True

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        navstokes.DoTimeStep()
        if t >= tstart:
            if first:
                gfuinitvec.data = navstokes.velocity.vec
                first = False
            else:
                p = navstokes.pressure
                gradu = Grad ( navstokes.gfu.components[0])
                c_drag, c_lift = [Integrate ( BoundaryFromVolumeCF( nu * (gradu * n)[i]  - p * n [i]) * ds(mesh.Boundaries("cyl")) , mesh ) for i in range(2)]
                c_drag = 2 * c_drag/(1.0 * 0.05)
                c_lift = -2*c_lift /(1.0*0.05)
                drag_x_vals . append ( c_drag )
                drag_y_vals . append ( c_lift)
                rom_v.Append_rapod(navstokes.velocity.vec - gfuinitvec,force_compression = (t+timestep >= tend))
                rom_v.Append_pod(navstokes.velocity.vec )
                rom_p.Append_rapod(navstokes.pressure.vec,force_compression = (t+timestep >= tend))
                rom_p.Append_pod(navstokes.pressure.vec )
                cnt = cnt + 1
        t = t+timestep
        Redraw(blocking=False)

print("total snapshots",cnt)
#print("c_lift",drag_y_vals)

velocity_rom = False
if velocity_rom: # *only* velocity
    rom_v.prepare_ROM(gfuinitvec) #for boundary conditions
    rom_v.simulate_ROM()
    rom_v.error_v_L2()
    rom_v.error_v_l2()
else:

    print("The number of dominant velocity modes using POD :",rom_v.output_pod())
    print("The number of dominant velocity modes using RAPOD :",len(rom_v.output_rapod()))
    print("The number of dominant pressure modes using POD :",rom_p.output_pod())
    print("The number of dominant pressure modes using RAPOD :",len(rom_p.output_rapod()))
    rom_v.prepare_ROM(gfuinitvec)
    modes_sup = MultiVector(navstokes.velocity.vec,0)
    for i in range(len(rom_p.vecs)):
        navstokes.pressure.vec.data = rom_p.vecs[i]
        navstokes.SolveForSupremizer(navstokes.pressure)
        modes_sup.Append(navstokes.gfu.components[0].vec)
    rom_p.offline_p(modes_sup,rom_v.vecs) #supremizer and velocity modes
    rom_dl = dragliftrom(rom_v,rom_p)
    rom_dl.offline_draglift(nu)
    rom_v.simulate_ROM()
    rom_p.online_p(rom_v.gfu) # coefficients
    rom_v.error_v_L2()
    rom_p.error_p_L2()
    rom_dl.compute_draglift_time_series()
    rom_drag, rom_lift = rom_dl.compute_draglift_time_series()
    #min and max drag in ROM and FOM
    max_drag = np.amax(rom_dl.rom_drag)
    min_drag = np.amin(rom_dl.rom_drag)
    max_lift = np.amax(rom_dl.rom_lift)
    min_lift = np.amin(rom_dl.rom_lift)
    max_drag_fom = np.amax(drag_x_vals)
    min_drag_fom = np.amin(drag_x_vals)
    max_lift_fom = np.amax(drag_y_vals)
    min_lift_fom = np.amin(drag_y_vals)
    print("max drag in ROM",max_drag)
    print("min drag in ROM", min_drag)
    print("max lift in ROM", max_lift)
    print("min lift in ROM",min_lift)
    print("max drag in FOM",max_drag_fom)
    print("min drag in FOM", min_drag_fom)
    print("max lift in FOM", max_lift_fom)
    print("min lift in FOM",min_lift_fom)
    #print("rom_drag = {}".format(rom_dl.rom_drag[0:999]))
    #input("")
    #print("rom_drag = {}".format(rom_dl.rom_lift[0:999]))
    error_d, error_l = rom_dl.compute_error_draglift(drag_x_vals,drag_y_vals)
    print("error_d = {}, error_l = {}".format(error_d,error_l))

#end of ROM/FOM
'''
#long term behaviour
#FOM H(div-HDG) Solver
explicit = False # if false: implicit time discretization
nu = 0.001 #viscosity
if explicit:
    timestep = 0.00025
    navstokes = NavierStokesExplicit (mesh, nu=0.001, order=3, timestep = timestep,
                                  inflow="inflow", outflow="outlet", wall="wall|cyl",
                                  ubnd=ubnd)
    navstokes.SolveInitial(uinflow)
else:
    timestep = 0.001
    navstokes = NavierStokes(mesh, nu=nu, order= 3, timestep = timestep,
                                  inflow="inflow", outflow="outlet", wall="wall|cyl",
                                  ubnd=ubnd, formulation="HdivHDG", divpenalty = None)
    navstokes.SolveInitial()
Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
Draw (div(navstokes.velocity), mesh, "div_velocity")
visoptions.scalfunction='velocity:0'

#FE Space
V = navstokes.velocity.space
Q = navstokes.pressure.space

t = 0
tend = 5
# for visualization
tstart = 0.2
rom_v_new = vprom(nu,V,Q,navstokes.velocity.vec,tol,relax,tend,timestep,tstart,
              spacing = 100)
rom_p_new = vprom(nu,Q,V,navstokes.pressure.vec,tol,relax,tend,timestep,tstart,
              spacing = 100)
first = True
with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        navstokes.DoTimeStep()
        if t >= tstart:
            if first:
                gfuinitvec.data = navstokes.velocity.vec
                first = False
            else:
                rom_v_new.Append_pod(navstokes.velocity.vec )
                rom_p_new.Append_pod(navstokes.pressure.vec)
        t = t+timestep
        Redraw(blocking=False)


num = len(rom_v_new.snapshot)
print("length of new snapshot",num)
gfu_fom = GridFunction(V)
gfu_rom = GridFunction(V)
err_col = []
for i in range(num):
    gfu_fom.vec.data = rom_v_new.snapshot[i]
    gfu_rom.vec.data = rom_v.sol_v[i]
    err = np.sqrt(Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,mesh))
    err_col.append(err)
print(err_col)'''
