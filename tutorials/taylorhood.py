from ngsolve import *
from solver import *
# timestepping parameters
timestep = 0.001
tend = 1
# viscosity
nu = 0.001
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
mesh = Mesh( geo.GenerateMesh(maxh=0.07))
mesh.Curve(3)

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


V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = FESpace([V,Q])
u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-8*p*q
a = BilinearForm(X)
a += stokes*dx
a.Assemble()

# nothing here ...
f = LinearForm(X)
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)
velocity = gfu.components[0]
pressure = gfu.components[1]
# parabolic inflow at inlet:
uinflow = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uinflow, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res


# matrix for implicit Euler
mstar = BilinearForm(X)
mstar += SymbolicBFI(u*v + timestep*stokes)
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term
conv = BilinearForm(X, nonassemble = True)
conv += (grad(u) * u) * v * dx

t = 0
tstart = 0
tol = 0.1
Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)
rom_v = vprom(nu,V,Q,velocity.vec,tol,0.5,tend,timestep,tstart,
              spacing = 100)
rom_p = vprom(nu,Q,V,pressure.vec,tol,0.5,tend,timestep,tstart,
              spacing = 100)

# implicit Euler/explicit Euler splitting method:

first = True
gfuinitvec = velocity.vec.CreateVector()
gfuinitvec_p = pressure.vec.CreateVector()
drag_x_vals = []
drag_y_vals = []
n = specialcf.normal(mesh.dim)
cnt = 0
with TaskManager():
    while t < tend:
        conv.Apply (gfu.vec, res)
        res.data += a.mat*gfu.vec
        gfu.vec.data -= timestep * inv * res
        #input("")
        #print("c_lift",drag_y_vals)
        if t >= tstart :
            if first:
                gfuinitvec.data = velocity.vec
                gfuinitvec_p.data = pressure.vec
                first = False
            else:
                p = pressure
                gradu = Grad(velocity)
                c_drag, c_lift = [Integrate ( BoundaryFromVolumeCF( nu * (gradu * n)[i]  - p * n [i]) * ds(mesh.Boundaries("cyl")) , mesh ) for i in range(2)]
                c_drag = 2 * c_drag/(1.0 * 0.05)
                c_lift = -2*c_lift /(1.0*0.05)
                drag_x_vals . append ( c_drag )
                drag_y_vals . append ( c_lift)
                rom_v.Append_rapod(velocity.vec - gfuinitvec,force_compression = (t+timestep >= tend))
                rom_v.Append_pod(velocity.vec )
                rom_p.Append_rapod(pressure.vec,force_compression = (t+timestep >= tend))
                rom_p.Append_pod(pressure.vec )
                cnt += 1
        t = t + timestep
        Redraw()
print("total snapshots",cnt)
#print("c_drag",drag_y_vals)

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
    rom_v.prepare_ROM(gfuinitvec) #for boundary conditions
    #Velocity Supremizer
    modes_sup = MultiVector(velocity.vec,0)
    for i in range(len(rom_p.vecs)):
        pressure.vec.data = rom_p.vecs[i]
        f = LinearForm(X)
        f +=  SymbolicLFI(pressure * X.TestFunction()[-1])
        f.Assemble()
        temp = a.mat.CreateColVector()
        inv = a.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky")
        temp.data = -a.mat * gfu.vec + f.vec
        temp.data = Projector(V.FreeDofs(),True) * temp
        for i in range(2):
            gfu.vec.data = inv * f.vec
            temp.data = -a.mat * gfu.vec + f.vec
            temp.data = Projector(V.FreeDofs(),True) * temp
        modes_sup.Append(gfu.components[0].vec)

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
    #print(len(rom_dl.rom_drag[0:1000]))
    #input("")
    #print("rom_lift = {}".format(rom_lift[0:999]))
    error_d, error_l = rom_dl.compute_error_draglift(drag_x_vals,drag_y_vals)
    print("error_d = {}, error_l = {}".format(error_d,error_l))

'''
#long term behaviour
V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = FESpace([V,Q])
u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-8*p*q
a = BilinearForm(X)
a += stokes*dx
a.Assemble()

# nothing here ...
f = LinearForm(X)
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)
velocity = gfu.components[0]
pressure = gfu.components[1]
# parabolic inflow at inlet:
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()

res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res

# matrix for implicit Euler
mstar = BilinearForm(X)
mstar += SymbolicBFI(u*v + timestep*stokes)
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term
conv = BilinearForm(X, nonassemble = True)
conv += (grad(u) * u) * v * dx
t = 0
tend = 5
# for visualization
tstart = 0.2
gfuinitvec = pressure.vec.CreateVector()
rom_v_new = vprom(nu,V,Q,velocity.vec,tol,0.5,tend,timestep,tstart,
              spacing = 100)
rom_p_new = vprom(nu,Q,V,pressure.vec,tol,0.5,tend,timestep,tstart,
              spacing = 100)
first = True
with TaskManager():
    while t < tend:
        conv.Apply (gfu.vec, res)
        res.data += a.mat*gfu.vec
        gfu.vec.data -= timestep * inv * res
        if t >= tstart:
            if first:
                gfuinitvec.data = pressure.vec
                first = False
            else:
                rom_p_new.Append_pod(pressure.vec)
                #rom_p_new.Append_rapod(pressure.vec)
        t = t + timestep
        Redraw()

num = len(rom_p_new.snapshot)
print("length of new snapshot",num)
gfu_fom = GridFunction(Q)
gfu_rom = GridFunction(Q)
err_col = []
for i in range(num):
    gfu_fom.vec.data = rom_p_new.snapshot[i]
    gfu_rom.vec.data = rom_p.sol_p[i]
    err = np.sqrt(Integrate(InnerProduct(gfu_fom-gfu_rom,gfu_fom-gfu_rom) ,mesh))
    err_col.append(err)
print(err_col)'''
