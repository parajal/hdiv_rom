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
geo.Append( ["line", pnums[4], pnums[5]], leftdomain=1, rightdomain=0, bc="inlet")
geo.Append( ["line", pnums[5], pnums[6]], leftdomain=1, rightdomain=0, bc="wall")
geo.Append( ["line", pnums[6], pnums[7]], leftdomain=1, rightdomain=0, bc="outlet")
geo.Append( ["line", pnums[7], pnums[0]], leftdomain=1, rightdomain=0, bc="wall")


from ngsolve import *
ngsglobals.msg_level = 1
SetHeapSize(10000000)



comm = mpi_world

realcompile = comm.size == 1

if comm.rank == 0:
    ngmesh = geo.GenerateMesh(maxh=0.04)
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
DomainVol = Integrate(CoefficientFunction(1),mesh,order=10)
# input("mesh is ready")

# -------------------- PARAMETERS --------------------
print("----------------")
Rey = 107.8
vel_in = 0.4
length = 0.042
nu = vel_in*length/Rey
print("nu=",nu)
order = 4

linsol = "masterinverse"
StatCond = True
ApplyL2 = True
relTOL = 1e-12
pReg = 1e-6
sigma = 5*(order+1)*(order+2)
perrors = False
subdiv = 2 - refine

t = 0
tend = 10
if refine == 0:
    dtinv = 100
if refine == 1:
    dtinv = 250
dt = 1/dtinv
print("dt = ", dt)
nsteps = int(tend/dt+0.5)
print("n.timesteps =",nsteps)
dt = tend / nsteps
# print("dt_used = ", dt)

# samples for projection output (5 samples per [0,1]):
isamples = [i for i in range(0,nsteps+1,int(nsteps/(5*tend)))]
# print("isamples: ",isamples)

# -------------------- FINITE ELEMENT SPACES --------------------
V1 = HDiv(mesh, order=order, hodivfree=True, dirichlet="wall|inlet")
V2 = TangentialFacetFESpace(mesh, order=order, dirichlet="wall|inlet", highest_order_dc=True)
Q = L2(mesh, order=0, lowest_order_wb=True)
V = FESpace([V1,V2,Q], dgjumps=False)
if ApplyL2 == True:
    VApply = VectorL2(mesh, order=order, dgjumps=False)
else:
    VApply = V.components[0]

# print("V1.ndof =",V.components[0].ndof)
# print("V1.ndof.free =",sum(V.components[0].FreeDofs(False)))
# print("V1.ndof.sc.free =",sum(V.components[0].FreeDofs(True)))
# print("V2.ndof =",V.components[1].ndof)
# print("V2.ndof.free =",sum(V.components[1].FreeDofs(False)))
# print("V2.ndof.sc.free =",sum(V.components[1].FreeDofs(True)))
# print("Q.ndof =",V.components[2].ndof)
# print("Q.ndof.free =",sum(V.components[2].FreeDofs(False)))
# print("Q.ndof.sc.free =",sum(V.components[2].FreeDofs(True)))
# print("V.ndof =",V.ndof)
# print("V.ndof.free =",sum(V.FreeDofs(False)))
# print("V.ndof.sc.free =",sum(V.FreeDofs(True)))

# -------------------- DEFINITIONS --------------------
def ABS(scalar):
    return IfPos(scalar,scalar,-scalar)

def GradHdiv2D(vec):
    return CoefficientFunction( (grad(vec),), dims=(2,2) ).trans

def GradH12D(vec):
    return CoefficientFunction( (grad(vec),), dims=(2,2) )

def VortHdiv2D(vec):
    return grad(vec)[1]-grad(vec)[2]

def VortH12D(vec):
    return grad(vec)[2]-grad(vec)[1]

def Grad2D(vec):
    return GradHdiv2D(vec)
    # return GradH12D(vec)

def Vort2D(vec):
    return VortHdiv2D(vec)
    # return VortH12D(vec)

def GradApply2D(vec):
    if ApplyL2 == True:
        return GradH12D(vec)
    else:
        return GradHdiv2D(vec)

def tang(vec):
    return vec - (vec*n)*n

# ---------------- IC + DIRICHLET BC -------------
gfu = GridFunction(V)
gfuApply = GridFunction(VApply)
vel = gfu.components[0]
pre = gfu.components[1]

gD = GridFunction(VectorH1(mesh, order=1, dirichlet="inlet"))
with TaskManager():
    gD.Set(CoefficientFunction( (vel_in, 0.0) ), definedon=mesh.Boundaries("inlet"))
    Draw(gD,mesh,"gD")
    vel.Set(gD, definedon=mesh.Boundaries("inlet"))

# -------------------- FINITE ELEMENT METHOD --------------------
(u,uhat,p),(v,vhat,q) = V.TnT()
uApply,vApply = VApply.TnT()

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

jmp_u = tang(u-uhat)
jmp_v = tang(v-vhat)

jmp_uApply = uApply-uApply.Other()
jmp_vApply = vApply-vApply.Other()
avg_vApply = 0.5*( vApply + vApply.Other() )

StokesVOL = nu*InnerProduct(Grad2D(u),Grad2D(v)) - div(u)*q - div(v)*p
StokesRegVOL = -pReg*p*q
StokesElementBoundaries = - nu*InnerProduct(Grad2D(u)*n,jmp_v) - nu*InnerProduct(jmp_u,Grad2D(v)*n) + nu*(sigma/h)*InnerProduct(jmp_u,jmp_v)
MassVOL = u*v
ConvVOL = InnerProduct(GradApply2D(uApply)*uApply,vApply)
ConvInnerFacet = - InnerProduct(uApply,n)*InnerProduct(jmp_uApply,avg_vApply)
ConvInnerFacet += 0.5*ABS(InnerProduct(uApply,n))*InnerProduct(jmp_uApply,jmp_vApply)

# Discrete Stokes term:
a = BilinearForm(V, symmetric=True, nonsym_storage=True)
a += SymbolicBFI( StokesVOL )
a += SymbolicBFI( StokesElementBoundaries, element_boundary=True )
with TaskManager():
    a.Assemble()

# Facet based convective term:
conv = BilinearForm(VApply, nonassemble=True)
conv += SymbolicBFI( ConvVOL.Compile(realcompile) ).SetIntegrationRule(TRIG,IntegrationRule(TRIG,3*order-1))
conv += SymbolicBFI( ConvInnerFacet.Compile(realcompile), skeleton=True ).SetIntegrationRule(SEGM,IntegrationRule(SEGM,3*order))

if ApplyL2 == True:
    bfmixed = BilinearForm(trialspace=V, testspace=VApply, nonassemble=True)
    bfmixed += SymbolicBFI( (vApply*u).Compile(realcompile) )
    bfmixedT = BilinearForm(trialspace=VApply, testspace=V, nonassemble=True)
    bfmixedT += SymbolicBFI( (uApply*v).Compile(realcompile) )

# System matrix for RK2:
gamma = 1 - sqrt(0.5)
delta = 1 - 1/(2*gamma)

mstarReg = BilinearForm(V, symmetric=True, nonsym_storage=True, eliminate_internal=StatCond, store_inner=False)
mstarReg += SymbolicBFI( MassVOL + dt*gamma*(StokesVOL+StokesRegVOL) )
mstarReg += SymbolicBFI( dt*gamma*StokesElementBoundaries, element_boundary=True )

mstarPre = Preconditioner(mstarReg, "bddc") # , usehypre=True)

with TaskManager():
    mstarReg.Assemble()
    invMstarPC = mstarReg.mat.Inverse(V.FreeDofs(coupling=StatCond), inverse=linsol)
    # invMstarPC = CGSolver(mstarReg.mat, mstarPre.mat, precision=1e-6, maxsteps=1000, printrates=mpi_world.rank==0)
mstar = BilinearForm(V, symmetric=True, nonsym_storage=True, eliminate_internal=StatCond)
mstar += SymbolicBFI( MassVOL + dt*gamma*StokesVOL )
mstar += SymbolicBFI( dt*gamma*StokesElementBoundaries, element_boundary=True )
with TaskManager():
    mstar.Assemble()
    # print("Mstar.nnz.sc =",mstar.mat.nze)
    # if StatCond == True:
    #     print("Mstar.InnerSolve.nnz.sc =",mstar.inner_solve.nze)
    #     print("Mstar.HarmonicExtension.nnz.sc =",mstar.harmonic_extension.nze)

# Right-hand side forcing term:
f = LinearForm(V)
with TaskManager():
    f.Assemble()

print("----------------")
# ------------------ SETTING VARIABLES, VISUALISATION --------------------
MagVel = Norm(vel)
gradvel = Grad2D(vel)
omega = Vort2D(vel)

Draw(div(vel),mesh,"div-vel")
Draw(pre,mesh,"pre")
Draw(vel, mesh, "vel")
Draw(omega, mesh, "vort")
Draw(MagVel,mesh,"vel-mag")

# # -------------------- OUTPUT --------------------
name_string = "-HDivHDG-order"+str(order)+"-refine"+str(refine)+"-dtinv"+str(dtinv)
# filename = "2D-PeterVogel"+name_string+".txt"
# NormFile = open(filename, "w")
# NormFile.write(
#     "#t"+"\t"+
#     "||uh||^2/2|Om|"+"\t"+
#     "||omh||^2/2|Om|"+"\t"+
#     "||div_uh||/|Om|"+"\t"
#     "\n")
# NormFile.close()
# def ComputeOutput():
#     NormFile = open(filename, "a")
#     ekin = (1/(2*DomainVol))*Integrate(InnerProduct(vel,vel),mesh,order=2*order)
#     ens = (1/(2*DomainVol))*Integrate(InnerProduct(omega,omega),mesh,order=2*(order-1))
#     diverr = (1/DomainVol)*sqrt(Integrate(div(vel)*div(vel),mesh,order=2*(order-1)))
#     NormFile.write(
#         str(format(t,".3f"))+"\t"+
#         str(format(ekin,".8e"))+"\t"+
#         str(format(ens,".8e"))+"\t"+
#         str(format(diverr,".8e"))+"\t"+
#         "\n")
#     NormFile.close()

if comm.size > 1:
    mpi = "MPI"
else:
    mpi = "serial"

vtk = VTKOutput(ma=mesh,
                coefs=[vel,omega,],
                names=["velocity","vorticity"],
                filename="VTK-"+mpi+"/2DPV_"+str(comm.rank),
                subdivision=subdiv
                )

with TaskManager():
    # ComputeOutput()
    vtk.Do()
    vtk.Do()

# -------------------- TIME STEPPING --------------------
convvec = gfuApply.vec.CreateVector()
resvec = gfuApply.vec.CreateVector()
convvec1 = gfu.vec.CreateVector()
convvec2 = gfu.vec.CreateVector()
diffvec1 = gfu.vec.CreateVector()
diffvec2 = gfu.vec.CreateVector()
rhs = f.vec.CreateVector()
rhs_corr = f.vec.CreateVector()
u1 = f.vec.CreateVector()
w = f.vec.CreateVector()

convvec1[:] = 0
convvec2[:] = 0

def ApplyConv(vecHDiv,resHDiv):
    if ApplyL2 == True:
        bfmixed.Apply(vecHDiv,convvec)
        VApply.SolveM(convvec,CoefficientFunction(1))
        conv.Apply(convvec,resvec)
        VApply.SolveM(resvec,CoefficientFunction(1))
        bfmixedT.Apply(resvec,resHDiv)
    else:
        conv.Apply(vecHDiv,resHDiv)

i = 0
# input("start time stepping")
with TaskManager():
    for i in range(0,nsteps):
        t = round( (i+1)*dt, 12 )
        print('\rNon-dimensional t={:1.8f}'.format(t) , end="")

        ApplyConv(gfu.vec,convvec1)
        diffvec1.data = a.mat*gfu.vec
        rhs.data =  (convvec1-f.vec) + diffvec1
        if StatCond == True:
            rhs_corr.data = mstar.harmonic_extension_trans*rhs
            rhs.data += rhs_corr
        w.data = solvers.PreconditionedRichardson(mstar, rhs, invMstarPC, printing=perrors, tol=relTOL, maxit=5)
        if StatCond == True:
            w.data += mstar.inner_solve*rhs
            w.data += mstar.harmonic_extension*w
        u1.data = gfu.vec - dt*gamma*w

        ApplyConv(u1,convvec2)
        diffvec2.data = a.mat*u1
        rhs.data = delta*(convvec1-f.vec) + (1-delta)*(convvec2-f.vec) + gamma*(diffvec1) + (1-gamma)*(diffvec2)
        if StatCond == True:
            rhs_corr.data = mstar.harmonic_extension_trans*rhs
            rhs.data += rhs_corr
        w.data = solvers.PreconditionedRichardson(mstar, rhs, invMstarPC, printing=perrors, tol=relTOL, maxit=5)
        if StatCond == True:
            w.data += mstar.inner_solve*rhs
            w.data += mstar.harmonic_extension*w
        gfu.vec.data -= dt*w

        if i+1 in isamples:
            # ComputeOutput()
            vtk.Do()
            Redraw(blocking=True)
            # input("proceed")

with TaskManager():
    vtk.Do()

end_time = time.time() - start_time
print("\n ---------- Total time: %s ----------" % time.strftime("%d:%H:%M:%S", time.gmtime(end_time)) )
