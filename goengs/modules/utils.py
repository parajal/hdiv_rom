from ngsolve import solvers, mpi_world, BaseMatrix

comm = mpi_world
rank = comm.rank
np = comm.size

def mprint(*args,**kwargs):
    if rank == 0:
        print(*args,**kwargs)


def IterativeSolve(mat, rhs, pre = None, freedofs = None, solver_type = "CG", tol = 1e-6, maxsteps = 200,
                   printrates = False, raise_on_failure = False, print_summary = True,
                   report_prefix = "its", tmp = None):
    class Counter:
        def __init__(self):
            self.counter=0
        def increase_counter(self,*args,**kwargs):
            self.counter+=1
    counter = Counter()
    if pre == None:
        raise Exception("if no preconditioner is provided freedofs should be used. Not implemented (TODO!)")
    if solver_type == "CG":
        tmp = solvers.CG(mat=mat, pre=pre, rhs=rhs, tol=tol, maxsteps=maxsteps, callback=counter.increase_counter, printrates=printrates)
    elif solver_type == "GMRes":
        tmp = rhs.CreateVector()
        tmp[:] = 0
        solvers.GMRes(mat, b=rhs, pre=pre, x=tmp, tol=tol, maxsteps=maxsteps, callback=counter.increase_counter, printrates=printrates)
    elif solver_type == "MinRes":
        tmp = solvers.MinRes(mat, rhs=rhs, pre=pre, tol=tol, maxsteps=maxsteps, printrates=printrates)
    else:
        raise Exception("solver type "+solver_type+" unknown.")
    if print_summary:
        mprint("<"+report_prefix+" its: "+str(counter.counter)+" >",end="",flush=True)
    if counter.counter >= maxsteps:
        raise Exception(report_prefix+": Iterative solver did not converge. Stopping the simulation here.")
    return tmp

class IterativeSolver(BaseMatrix):
    def __init__(self, mat, *args, **kwargs):
        super(IterativeSolver, self).__init__()
        self.mat = mat
        self.init_args = args
        self.init_kwargs = kwargs
        self.temp = self.mat.CreateColVector()        
        if not "tmp" in self.init_kwargs:
            self.init_kwargs["tmp"] = self.mat.CreateColVector()
    def Height(self):
        return self.mat.height
    def Width(self):
        return self.mat.width
    def CreateColVector(self):
        return self.mat.CreateColVector()
    def CreateRowVector(self):
        return self.mat.CreateRowVector()
    def Mult(self, b, x):
        x.data = IterativeSolve(self.mat, rhs = b, *self.init_args, **self.init_kwargs)
    def MultAdd(self, scal, b, x):
        x.data += scal * IterativeSolve(self.mat, rhs = b, *self.init_args, **self.init_kwargs)
        
