from netgen.geom2d import SplineGeometry
from ngsolve import Mesh
import math

def GenerateNACA0015Mesh(maxh=0.07,curve_order=3):
    # naca0015
    t = 0.15
    def profile(x):
        return 5*t*(0.2969*math.sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x*x*x-0.1015*x*x*x*x)

    geo = SplineGeometry()
    geo.AddCurve( lambda t : (t,profile(t)), leftdomain=1, rightdomain=0, bc="wall", maxh=0.01)
    geo.AddCurve( lambda t : (t,-profile(t)), leftdomain=0, rightdomain=1, bc="wall",maxh=0.01)
    geo.AddCurve( lambda t : (1, profile(1)*t-profile(1)*(1-t)), leftdomain=0, rightdomain=1, bc="wall", maxh=0.01)
    geo.AddRectangle((-1,-2),(3,2), bcs=["inflow","outflow","inflow","inflow"])
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(curve_order)
    return mesh

if __name__ == "__main__":
    mesh = GenerateNACA0015Mesh()
    Draw(mesh)
